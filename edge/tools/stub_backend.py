#!/usr/bin/env python3
"""Stand-in for the laptop backend, so the Jetson edge can be tested before backend/ exists.

NOT the real backend: no fall logic, no alerts, no WebSocket, no storage. It only:
  * accepts POST /ingest/observations (ObservationBatch) and POST /ingest/heartbeat (Heartbeat)
  * validates every message against the shared contract (hearth_schema), answering 422 with the
    reason when one doesn't match - the edge client logs that as REJECTED
  * prints a one-line summary per second per source, and marks a source OFFLINE after 3 s
    without any message (the README's rule)
  * GET / returns the same state as JSON

Run on the laptop from the repo root (listens on all interfaces so the Jetson can reach it):
    .venv/bin/python edge/tools/stub_backend.py [--port 8000] [--save recordings/stub.jsonl]
"""

import argparse
import json
from datetime import datetime
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from hearth_schema import Heartbeat, ObservationBatch
    from pydantic import ValidationError
except ImportError:  # still usable, just without contract checks
    Heartbeat = ObservationBatch = None
    ValidationError = ValueError

OFFLINE_AFTER_S = 3.0


def emit(msg):
    """Print one line. On a terminal end it with \\r\\n: when live_test.sh runs `ssh -tt` the
    terminal is in raw mode, and a bare \\n would make each line start where the last ended."""
    sys.stdout.write(msg + ("\r\n" if sys.stdout.isatty() else "\n"))
    sys.stdout.flush()


def capture_age_ms(timestamp, now):
    """How long ago (ms) the observation's frame was captured, by the two machines' clocks.

    Includes any clock difference between the Jetson and this laptop (both normally NTP-synced).
    """
    try:
        t = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None
    return (now - t) * 1000.0


def source_online(last_seen, now, timeout=OFFLINE_AFTER_S):
    return last_seen is not None and now - last_seen <= timeout


class State(object):
    def __init__(self, save_path=None):
        self.lock = threading.Lock()
        self.sources = {}  # source -> dict
        self.invalid = 0
        self.save = open(save_path, "a") if save_path else None

    def _src(self, name, source_type):
        s = self.sources.get(name)
        if s is None:
            s = self.sources[name] = {"source_type": source_type, "last_seen": None,
                                      "online": False, "observations": 0, "heartbeats": 0,
                                      "window_obs": 0, "latest": None, "heartbeat": None,
                                      "last_face": None, "window_lag": []}
        return s

    def add_observations(self, observations):
        now = time.time()
        with self.lock:
            for o in observations:
                s = self._src(o["source"], o["source_type"])
                s["last_seen"] = now
                s["observations"] += 1
                s["window_obs"] += 1
                s["latest"] = o
                lag = capture_age_ms(o.get("timestamp"), now)
                if lag is not None:
                    s["window_lag"].append(lag)
                if o.get("face_present") is not None:  # null = no face check on that frame
                    s["last_face"] = (o["face_present"], o.get("face_confidence"), now)
                if self.save:
                    self.save.write(json.dumps(o) + "\n")
            if self.save:
                self.save.flush()

    def add_heartbeat(self, hb):
        with self.lock:
            s = self._src(hb["source"], hb["source_type"])
            s["last_seen"] = time.time()
            s["heartbeats"] += 1
            s["heartbeat"] = hb

    def snapshot(self):
        now = time.time()
        with self.lock:
            out = {"invalid": self.invalid, "sources": {}}
            for name, s in self.sources.items():
                out["sources"][name] = {
                    "source_type": s["source_type"],
                    "online": source_online(s["last_seen"], now),
                    "seconds_since_last_message": None if s["last_seen"] is None else round(now - s["last_seen"], 1),
                    "observations": s["observations"], "heartbeats": s["heartbeats"],
                    "latest_observation": s["latest"], "latest_heartbeat": s["heartbeat"],
                    "last_face_result": None if s["last_face"] is None else {
                        "present": s["last_face"][0], "confidence": s["last_face"][1],
                        "age_s": round(now - s["last_face"][2], 1)},
                }
            return out


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reject(self, detail):
            with state.lock:
                state.invalid += 1
            emit("INVALID %s: %s" % (self.path, detail))
            self._reply(422, {"detail": detail})

        def do_GET(self):
            if self.path == "/":
                self._reply(200, state.snapshot())
            else:
                self._reply(404, {"detail": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            if len(raw) < length or not raw:
                return self._reject("empty or cut-off request body (%d of %d bytes); the sender "
                                    "probably disconnected mid-request" % (len(raw), length))
            try:
                body = json.loads(raw)
            except ValueError as e:
                return self._reject("body is not JSON: %s" % e)
            try:
                if self.path == "/ingest/observations":
                    if ObservationBatch is not None:
                        ObservationBatch.model_validate(body)
                    observations = body["observations"]
                    state.add_observations(observations)
                    return self._reply(200, {"accepted": len(observations)})
                if self.path == "/ingest/heartbeat":
                    if Heartbeat is not None:
                        Heartbeat.model_validate(body)
                    state.add_heartbeat(body)
                    return self._reply(200, {"accepted": 1})
            except (ValidationError, KeyError, TypeError) as e:
                return self._reject(str(e).replace("\n", " | ")[:800])
            self._reply(404, {"detail": "unknown path %s" % self.path})

        def log_message(self, *args):
            pass

    return Handler


def make_server(state, host="0.0.0.0", port=8000):
    return ThreadingHTTPServer((host, port), make_handler(state))


def _fmt(v, fmt):
    return "-" if v is None else fmt % v


def summary_line(name, s, window_obs, lag_ms=None):
    if not s["online"]:
        since = s["seconds_since_last_message"]
        return "%s OFFLINE (no data for %s s)" % (name, _fmt(since, "%.1f"))
    o = s["latest_observation"] or {}
    f = o.get("features") or {}
    hb = s["latest_heartbeat"] or {}
    lf = s.get("last_face_result")
    if lf is None:
        face = "- (no check yet)"
    else:  # face checks run ~2/s, so show the last real result, not the null in between
        face = "%s%s (%.1f s ago)" % (str(lf["present"]).lower(),
                                     "" if lf["confidence"] is None else " %.2f" % lf["confidence"],
                                     lf["age_s"])
    return ("%s online | obs/s %d | lag %s ms | person %s | torso %s head %s hip %s aspect %s vel %s | face %s"
            " | hb fps %s inf %s ms mem %s MB" % (
                name, window_obs, _fmt(lag_ms, "%.0f"), "yes" if o.get("person_detected") else "no",
                _fmt(f.get("torso_angle_deg"), "%.0f"), _fmt(f.get("head_y"), "%.2f"),
                _fmt(f.get("hip_y"), "%.2f"), _fmt(f.get("bbox_aspect"), "%.2f"),
                _fmt(f.get("vertical_velocity"), "%+.2f"),
                face,
                _fmt(hb.get("fps"), "%.1f"), _fmt(hb.get("inference_ms"), "%.0f"),
                _fmt(hb.get("mem_used_mb"), "%.0f")))


def print_loop(state, stop):
    was_online = {}
    while not stop.wait(1.0):
        snap = state.snapshot()
        with state.lock:
            windows = dict((n, s["window_obs"]) for n, s in state.sources.items())
            lags = dict((n, sum(s["window_lag"]) / len(s["window_lag"]) if s["window_lag"] else None)
                        for n, s in state.sources.items())
            for s in state.sources.values():
                s["window_obs"] = 0
                s["window_lag"] = []
        stamp = time.strftime("%H:%M:%S")
        for name, s in sorted(snap["sources"].items()):
            if was_online.get(name) != s["online"]:
                emit("[%s] %s -> %s" % (stamp, name, "source_online" if s["online"] else "source_offline"))
                was_online[name] = s["online"]
            emit("[%s] %s" % (stamp, summary_line(name, s, windows.get(name, 0), lags.get(name))))
        if snap["invalid"]:
            emit("[%s] invalid messages so far: %d" % (stamp, snap["invalid"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stand-in Hearth backend for testing the edge.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--save", default=None, help="append accepted observations to this JSONL file")
    args = ap.parse_args(argv)

    if ObservationBatch is None:
        print("WARNING: hearth_schema not installed; messages are NOT checked against the contract.\n"
              "         Run with the repo's .venv (pip install -e shared/python).", file=sys.stderr)
    state = State(args.save)
    server = make_server(state, args.host, args.port)
    stop = threading.Event()
    threading.Thread(target=print_loop, args=(state, stop), daemon=True).start()
    emit("stub backend (NOT the real backend) listening on %s:%d - Ctrl+C to stop"
         % (args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

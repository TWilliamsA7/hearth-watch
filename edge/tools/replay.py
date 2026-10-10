#!/usr/bin/env python3
"""Replays recorded observations into a backend, clearly labelled as replay.

Every message is rewritten before sending:
    source      -> "replay-01" (never the live camera's name, so their online status can't mix)
    source_type -> "replay"    (README invariant: every record says where it came from)
    run_id      -> a new id per playback pass (also on heartbeats), so repeated runs of the same
                   clip stay separate in the backend's history and evaluation
    obs_id, timestamp -> fresh ids and the current time, so the backend's timing works normally
Use --speed 1 when comparing alert timing: faster playback shortens time spent on the floor.
Everything else (keypoints, features, face_present) is sent exactly as recorded. The original
spacing between observations is kept (scaled by --speed). Heartbeats are sent while playing and
stop afterwards, so the backend shows the replay source going offline when playback ends.

Standard library only (works on the laptop or the Jetson). From the repo root on the laptop:
    .venv/bin/python edge/tools/replay.py recordings/lie_down-20261010-031500.jsonl
    .venv/bin/python edge/tools/replay.py recordings/*.jsonl --speed 2
    .venv/bin/python edge/tools/replay.py recordings/sit-*.jsonl --loop     # Ctrl+C to stop
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hearth_edge import contract  # noqa: E402
from hearth_edge.client import EdgeClient  # noqa: E402

REPLAY_SOURCE = "replay-01"


def load_recording(path):
    """Observations from a JSONL recording, in file order. Unreadable lines are skipped."""
    observations = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obs = json.loads(line)
                contract.parse_utc_iso(obs["timestamp"])
            except (ValueError, KeyError, TypeError) as e:
                sys.stderr.write("%s:%d skipped (%s)\n" % (path, n, e))
                continue
            observations.append(obs)
    return observations


def schedule(observations, speed=1.0):
    """Seconds after playback start at which each observation is sent."""
    if not observations:
        return []
    t0 = contract.parse_utc_iso(observations[0]["timestamp"])
    return [max(0.0, (contract.parse_utc_iso(o["timestamp"]) - t0) / speed) for o in observations]


def new_run_id(source, n=1, clock=time.time):
    """e.g. replay-01-20261010-043512-1: unique per playback pass."""
    return "%s-%s-%d" % (source, time.strftime("%Y%m%d-%H%M%S", time.localtime(clock())), n)


def rewrite_for_replay(obs, obs_id, ts, source=REPLAY_SOURCE, run_id=None):
    out = dict(obs)
    out["obs_id"] = obs_id
    out["timestamp"] = contract.utc_iso(ts)
    out["source"] = source
    out["source_type"] = "replay"
    out["scenario_id"] = None  # recordings are live captures, never simulator scenarios
    if run_id is not None:
        out["run_id"] = run_id
    else:
        out.pop("run_id", None)
    return out


class PlaybackStats(object):
    """Heartbeat data while playing; None (heartbeats pause) when not."""

    def __init__(self):
        self.playing = False
        self.fps = None

    def snapshot(self):
        if not self.playing:
            return None
        return {"fps": self.fps, "inference_ms": None, "mem_used_mb": None}


def recording_rate(observations):
    if len(observations) < 2:
        return None
    span = (contract.parse_utc_iso(observations[-1]["timestamp"])
            - contract.parse_utc_iso(observations[0]["timestamp"]))
    return (len(observations) - 1) / span if span > 0 else None


def play(client, observations, ids, speed=1.0, source=REPLAY_SOURCE, run_id=None,
         clock=time.time, sleep=time.sleep):
    """Submit each observation at its scheduled time. Returns how many were submitted."""
    t0 = clock()
    sent = 0
    for obs, offset in zip(observations, schedule(observations, speed)):
        wait = t0 + offset - clock()
        if wait > 0:
            sleep(wait)
        client.submit(rewrite_for_replay(obs, ids.next(), clock(), source, run_id))
        sent += 1
    return sent


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay recorded observations as source_type 'replay'.")
    ap.add_argument("files", nargs="+", help="recordings/*.jsonl files, played in order")
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--source", default=REPLAY_SOURCE)
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    ap.add_argument("--loop", action="store_true", help="repeat until Ctrl+C")
    args = ap.parse_args(argv)
    if args.speed <= 0:
        ap.error("--speed must be positive")

    recordings = [(path, load_recording(path)) for path in args.files]
    recordings = [(p, obs) for p, obs in recordings if obs]
    if not recordings:
        sys.stderr.write("nothing to replay\n")
        return 1

    room = recordings[0][1][0].get("room_id", "room-01")
    stats = PlaybackStats()
    client = EdgeClient(args.backend, args.source, room, stats=stats, source_type="replay",
                        idle_reason="replay finished")
    ids = contract.ObservationIds(args.source)
    client.start()
    total = 0
    passes = 0
    try:
        while True:
            for path, observations in recordings:
                passes += 1
                run_id = new_run_id(args.source, passes)
                client.run_id = run_id
                duration = schedule(observations, args.speed)[-1]
                print("replaying %s: %d observations over %.1f s (x%g) as %s/replay, run %s"
                      % (os.path.basename(path), len(observations), duration, args.speed,
                         args.source, run_id), flush=True)
                rate = recording_rate(observations)
                stats.fps = rate * args.speed if rate else None
                stats.playing = True
                total += play(client, observations, ids, args.speed, args.source, run_id)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stats.playing = False
        end = time.time() + 5.0
        while client.queued() and time.time() < end and client.online is not False:
            time.sleep(0.05)
        client.stop()
    c = client.counts
    print("done: submitted %d | sent %d dropped %d rejected %d | backend %s"
          % (total, c["sent"], c["dropped"], c["rejected"],
             {None: "not reached", True: "online", False: "OFFLINE"}[client.online]), flush=True)
    return 0 if c["rejected"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

"""Sends Observations and Heartbeats to the laptop backend. Standard library only (Python 3.6).

The capture loop never waits on the network: submit() only appends to an in-memory queue.
Two background threads do the sending:
  * sender    - every ~0.15 s POSTs queued observations as one batch to /ingest/observations
  * heartbeat - every 1 s POSTs a Heartbeat to /ingest/heartbeat

Failure handling:
  * backend unreachable / 5xx -> keep the data, retry with backoff (max 2 s), recover on its own
  * backend rejects (4xx)     -> drop that batch and log the reason; resending can't succeed and
                                 it means our messages and the contract disagree
  * long outage               -> only the newest `queue_max` observations (~10 s) are kept
  * no camera frames          -> heartbeats pause, so the backend marks this source offline
                                 ("silence is not safety": a dead camera must not look fine)
"""

import json
import sys
import threading
import time
from collections import deque
from urllib import error as urlerror
from urllib import request as urlrequest

from hearth_edge import contract


def _default_log(msg):
    try:  # never let a closed stderr (ssh gone) kill a sender thread
        sys.stderr.write("[edge-client %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        sys.stderr.flush()
    except (IOError, OSError, ValueError):
        pass


def read_mem_used_mb(path="/proc/meminfo"):
    """System memory in use (MemTotal - MemAvailable) in MB; the Jetson shares it with the GPU."""
    try:
        vals = {}
        with open(path) as fh:
            for line in fh:
                key, rest = line.split(":", 1)
                vals[key] = int(rest.split()[0])
        return (vals["MemTotal"] - vals["MemAvailable"]) / 1024.0
    except (IOError, OSError, KeyError, ValueError):
        return None


class FrameStats(object):
    """Rolling frame rate and inference time, written by the capture loop, read by heartbeats.

    snapshot() returns None when no frame arrived within `stale_s`; the heartbeat thread then
    stays silent so the backend sees this source go offline.
    """

    def __init__(self, window_s=2.0, stale_s=2.0, clock=time.time, mem_reader=read_mem_used_mb):
        self.window_s = window_s
        self.stale_s = stale_s
        self.clock = clock
        self.mem_reader = mem_reader
        self._lock = threading.Lock()
        self._frames = deque()  # (t, inference_ms)

    def record(self, inference_ms, t=None):
        t = self.clock() if t is None else t
        with self._lock:
            self._frames.append((t, inference_ms))
            self._trim(t)

    def _trim(self, now):
        while self._frames and now - self._frames[0][0] > self.window_s:
            self._frames.popleft()

    def snapshot(self):
        now = self.clock()
        with self._lock:
            self._trim(now)
            frames = list(self._frames)
        if not frames or now - frames[-1][0] > self.stale_s:
            return None
        fps = None
        if len(frames) >= 2 and frames[-1][0] > frames[0][0]:
            fps = (len(frames) - 1) / (frames[-1][0] - frames[0][0])
        times = [ms for _, ms in frames if ms is not None]
        return {
            "fps": fps,
            "inference_ms": sum(times) / len(times) if times else None,
            "mem_used_mb": self.mem_reader(),
        }


class EdgeClient(object):
    def __init__(self, backend_url, source, room_id, stats=None, source_type="edge",
                 observations_path="/ingest/observations", heartbeat_path="/ingest/heartbeat",
                 batch_interval_s=0.15, heartbeat_interval_s=1.0, timeout_s=2.0,
                 queue_max=100, max_batch=50, max_backoff_s=2.0, log=_default_log,
                 clock=time.time, idle_reason="no camera frames"):
        self.base = backend_url.rstrip("/")
        self.source = source
        self.room_id = room_id
        self.stats = stats
        self.source_type = source_type
        self.observations_path = observations_path
        self.heartbeat_path = heartbeat_path
        self.batch_interval_s = batch_interval_s
        self.heartbeat_interval_s = heartbeat_interval_s
        self.timeout_s = timeout_s
        self.queue_max = queue_max
        self.max_batch = max_batch
        self.max_backoff_s = max_backoff_s
        self.log = log
        self.clock = clock
        self.idle_reason = idle_reason  # why heartbeats paused, for the log line

        self._queue = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self.online = None          # None = not tried yet, True/False after the first attempt
        self._heartbeat_paused = False
        self.counts = {"sent": 0, "dropped": 0, "rejected": 0, "heartbeats": 0}

    # --- called from the capture loop -------------------------------------------------------

    def submit(self, observation):
        """Queue one observation. Never blocks on the network."""
        with self._lock:
            self._queue.append(observation)
            self._drop_oldest_over_limit()

    def _drop_oldest_over_limit(self):
        while len(self._queue) > self.queue_max:
            self._queue.popleft()
            self.counts["dropped"] += 1

    def start(self):
        for target, name in ((self._send_loop, "edge-sender"), (self._heartbeat_loop, "edge-heartbeat")):
            t = threading.Thread(target=target, name=name)
            t.daemon = True
            t.start()
            self._threads.append(t)

    def stop(self, join_timeout_s=3.0):
        """Stop both threads after one last attempt to send what's queued."""
        self._stop.set()
        for t in self._threads:
            t.join(join_timeout_s)
        self._threads = []

    def queued(self):
        with self._lock:
            return len(self._queue)

    # --- HTTP -------------------------------------------------------------------------------

    def _post(self, path, payload):
        """POST JSON. Returns (status, body); status is None when the backend wasn't reached."""
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(self.base + path, data=data,
                                 headers={"Content-Type": "application/json"})
        try:
            resp = urlrequest.urlopen(req, timeout=self.timeout_s)
            try:
                return resp.getcode(), resp.read().decode("utf-8", "replace")
            finally:
                resp.close()
        except urlerror.HTTPError as e:
            body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
            return e.code, body
        except Exception as e:  # URLError, timeout, connection reset, ...
            return None, "%s: %s" % (type(e).__name__, e)

    def _set_online(self, online, detail=""):
        if online == self.online:
            return
        if online:
            self.log("backend reachable at %s" % self.base)
        else:
            self.log("backend unreachable (%s); buffering the newest %d observations"
                     % (detail, self.queue_max))
        self.online = online

    # --- background threads -----------------------------------------------------------------

    def _take_batch(self):
        with self._lock:
            n = min(self.max_batch, len(self._queue))
            return [self._queue.popleft() for _ in range(n)]

    def _requeue_front(self, batch):
        """Put an unsent batch back in front, keeping only the newest queue_max overall."""
        with self._lock:
            combined = list(batch) + list(self._queue)
            overflow = max(0, len(combined) - self.queue_max)
            self.counts["dropped"] += overflow
            self._queue = deque(combined[overflow:])

    def _send_once(self):
        """Send one batch. Returns True if the backend was reached (or nothing to send)."""
        batch = self._take_batch()
        if not batch:
            return True
        status, body = self._post(self.observations_path, {"observations": batch})
        if status is not None and 200 <= status < 300:
            self.counts["sent"] += len(batch)
            self._set_online(True)
            return True
        if status is not None and 400 <= status < 500:
            self.counts["rejected"] += len(batch)
            self._set_online(True)
            self.log("backend REJECTED %d observations (HTTP %d): %s"
                     % (len(batch), status, body[:500]))
            return True
        self._requeue_front(batch)
        self._set_online(False, body if status is None else "HTTP %d" % status)
        return False

    def _send_loop(self):
        delay = self.batch_interval_s
        while not self._stop.wait(delay):
            if self._send_once():
                delay = self.batch_interval_s
            else:
                delay = min(max(delay * 2, self.batch_interval_s), self.max_backoff_s)
        if self.online:
            self._send_once()  # final flush on shutdown

    def _heartbeat_loop(self):
        while not self._stop.wait(self.heartbeat_interval_s):
            snap = self.stats.snapshot() if self.stats is not None else {}
            if snap is None:
                if not self._heartbeat_paused:
                    self.log("%s: heartbeat paused so the backend shows this source offline"
                             % self.idle_reason)
                    self._heartbeat_paused = True
                continue
            if self._heartbeat_paused:
                self.log("data resumed: heartbeat restarted")
                self._heartbeat_paused = False
            hb = contract.build_heartbeat(self.clock(), self.source, self.room_id,
                                          fps=snap.get("fps"),
                                          inference_ms=snap.get("inference_ms"),
                                          mem_used_mb=snap.get("mem_used_mb"),
                                          source_type=self.source_type)
            status, body = self._post(self.heartbeat_path, hb)
            if status is not None and 200 <= status < 300:
                self.counts["heartbeats"] += 1
            elif status is not None and 400 <= status < 500:
                self.log("backend REJECTED heartbeat (HTTP %d): %s" % (status, body[:300]))

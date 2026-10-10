"""EdgeClient tests against a tiny in-process fake backend.

Run on the laptop: .venv/bin/python -m pytest edge/tests
"""

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hearth_edge import client as cl  # noqa: E402
from hearth_edge import contract as c  # noqa: E402

hs = pytest.importorskip("hearth_schema")

FAST = dict(batch_interval_s=0.02, heartbeat_interval_s=0.05, timeout_s=0.5,
            max_backoff_s=0.1, log=lambda msg: None)


class FakeBackend(object):
    """Records POSTed JSON per path. mode: 'ok' -> 200, 'reject' -> 422 for observations."""

    def __init__(self, port=0):
        self.received = {"/ingest/observations": [], "/ingest/heartbeat": []}
        self.requests = {"/ingest/observations": 0, "/ingest/heartbeat": 0}
        self.mode = "ok"
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                backend.requests[self.path] += 1
                if self.path == "/ingest/observations" and backend.mode == "reject":
                    self.send_response(422)
                    self.end_headers()
                    self.wfile.write(b'{"detail": "extra field"}')
                    return
                backend.received[self.path].append(body)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = "http://127.0.0.1:%d" % self.port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def observations(self):
        return [o for batch in self.received["/ingest/observations"] for o in batch["observations"]]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class FixedStats(object):
    def __init__(self, snap):
        self.snap = snap

    def snapshot(self):
        return self.snap


def make_observations(n):
    ids = c.ObservationIds("jetson-01", start=1791700000)
    return [c.build_observation(ids.next(), 1791700000.0 + i * 0.1, "jetson-01", "room-01",
                                person_detected=False) for i in range(n)]


@pytest.fixture
def backend():
    b = FakeBackend()
    yield b
    b.close()


def test_batches_arrive_in_order_and_pass_contract(backend):
    client = cl.EdgeClient(backend.url, "jetson-01", "room-01", **FAST)
    client.start()
    obs = make_observations(25)
    for o in obs:
        client.submit(o)
    try:
        assert wait_for(lambda: len(backend.observations()) == 25)
    finally:
        client.stop()
    for batch in backend.received["/ingest/observations"]:
        hs.ObservationBatch.model_validate(batch)
    assert [o["obs_id"] for o in backend.observations()] == [o["obs_id"] for o in obs]
    assert client.counts["sent"] == 25 and client.online is True


def test_reconnects_and_sends_only_newest_after_outage():
    port = free_port()  # nothing listening yet: backend is "down"
    client = cl.EdgeClient("http://127.0.0.1:%d" % port, "jetson-01", "room-01",
                           queue_max=5, **FAST)
    client.start()
    obs = make_observations(10)
    for o in obs:
        client.submit(o)
    try:
        assert wait_for(lambda: client.online is False)
        backend = FakeBackend(port)  # backend comes back on the same port
        try:
            assert wait_for(lambda: len(backend.observations()) == 5)
            time.sleep(0.1)
            got = [o["obs_id"] for o in backend.observations()]
        finally:
            backend.close()
    finally:
        client.stop()
    assert got == [o["obs_id"] for o in obs[5:]]  # oldest 5 dropped, newest 5 delivered
    assert client.counts["dropped"] == 5 and client.online is True


def test_rejected_batch_is_dropped_not_retried(backend):
    backend.mode = "reject"
    logs = []
    fast = dict(FAST, log=logs.append)
    client = cl.EdgeClient(backend.url, "jetson-01", "room-01", **fast)
    client.start()
    for o in make_observations(3):
        client.submit(o)
    try:
        assert wait_for(lambda: client.counts["rejected"] == 3)
        time.sleep(0.15)
        assert backend.requests["/ingest/observations"] == 1  # not resent
        backend.mode = "ok"
        for o in make_observations(2):
            client.submit(o)
        assert wait_for(lambda: len(backend.observations()) == 2)
    finally:
        client.stop()
    assert any("REJECTED" in m and "extra field" in m for m in logs)


def test_queue_drops_oldest_when_full():
    client = cl.EdgeClient("http://127.0.0.1:1", "jetson-01", "room-01", queue_max=5, **FAST)
    obs = make_observations(8)
    for o in obs:
        client.submit(o)  # threads not started: nothing is sent
    assert client.queued() == 5 and client.counts["dropped"] == 3
    assert [o["obs_id"] for o in list(client._queue)] == [o["obs_id"] for o in obs[3:]]


def test_heartbeats_arrive_and_pass_contract(backend):
    stats = FixedStats({"fps": 14.1, "inference_ms": 57.4, "mem_used_mb": 1403.0})
    client = cl.EdgeClient(backend.url, "jetson-01", "room-01", stats=stats, **FAST)
    client.start()
    try:
        assert wait_for(lambda: len(backend.received["/ingest/heartbeat"]) >= 2)
    finally:
        client.stop()
    for hb in backend.received["/ingest/heartbeat"]:
        assert hs.Heartbeat.model_validate(hb).fps == pytest.approx(14.1)


def test_heartbeat_pauses_when_no_camera_frames(backend):
    logs = []
    client = cl.EdgeClient(backend.url, "jetson-01", "room-01", stats=FixedStats(None),
                           **dict(FAST, log=logs.append))
    client.start()
    time.sleep(0.3)
    client.stop()
    assert backend.received["/ingest/heartbeat"] == []
    assert any("heartbeat paused" in m for m in logs)


def test_frame_stats_rate_and_staleness():
    now = [1000.0]
    stats = cl.FrameStats(window_s=2.0, stale_s=2.0, clock=lambda: now[0], mem_reader=lambda: 512.0)
    assert stats.snapshot() is None  # no frames yet
    for i in range(11):  # 10 frames per second for 1 second
        now[0] = 1000.0 + i * 0.1
        stats.record(50.0 + i)
    snap = stats.snapshot()
    assert snap["fps"] == pytest.approx(10.0)
    assert snap["inference_ms"] == pytest.approx(55.0)
    assert snap["mem_used_mb"] == 512.0
    now[0] += 2.5  # camera stopped
    assert stats.snapshot() is None


def test_read_mem_used_mb(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:        2026704 kB\nMemFree: 70000 kB\nMemAvailable:     593920 kB\n")
    assert cl.read_mem_used_mb(str(p)) == pytest.approx((2026704 - 593920) / 1024.0)
    assert cl.read_mem_used_mb(str(tmp_path / "missing")) is None

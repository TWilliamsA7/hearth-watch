"""Stand-in backend tests. Run on the laptop: .venv/bin/python -m pytest edge/tests"""

import importlib.util
import json
import os
import sys
import threading
import time
from urllib import error, request

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from hearth_edge import contract as c  # noqa: E402

pytest.importorskip("hearth_schema")

spec = importlib.util.spec_from_file_location(
    "stub_backend", os.path.join(HERE, "..", "tools", "stub_backend.py"))
stub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stub)


@pytest.fixture
def server(tmp_path):
    state = stub.State(str(tmp_path / "saved.jsonl"))
    srv = stub.make_server(state, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1], state, tmp_path / "saved.jsonl"
    srv.shutdown()
    srv.server_close()


def post(url, payload):
    req = request.Request(url, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=2) as r:
            return r.getcode(), json.loads(r.read())
    except error.HTTPError as e:
        return e.code, json.loads(e.read())


def obs(n=1):
    ids = c.ObservationIds("jetson-01", start=1791700000)
    return [c.build_observation(ids.next(), time.time(), "jetson-01", "room-01", False)
            for _ in range(n)]


def test_accepts_valid_batch_and_saves_it(server):
    url, state, saved = server
    code, body = post(url + "/ingest/observations", {"observations": obs(3)})
    assert code == 200 and body == {"accepted": 3}
    snap = state.snapshot()["sources"]["jetson-01"]
    assert snap["online"] and snap["observations"] == 3
    assert len(saved.read_text().splitlines()) == 3


def test_rejects_message_that_breaks_contract(server):
    url, state, _ = server
    bad = obs(1)
    bad[0]["torso_angle"] = 3.0  # misspelled field: contract forbids unknown keys
    code, body = post(url + "/ingest/observations", {"observations": bad})
    assert code == 422 and "torso_angle" in body["detail"]
    assert state.snapshot()["invalid"] == 1


def test_accepts_heartbeat(server):
    url, state, _ = server
    hb = c.build_heartbeat(time.time(), "jetson-01", "room-01", fps=14.0, inference_ms=57.0,
                           mem_used_mb=1400.0)
    assert post(url + "/ingest/heartbeat", hb)[0] == 200
    assert state.snapshot()["sources"]["jetson-01"]["heartbeats"] == 1


def test_offline_after_three_seconds_of_silence():
    assert stub.source_online(100.0, 102.9)
    assert not stub.source_online(100.0, 103.1)
    assert not stub.source_online(None, 100.0)


def test_summary_line_shows_features_and_offline():
    s = {"online": True, "seconds_since_last_message": 0.1,
         "latest_observation": {"person_detected": True, "face_present": None,
                                "features": {"torso_angle_deg": 3.0, "hip_y": 0.55}},
         "latest_heartbeat": {"fps": 14.6, "inference_ms": 57.0, "mem_used_mb": 1430.0}}
    line = stub.summary_line("jetson-01", s, 10)
    assert "obs/s 10" in line and "torso 3" in line and "hip 0.55" in line and "face null" in line
    s["online"], s["seconds_since_last_message"] = False, 4.2
    assert "OFFLINE (no data for 4.2 s)" in stub.summary_line("jetson-01", s, 0)


def test_empty_body_is_rejected_with_a_clear_reason(server):
    url, state, _ = server
    req = request.Request(url + "/ingest/observations", data=b"",
                          headers={"Content-Type": "application/json"})
    with pytest.raises(error.HTTPError) as exc:
        request.urlopen(req, timeout=2)
    assert exc.value.code == 422
    assert "empty or cut-off" in json.loads(exc.value.read())["detail"]

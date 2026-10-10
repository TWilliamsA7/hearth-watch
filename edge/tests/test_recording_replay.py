"""Recorder, record timer and replay tests. Run on the laptop: .venv/bin/python -m pytest edge/tests"""

import importlib.util
import json
import os
import sys
import threading
import time

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import run_edge  # noqa: E402
from hearth_edge import contract as c  # noqa: E402
from hearth_edge import recorder as rec  # noqa: E402
from hearth_edge.client import EdgeClient  # noqa: E402

hs = pytest.importorskip("hearth_schema")


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", "tools", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


replay = load_tool("replay")
stub = load_tool("stub_backend")

T0 = 1791700000.0
POSE_KPS = c.normalize_keypoints({"neck": (320, 110), "left_hip": (305, 250)}, 640, 480, None)


def recorded_observations(n=20, hz=10.0):
    ids = c.ObservationIds("jetson-01", start=int(T0))
    return [c.build_observation(ids.next(), T0 + i / hz, "jetson-01", "room-01", True,
                                keypoints=POSE_KPS, features={"torso_angle_deg": 80.0 - i})
            for i in range(n)]


def write_recording(tmp_path, observations, label="lie_down"):
    r = rec.Recorder(label, str(tmp_path), clock=lambda: T0)
    for o in observations:
        r.write(o)
    r.close()
    return r.path


# --- recorder ---------------------------------------------------------------------------

def test_recorder_writes_valid_jsonl(tmp_path):
    path = write_recording(tmp_path, recorded_observations(5))
    assert os.path.basename(path).startswith("lie_down-") and path.endswith(".jsonl")
    lines = open(path).read().splitlines()
    assert len(lines) == 5
    for line in lines:
        hs.Observation.model_validate(json.loads(line))


def test_recorder_label_cannot_escape_the_folder(tmp_path):
    assert rec.safe_label("../../etc/passwd") == "etc_passwd"
    assert rec.safe_label("lie down!") == "lie_down"
    assert rec.safe_label("///") == "clip"
    r = rec.Recorder("../x", str(tmp_path), clock=lambda: T0)
    r.close()
    assert os.path.dirname(r.path) == str(tmp_path)


def test_two_takes_in_the_same_second_get_different_files(tmp_path):
    a = rec.Recorder("sit", str(tmp_path), clock=lambda: T0)
    b = rec.Recorder("sit", str(tmp_path), clock=lambda: T0)
    a.close(), b.close()
    assert a.path != b.path


def test_default_dir_is_repo_recordings():
    # edge/hearth_edge/recorder.py -> <repo>/recordings (the git-ignored folder from the README)
    expected = os.path.abspath(os.path.join(HERE, "..", "..", "recordings"))
    assert os.path.abspath(rec.default_dir()) == expected


# --- record timer (countdown + window) --------------------------------------------------

def test_record_timer_counts_down_then_records_for_the_window():
    t = run_edge.RecordTimer(start=100.0, countdown_s=3.0, seconds=10.0)
    assert [t.countdown_left(x) for x in (100.0, 100.5, 101.2, 102.9, 103.0)] == [3, 3, 2, 1, 0]
    assert not t.recording(102.99) and t.recording(103.0) and t.recording(112.99)
    assert not t.recording(113.0) and t.done(113.0) and not t.done(112.0)


def test_record_timer_without_seconds_records_until_stopped():
    t = run_edge.RecordTimer(start=0.0, countdown_s=0.0, seconds=0.0)
    assert t.recording(0.0) and t.recording(1e6) and not t.done(1e6)


# --- timestamps -------------------------------------------------------------------------

def test_parse_utc_iso_round_trips():
    for ts in (T0, T0 + 0.481, 1.5):
        assert c.parse_utc_iso(c.utc_iso(ts)) == pytest.approx(ts, abs=0.001)
    assert c.parse_utc_iso("2026-10-10T05:12:03Z") == pytest.approx(c.parse_utc_iso("2026-10-10T05:12:03.000Z"))


# --- replay -----------------------------------------------------------------------------

def test_rewrite_relabels_but_keeps_the_data():
    o = recorded_observations(1)[0]
    r = replay.rewrite_for_replay(o, "replay-01-1-000001", T0 + 50, run_id="replay-01-run-1")
    assert r["source"] == "replay-01" and r["source_type"] == "replay"
    assert r["run_id"] == "replay-01-run-1" and r["scenario_id"] is None
    assert r["obs_id"] == "replay-01-1-000001" and r["timestamp"] == c.utc_iso(T0 + 50)
    assert r["keypoints"] == o["keypoints"] and r["features"] == o["features"]
    assert o["source"] == "jetson-01"  # original untouched
    hs.Observation.model_validate(r)


def test_schedule_keeps_spacing_and_scales_with_speed():
    obs = recorded_observations(5, hz=10.0)
    assert replay.schedule(obs) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert replay.schedule(obs, speed=2.0) == pytest.approx([0.0, 0.05, 0.1, 0.15, 0.2])


def test_load_recording_skips_bad_lines(tmp_path):
    path = write_recording(tmp_path, recorded_observations(3))
    with open(path, "a") as fh:
        fh.write("not json\n\n{\"no_timestamp\": 1}\n")
    assert len(replay.load_recording(path)) == 3


def test_replay_end_to_end_into_stub_backend(tmp_path):
    # 40 observations = 3.9 s of recording, 1.95 s at 2x: long enough for >= 1 heartbeat (1/s).
    path = write_recording(tmp_path, recorded_observations(40, hz=10.0))
    state = stub.State()
    srv = stub.make_server(state, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % srv.server_address[1]
    try:
        start = time.time()
        assert replay.main([path, "--backend", url, "--speed", "2"]) == 0
        elapsed = time.time() - start
        snap = state.snapshot()
    finally:
        srv.shutdown()
        srv.server_close()
    assert snap["invalid"] == 0
    assert "jetson-01" not in snap["sources"]  # replay never appears as the live camera
    src = snap["sources"]["replay-01"]
    assert src["source_type"] == "replay" and src["observations"] == 40
    assert src["latest_observation"]["features"]["torso_angle_deg"] == pytest.approx(41.0)
    assert src["latest_observation"]["run_id"].startswith("replay-01-")       # per-run id
    assert src["latest_heartbeat"]["run_id"] == src["latest_observation"]["run_id"]
    assert elapsed >= 3.9 / 2  # original 3.9 s spacing, at 2x speed


def test_each_playback_pass_gets_a_new_run_id():
    a = replay.new_run_id("replay-01", 1, clock=lambda: T0)
    b = replay.new_run_id("replay-01", 2, clock=lambda: T0)
    assert a != b and a.startswith("replay-01-") and a.endswith("-1")


# --- control commands (record_clips.sh drives one long-running run_edge.py) ------------

def test_parse_control_line():
    assert run_edge.parse_control_line("record stand 10 5\n") == ("record", "stand", 10.0, 5.0)
    assert run_edge.parse_control_line("record lie_down 8") == ("record", "lie_down", 8.0, None)
    assert run_edge.parse_control_line("quit") == ("quit",)
    for bad in ("", "record", "record x", "record x ten", "record x 0", "record x 5 -1", "dance"):
        assert run_edge.parse_control_line(bad) is None


@pytest.mark.skipif(sys.platform == "win32", reason="Jetson stdin control uses POSIX pipe select; Windows select accepts sockets only")
def test_read_control_never_blocks_and_reports_end_of_input():
    r, w = os.pipe()
    reader = os.fdopen(r, "r")
    writer = os.fdopen(w, "w")
    try:
        assert run_edge.read_control(reader) is None          # nothing waiting: returns at once
        writer.write("record sit 10 5\n")
        writer.flush()
        assert run_edge.read_control(reader) == "record sit 10 5\n"
        writer.close()
        assert run_edge.read_control(reader) == ""             # sender went away
    finally:
        reader.close()


def test_join_makes_one_continuous_clip():
    a = recorded_observations(5, hz=10.0)                       # 0.0 .. 0.4 s
    b = recorded_observations(3, hz=10.0)                       # also starts at T0
    joined = replay.join_recordings([a, b], gap_s=0.1)
    offsets = replay.schedule(joined)
    assert offsets == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert a[0]["timestamp"] == c.utc_iso(T0)                   # originals untouched


def test_joined_stand_lie_stand_gives_alert_then_recovery_in_one_run(tmp_path):
    stand = c.normalize_keypoints({"nose": (320, 40), "neck": (320, 80), "left_hip": (320, 280)}, 640, 480, None)
    lie = c.normalize_keypoints({"nose": (100, 380), "neck": (160, 385), "left_hip": (330, 390)}, 640, 480, None)

    def clip(n, kps, feats):
        ids = c.ObservationIds("jetson-01", start=int(T0))
        return [c.build_observation(ids.next(), T0 + i / 10.0, "jetson-01", "room-01", True,
                                    keypoints=kps, features=feats) for i in range(n)]
    up = {"torso_angle_deg": 2.0, "head_y": 0.08, "hip_y": 0.58, "bbox_aspect": 0.3}
    down = {"torso_angle_deg": 85.0, "head_y": 0.79, "hip_y": 0.81, "bbox_aspect": 3.0}
    paths = []
    for name, obs in (("stand", clip(10, stand, up)), ("lie", clip(45, lie, down)), ("stand2", clip(15, stand, up))):
        p = tmp_path / (name + ".jsonl")
        p.write_text("\n".join(json.dumps(o) for o in obs) + "\n")
        paths.append(str(p))
    pytest.importorskip("hearth.api.app")
    from fastapi.testclient import TestClient
    from hearth.api.app import create_app
    from hearth.monitor.config import MonitorConfig
    app = create_app(MonitorConfig.load())
    joined = replay.join_recordings([replay.load_recording(p) for p in paths])
    run_id = replay.new_run_id("replay-01", 1)
    ids = c.ObservationIds("replay-01", start=int(T0))
    rewritten = [replay.rewrite_for_replay(o, ids.next(), c.parse_utc_iso(o["timestamp"]),
                                           run_id=run_id) for o in joined]
    with TestClient(app) as http:
        assert http.post("/ingest/observations", json={"observations": rewritten}).status_code == 200
        alerts = http.get("/alerts", params={"source": "replay-01"}).json()
    assert [a["kind"] for a in alerts] == ["possible_fall"]
    assert alerts[0]["recovered_at"] is not None and alerts[0]["run_id"] == run_id

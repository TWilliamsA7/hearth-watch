from fastapi.testclient import TestClient

from hearth.api.app import create_app
from hearth.eval.batch import drive, load_monitor_factory
from hearth.monitor.config import MonitorConfig
from hearth.monitor.detector import source_key, track_key
from hearth_schema import Heartbeat
from hearth.sim.runner import run_headless
from hearth.sim.scenario import load_all
from conftest import FLOOR, UPRIGHT, observation


def test_default_factory_uses_real_monitor_and_virtual_health_clock():
    scenarios = load_all()
    fall = drive(run_headless(scenarios["possible-fall-01"]), load_monitor_factory()())
    assert len([e for e in fall if e.event_type == "possible_fall"]) == 1
    run = run_headless(scenarios["camera-outage-01"])
    events = drive(run, load_monitor_factory()())
    assert any(e.event_type == "source_offline" and
               10 <= (e.timestamp - run.start).total_seconds() <= 11 for e in events)


def test_residents_at_same_timestamp_keep_independent_holds(service):
    for t in (0, .5, 1, 1.5, 2, 2.5, 3, 3.5):
        a = observation(t, FLOOR).model_copy(update={"person_id": "a"})
        b = observation(t, UPRIGHT).model_copy(update={"person_id": "b"})
        assert service.ingest([a, b])["accepted"] == 2
    falls = [e for e in service.events if e.event_type == "possible_fall"]
    assert [e.person_id for e in falls] == ["a"]
    assert service.detector.states[track_key(b)].posture == "upright"
    service.ingest([observation(4, person_detected=False)])
    assert service.detector.states[track_key(b)].posture == "unknown"


def test_mounted_simulation_replay_creates_new_run_and_real_alerts(service, tmp_path):
    app = create_app(service=service)
    ctl = app.state.sim_controller
    ctl._live_dir = tmp_path

    async def play_once():
        await ctl.load("possible-fall-01")
        await ctl.play(autorun=False)
        run_id = ctl.status().run_id
        # Accelerated arrival must preserve the observation-time floor hold.
        while ctl.state != "ended":
            await ctl.tick(.25)
        return run_id

    with TestClient(app) as client:
        cors = client.get("/sim/status", headers={"Origin": "http://localhost:5174"})
        assert cors.headers["access-control-allow-origin"] == "http://localhost:5174"
        assert client.get("/sim/scenarios").status_code == 200
        assert client.get("/eval/runs").json() == []
        first = client.portal.call(play_once)
        second = client.portal.call(play_once)
        assert first != second
        for run_id in (first, second):
            alerts = client.get("/alerts", params={"run_id": run_id}).json()
            assert len([a for a in alerts if a["kind"] == "possible_fall"]) == 1
            report = client.get(f"/eval/runs/{run_id}")
            assert report.status_code == 200
            assert report.json()["counts"]["true_positives"] == 1
        assert all(k[-1] in {first, second} for k in service.health)


def test_framing_profile_explicitly_allows_pose():
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "recordings" / "config_framing.json"
    config = MonitorConfig.load(path)
    assert config.posture_mode == "pose_fallback"
    assert config.upright_hip_y_max > config.floor_hip_y_min


def test_explicit_pause_preserves_hold_but_silence_still_invalidates(service, clock):
    service.ingest([observation(0), observation(.5), observation(1)])
    key, track = source_key(observation(0)), track_key(observation(0))
    service.observation_clock_paused = lambda source: source == key
    for second in range(1, 6):
        clock.seconds = second
        service.heartbeat(Heartbeat(timestamp=observation(1).timestamp, source="stub",
                                   source_type="simulation", room_id="room-01",
                                   scenario_id="scenario", run_id="run-1"))
    assert service.health[key].status.observation_stale
    assert service.health[key].status.posture == "unknown"
    assert service.detector.states[track].floor_since is not None
    assert not service.alerts
    clock.seconds = 8
    service.check_health()
    assert not service.health[key].status.online
    assert service.detector.states[track].floor_since is None

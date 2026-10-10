"""The live app plays simulated scenarios through the real monitoring service."""

import time

from fastapi.testclient import TestClient
from hearth.api.app import create_app, ingest_records
from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService
from hearth.sim.runner import run_headless
from hearth.sim.scenario import load_all

SCENARIOS = load_all()


def shipped_service() -> MonitorService:
    return MonitorService(MonitorConfig.load())


def test_app_mounts_sim_and_eval_routes():
    with TestClient(create_app(service=shipped_service())) as client:
        ids = {s["scenario_id"] for s in client.get("/sim/scenarios").json()}
        assert "possible-fall-01" in ids
        assert client.get("/sim/status").json()["state"] == "idle"
        assert client.get("/eval/runs").json() == []


def test_simulation_can_be_left_out():
    with TestClient(create_app(service=shipped_service(), simulation=False)) as client:
        assert client.get("/sim/status").status_code == 404


def test_several_people_in_one_frame_all_reach_the_live_monitor():
    """Three residents share each frame's timestamp; none may be dropped as out_of_order."""
    service = shipped_service()
    run = run_headless(SCENARIOS["day-room-01"])  # p3 falls
    events = ingest_records(service, run.items)
    assert service.ingest_totals["out_of_order"] == 0
    assert service.ingest_totals["accepted"] == len(run.observations)
    falls = [e for e in events if e.event_type == "possible_fall"]
    assert [e.person_id for e in falls] == ["p3"]
    assert not service.event_listeners  # the hook is removed afterwards


def test_a_loaded_scenario_plays_into_the_live_alert_queue():
    service = shipped_service()
    with TestClient(create_app(service=service)) as client:
        assert client.post("/sim/load", json={"scenario_id": "possible-fall-01"}).status_code == 200
        client.post("/sim/speed", json={"speed": 4})
        client.post("/sim/play")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not client.get("/alerts").json():
            time.sleep(0.05)
        alerts = client.get("/alerts").json()
        assert [a["kind"] for a in alerts] == ["possible_fall"]
        assert alerts[0]["source_type"] == "simulation"

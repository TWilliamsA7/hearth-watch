"""The /sim and /eval routes, mounted the way the monitoring core's app will mount them."""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hearth.eval.api import create_eval_router
from hearth.sim.api import create_sim_router
from hearth.sim.controller import SimController
from test_controller import SCENARIOS, FakeClock, Pipeline


@pytest.fixture
def client(tmp_path):
    pipe = Pipeline()
    controller = SimController(pipe.ingest, clock=FakeClock(), scenarios=SCENARIOS, live_dir=tmp_path / "live")
    app = FastAPI()
    app.include_router(create_sim_router(controller), prefix="/sim")
    app.include_router(create_eval_router(controller, batch_dir=tmp_path / "batch"), prefix="/eval")
    with TestClient(app) as c:
        c.pipe = pipe
        c.batch_dir = tmp_path / "batch"
        yield c


def wait_for(client, state, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/sim/status").json()
        if status["state"] == state:
            return status
        time.sleep(0.01)
    raise AssertionError(f"never reached {state!r}: {status}")


def test_lists_scenarios(client):
    ids = {s["scenario_id"] for s in client.get("/sim/scenarios").json()}
    assert {"possible-fall-01", "day-room-01"} <= ids


def test_idle_until_loaded(client):
    assert client.get("/sim/status").json()["state"] == "idle"
    assert client.post("/sim/play").status_code == 409
    assert client.post("/sim/load", json={"scenario_id": "nope"}).status_code == 404


def test_full_live_run_produces_an_after_action_report(client):
    loaded = client.post("/sim/load", json={"scenario_id": "possible-fall-01"}).json()
    assert loaded["state"] == "loaded" and loaded["scene"]["floor_y"] == 0.92
    client.post("/sim/play")
    ended = wait_for(client, "ended")  # the fake clock runs the 30 s scenario near-instantly

    runs = client.get("/eval/runs").json()
    assert [r["run_id"] for r in runs] == [ended["run_id"]]
    report = client.get(f"/eval/runs/{ended['run_id']}").json()
    assert report["counts"]["true_positives"] == 1
    assert client.pipe.observations  # records went through the injected ingest
    assert client.get("/eval/runs/unknown").status_code == 404


def test_status_shows_truth_per_resident(client):
    client.post("/sim/load", json={"scenario_id": "day-room-01", "seed": 7})
    status = client.get("/sim/status").json()
    assert status["seed"] == 7
    assert [t["person_id"] for t in status["truth"]] == ["p1", "p2", "p3", "p4"]


def test_speed_is_validated(client):
    assert client.post("/sim/speed", json={"speed": 2}).json()["speed"] == 2
    assert client.post("/sim/speed", json={"speed": 10}).status_code == 422


def test_summary_comes_from_the_latest_batch(client):
    assert client.get("/eval/summary").status_code == 404
    from hearth.eval.scoring import summarize

    client.batch_dir.mkdir(parents=True)
    (client.batch_dir / "summary.json").write_text(summarize([]).model_dump_json())
    assert client.get("/eval/summary").json()["runs"] == 0

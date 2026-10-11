"""The live engine's controller and /sim/live routes, and the app mounting the simulator."""

import asyncio
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
from hearth_schema.models import GroundTruth, Heartbeat, Observation

from hearth.api.app import create_app
from hearth.sim.api import create_sim_router
from hearth.sim.controller import SimController
from hearth.sim.live.controller import LiveController
from hearth.sim.live.spec import EngineSpec, IncidentSpec
from test_controller import SCENARIOS, FakeClock, Pipeline

SMALL = {"actors": 3, "room": {"width": 5.5, "depth": 5.0}}


def run(coro):
    return asyncio.run(coro)


def test_plays_through_ingest_on_one_continuous_clock() -> None:
    clock, pipe = FakeClock(), Pipeline()
    live = LiveController(pipe.ingest, clock=clock)

    async def go():
        await live.new(EngineSpec(actors=3), 2)
        start = clock.now()
        await live.play(autorun=False)
        for _ in range(100):  # 5 s
            await live.tick(0.05)
        await live.pause()
        for _ in range(60):  # 3 s paused: heartbeats only
            clock.t += __import__("datetime").timedelta(seconds=0.05)
            await live.tick(0.05)
        return start

    start = run(go())
    obs = pipe.observations
    assert obs and not any(isinstance(r, GroundTruth) for r in pipe.received)
    stamps = [r.timestamp for r in pipe.received]
    assert stamps == sorted(stamps)
    assert abs((obs[-1].timestamp - start).total_seconds() - live.world.t) < 0.2
    paused = [r for r in pipe.received if r.timestamp > obs[-1].timestamp]
    assert paused and all(isinstance(r, Heartbeat) for r in paused)  # a paused run is not an outage


def test_speed_runs_the_engine_faster() -> None:
    clock, pipe = FakeClock(), Pipeline()
    live = LiveController(pipe.ingest, clock=clock)

    async def go():
        await live.new(EngineSpec(actors=2), 1)
        await live.set_speed(4.0)
        await live.play(autorun=False)
        for _ in range(20):  # 1 s of wall time
            await live.tick(0.05)

    run(go())
    assert abs(live.world.t - 4.0) < 0.06


def test_reset_replays_the_same_run() -> None:
    def first_records():
        pipe = Pipeline()
        live = LiveController(pipe.ingest, clock=FakeClock())

        async def go():
            await live.new(EngineSpec(actors=3), 9)
            await live.play(autorun=False)
            for _ in range(40):
                await live.tick(0.05)
            await live.reset()
            await live.play(autorun=False)
            for _ in range(40):
                await live.tick(0.05)

        run(go())
        obs = [o for o in pipe.observations]
        half = len(obs) // 2
        return [o.keypoints for o in obs[:half]], [o.keypoints for o in obs[half:]]

    before, after = first_records()
    assert before == after


def make_client():
    pipe = Pipeline()
    scripted = SimController(pipe.ingest, clock=FakeClock(), scenarios=SCENARIOS, live_dir=None)
    live = LiveController(pipe.ingest)
    app = FastAPI()
    app.include_router(create_sim_router(scripted, live), prefix="/sim")
    return TestClient(app), pipe, scripted, live


def test_live_routes() -> None:
    client, pipe, scripted, live = make_client()
    with client:
        assert client.get("/sim/live/status").json()["state"] == "idle"
        assert client.post("/sim/live/play").status_code == 409
        world = client.post("/sim/live/new", json={"spec": SMALL, "seed": 4}).json()
        assert world["room"]["bounds"] and world["room"]["doors"] and len(world["actors"]) == 3
        assert any(o["variant"] for o in world["room"]["objects"])
        assert world["scene"]["objects"]
        assert client.post("/sim/live/play").json()["state"] == "playing"
        deadline = time.monotonic() + 10
        while client.get("/sim/live/status").json()["t"] < 1.0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pipe.observations
        detail = client.get("/sim/live/actors/p1").json()
        assert detail["info"]["person_id"] == "p1" and detail["now"]["joints"]
        assert client.get("/sim/live/actors/zz").status_code == 404
        rec = client.post("/sim/live/incidents", json={"kind": "inactivity", "actor": "p2", "duration": 5}).json()
        assert rec["kind"] == "inactivity" and rec["source"] == "manual"
        assert client.post("/sim/live/incidents", json={"kind": "fall", "actor": "zz"}).status_code == 404
        assert client.post("/sim/live/actors/p2/reset").json()["now"]["incident"] is None
        spec = client.post("/sim/live/config", json={"social_frequency": 2.5}).json()
        assert spec["social_frequency"] == 2.5
        assert client.post("/sim/live/speed", json={"speed": 99}).status_code == 422
        assert client.get("/sim/live/report").json()["counts"] is not None
        assert client.post("/sim/live/pause").json()["state"] == "paused"


def test_ws_pushes_the_world_then_frames() -> None:
    client, *_ = make_client()
    with client:
        client.post("/sim/live/new", json={"spec": SMALL, "seed": 3, "play": True})
        with client.websocket_connect("/sim/live/ws") as ws:
            first = ws.receive_json()
            assert first["type"] == "world" and first["data"]["actors"]
            kinds = {ws.receive_json()["type"] for _ in range(4)}
            assert kinds == {"frame"}
        frame = None
        with client.websocket_connect("/sim/live/ws") as ws:
            ws.receive_json()
            frame = ws.receive_json()["data"]
        person = frame["people"][0]
        assert {"label", "activity", "intent", "posture", "joints", "visible"} <= set(person)


def test_playing_one_simulated_source_pauses_the_other() -> None:
    client, pipe, scripted, live = make_client()
    with client:
        client.post("/sim/live/new", json={"spec": SMALL, "seed": 1, "play": True})
        client.post("/sim/load", json={"scenario_id": "possible-fall-01"})
        client.post("/sim/play")
        assert client.get("/sim/live/status").json()["state"] == "paused"
        client.post("/sim/live/play")
        assert client.get("/sim/status").json()["state"] in ("paused", "ended")


def test_the_app_mounts_the_simulator_on_the_real_monitor() -> None:
    with TestClient(create_app()) as client:
        client.post("/sim/live/new", json={"spec": SMALL, "seed": 1})
        app_live = client.app.state.simulation.live
        app_live.trigger(IncidentSpec(kind="fall", duration=20))
        client.post("/sim/live/play")
        client.post("/sim/live/speed", json={"speed": 8})
        deadline = time.monotonic() + 20
        while client.get("/sim/live/status").json()["t"] < 25 and time.monotonic() < deadline:
            time.sleep(0.1)
        client.post("/sim/live/pause")
        sources = client.get("/sources").json()
        assert any(s["source"] == "sim" for s in sources)
        events = client.get("/events").json()
        assert any(e["source"] == "sim" for e in events)
        report = client.get("/sim/live/report").json()
        assert report["segments"]

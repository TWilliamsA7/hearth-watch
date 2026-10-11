"""Exercise actual branch APIs in isolated temporary checkouts (no merges).

Requires fastapi, httpx, pydantic and pyyaml. Produces serialized integration
fixtures for the dashboard tests. Hardware inference is not exercised.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "contracts"


def extract(ref: str, destination: Path):
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", ref], cwd=ROOT, text=True).splitlines()
    for name in names:
        if name.startswith(("backend/", "shared/python/", "edge/hearth_edge/", "scenarios/")):
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=ROOT))


def monitor(checkout: Path, edge: Path):
    sys.path[:0] = [str(checkout / "backend"), str(checkout / "shared/python"), str(edge / "edge")]
    from datetime import datetime, timezone, timedelta
    from fastapi.testclient import TestClient
    from hearth.api.app import create_app
    from hearth.monitor.config import MonitorConfig
    from hearth.monitor.service import MonitorService
    from hearth_schema import Observation
    from hearth_edge.contract import build_observation

    fixture = json.loads((checkout / "backend/tests/fixtures/gaston_edge.json").read_text())
    epoch = datetime(2026, 10, 10, tzinfo=timezone.utc)
    elapsed = [0.0]
    service = MonitorService(MonitorConfig.load(checkout / "backend/hearth/monitor/config.json"),
                             clock=lambda: elapsed[0], utcnow=lambda: epoch + timedelta(seconds=elapsed[0]))
    messages = []
    with TestClient(create_app(service=service)) as client:
        assert client.get("/health").status_code == 200
        with client.websocket_connect("/ws/room/room-01") as socket:
            for item in fixture["observations"]:
                ts = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                elapsed[0] = (ts - epoch).total_seconds()
                # Use the current edge branch's serializer, not a hand-written UI message.
                payload = build_observation(item["obs_id"], ts.timestamp(), item["source"], item["room_id"],
                                            item["person_detected"], item["keypoints"], item["features"],
                                            item["face_present"], item["face_confidence"])
                Observation.model_validate(payload)
                response = client.post("/ingest/observations", json={"observations": [payload]})
                assert response.status_code == 200, response.text
            # Every ingested observation is delivered, then the subsequent alert updates.
            for _ in range(len(fixture["observations"])):
                while True:
                    message = socket.receive_json()
                    messages.append(message)
                    if message["type"] == "observation":
                        break
            alerts = client.get("/alerts").json()
            fall = next(alert for alert in alerts if alert["kind"] == "possible_fall")
            acknowledged = client.post(f'/alerts/{fall["alert_id"]}/ack').json()
            assert acknowledged["status"] == "acknowledged"
            messages += [{"type": "alert", "data": alert} for alert in client.get("/alerts").json()]
            elapsed[0] += 4
            service.check_health()
            messages += [{"type": "source_status", "data": status} for status in client.get("/sources").json()]
            assert messages[-1]["data"]["observation_stale"] is True
        assert client.get("/sim/scenarios").status_code == 404
    (OUT / "monitoring.integration.fixture.json").write_text(json.dumps(messages, indent=2) + "\n")
    sim_observation = json.loads((OUT / "simulation.integration.fixture.json").read_text())["observation"]
    try:
        Observation.model_validate(sim_observation)
        raise AssertionError("Expected the existing branch contract mismatch")
    except ValueError as error:
        assert "person_id" in str(error)
    print("PASS: current edge serializer -> monitoring ingestion -> room WebSocket -> real alert acknowledgement -> stale source")
    print("CONFIRMED BLOCKERS: /sim routes absent; monitoring rejects simulator person_id")


def simulation(checkout: Path):
    sys.path[:0] = [str(checkout / "backend"), str(checkout / "shared/python")]
    import asyncio
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hearth.sim.api import create_sim_router
    from hearth.eval.api import create_eval_router
    from hearth.sim.controller import SimController

    async def ingest(items):
        return []  # Interface test only: no fake fall-detection results.

    controller = SimController(ingest, live_dir=None)
    app = FastAPI()
    app.include_router(create_sim_router(controller), prefix="/sim")
    app.include_router(create_eval_router(controller), prefix="/eval")
    with TestClient(app) as client:
        catalog = client.get("/sim/scenarios").json()
        response = client.post("/sim/load", json={"scenario_id": "possible-fall-01"})
        assert response.status_code == 200, response.text
        observation = next(item for item in controller._live.run.items if hasattr(item, "obs_id"))
        loaded = response.json()
        assert client.post("/sim/play").status_code == 200
        assert client.post("/sim/pause").json()["state"] == "paused"
        async def finish():
            await controller.play(autorun=False)
            await controller.tick(1000)
            await controller.close()
        client.portal.call(finish)
        status = client.get("/sim/status").json()
        assert status["state"] == "ended"
        report_response = client.get(f'/eval/runs/{status["run_id"]}')
        assert report_response.status_code == 200
        report = report_response.json()
        assert report["counts"]["true_positives"] == 0
        assert client.post("/sim/reset").json()["state"] == "loaded"
    data = {"catalog": catalog, "loaded": loaded, "status": status, "report": report,
            "observation": observation.model_dump(mode="json")}
    (OUT / "simulation.integration.fixture.json").write_text(json.dumps(data, indent=2) + "\n")
    print("PASS: actual simulator scenario list/load/play/pause/reset and completed-run evaluation API (empty monitor callback)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "sim": simulation(Path(sys.argv[2]))
        else: monitor(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        with tempfile.TemporaryDirectory(prefix="hearth-branches-") as directory:
            workspace = Path(directory)
            for name, ref in [("monitor", "origin/monitoring"), ("edge", "origin/gaston/edge"), ("sim", "origin/feat/simulation")]:
                extract(ref, workspace / name)
            for args in [("sim", str(workspace / "sim")), ("monitor", str(workspace / "monitor"), str(workspace / "edge"))]:
                subprocess.run([sys.executable, __file__, *args], cwd=ROOT, env=os.environ, check=True)

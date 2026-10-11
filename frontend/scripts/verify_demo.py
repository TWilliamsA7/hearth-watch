"""Exercise the composed demo with real services and a deterministic clock."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from demo_backend import create_demo_app
from fastapi.testclient import TestClient

epoch = datetime(2026, 10, 10, tzinfo=timezone.utc)
elapsed = [0.0]
app = create_demo_app(clock=lambda: elapsed[0], utcnow=lambda: epoch + timedelta(seconds=elapsed[0]))
monitor, controller = app.state.monitor, app.state.controller
messages = []
queue = monitor.subscribe(room_id="room-01")


def drain():
    while not queue.empty():
        message = queue.get_nowait()
        assert message is not None, "Test subscriber overflowed"
        messages.append(message)


async def exercise():
    # Controller uses its own record clock; monitor health uses the injected arrival clock.
    await controller.load("possible-fall-01")
    first = controller.status().run_id
    await controller.play(autorun=False)
    while controller.state != "ended":
        elapsed[0] += .1
        await controller.tick(.1)
        drain()
    report = controller.report(first)
    assert report.counts.true_positives >= 1, report
    assert any(a.kind == "possible_fall" and a.run_id == first for a in monitor.alerts.values())
    before = monitor.ingest_totals["accepted"]
    await controller.reset()
    second = controller.status().run_id
    assert second != first
    await controller.play(autorun=False)
    elapsed[0] += .1
    await controller.tick(.1)
    assert monitor.ingest_totals["accepted"] > before, "Reset observations were discarded"
    assert all(not s.status.online for s in monitor.health.values()
               if s.status.source_type == "simulation" and s.status.run_id == first)
    await controller.pause()
    elapsed[0] += 1
    await controller.tick(1)
    assert any(s.status.run_id == second and s.status.online for s in monitor.health.values())
    elapsed[0] += 4
    monitor.check_health()
    assert any(e.event_type == "source_offline" and e.run_id == second for e in monitor.events)
    assert not any(e.event_type == "source_offline" and e.run_id == first for e in monitor.events)
    await controller.close()
    print("PASS: actual fall detection, evaluation, reset isolation, pause heartbeat and cleanup")


asyncio.run(exercise())
with TestClient(app) as client:
    assert client.get("/sim/scenarios").status_code == 200
    assert client.get("/sim/status").status_code == 200
    response = client.post("/help-requests", json={"request_id": "demo-test", "room_id": "room-01"})
    assert response.status_code == 200, response.text
    alert = response.json()
    assert client.post("/help-requests", json={"request_id": "demo-test", "room_id": "room-01"}).json()["alert_id"] == alert["alert_id"]
    ack = client.post(f'/alerts/{alert["alert_id"]}/ack').json()
    assert ack["status"] == "acknowledged"
    print("PASS: HTTP routes, real help request, retry deduplication and acknowledgement")
drain()
samples = {}
for message in messages:
    data = message["data"]
    key = (message["type"], data.get("event_type"), data.get("kind"), data.get("status"),
           data.get("person_detected"), data.get("online"), data.get("observation_stale"))
    samples[key] = message
Path(__file__).resolve().parents[1].joinpath("contracts/demo-integration.fixture.json").write_text(json.dumps(list(samples.values()), indent=2), encoding="utf-8")

import json

from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from hearth_schema import ServerMessage
from hearth.api.app import create_app
from conftest import observation


def batch(*observations):
    return {"observations": [o.model_dump(mode="json") for o in observations]}


def test_http_websocket_alert_workflow(service):
    with TestClient(create_app(service=service)) as client:
        assert client.get("/health").json()["status"] == "ok"
        with client.websocket_connect("/ws/room/room-01") as ws:
            result = client.post("/ingest/observations", json=batch(observation(0)))
            assert result.json()["accepted"] == 1
            messages = [ws.receive_json() for _ in range(4)]
            for message in messages:
                TypeAdapter(ServerMessage).validate_python(message)
            assert {m["type"] for m in messages} == {"event", "source_status", "observation"}
        assert not service.subscribers
        assert client.post("/ingest/observations", json=batch(
            *[observation(t / 2) for t in range(1, 7)])).status_code == 200
        alert = client.get("/alerts").json()[0]
        assert client.post(f"/alerts/{alert['alert_id']}/resolve").status_code == 409
        assert client.post(f"/alerts/{alert['alert_id']}/ack").json()["status"] == "acknowledged"
        assert client.post(f"/alerts/{alert['alert_id']}/resolve").json()["status"] == "resolved"
        assert client.post("/alerts/missing/ack").status_code == 404
        snapshot = client.get("/rooms/room-01/snapshot").json()
        assert snapshot["alerts"][0]["status"] == "resolved"
        exported = client.get("/events/export?run_id=run-1")
        assert [json.loads(line) for line in exported.text.splitlines()] == client.get("/events?run_id=run-1").json()
        assert client.get("/events?run_id=other").json() == []
        # Reconnect gets current sources, latest observations, event history, alerts.
        expected = sum(len(snapshot[k]) for k in ("sources", "observations", "events", "alerts"))
        with client.websocket_connect("/ws/room/room-01") as ws:
            messages = [ws.receive_json() for _ in range(expected)]
            assert any(m["type"] == "alert" and m["data"]["status"] == "resolved" for m in messages)


def test_invalid_batch_is_atomic_and_ground_truth_rejected(service):
    with TestClient(create_app(service=service)) as client:
        assert client.post("/ingest/observations", json=batch(observation(0), observation(1, room_id="room-02"))).status_code == 422
        assert not service.events
        payload = batch(observation(0))
        payload["observations"][0]["truth_label"] = "fall"
        assert client.post("/ingest/observations", json=payload).status_code == 422
        assert client.post("/ingest/observations", json={"observations": []}).status_code == 422
        payload = batch(observation(0))
        payload["observations"][0]["timestamp"] = "2026-10-10T00:00:00"
        assert client.post("/ingest/observations", json=payload).status_code == 422
        assert client.get("/rooms/room-02/snapshot").status_code == 404


def test_heartbeat_and_cors(service):
    with TestClient(create_app(service=service)) as client:
        body = dict(timestamp="2026-10-10T00:00:00Z", source="jetson", source_type="edge",
                    room_id="room-01", fps=5, inference_ms=120, mem_used_mb=900)
        response = client.post("/ingest/heartbeat", json=body)
        assert response.status_code == 200
        assert response.json()["online"] and response.json()["observation_stale"]
        assert client.get("/sources").json()[0]["fps"] == 5
        response = client.options("/ingest/observations", headers={"Origin": "http://localhost:5173",
                                  "Access-Control-Request-Method": "POST"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

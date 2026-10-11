"""Run: python -m uvicorn hearth.api.app:app --host 0.0.0.0 --port 8000."""
import asyncio
from contextlib import asynccontextmanager, suppress
import os
from pathlib import Path

import anyio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from hearth_schema import (Alert, AlertMessage, Event, EventMessage, Heartbeat,
                           HelpRequest, NurseAction, NursePreferences, NurseProfile,
                           NurseSnapshot, NurseSnapshotMessage,
                           ObservationBatch, ObservationMessage, RoomSnapshot,
                           SourceStatus, SourceStatusMessage)
from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService
from hearth.monitor.appointments import Appointments, Appointment, AppointmentInput, CheckIn
from hearth.sim.mount import mount_simulation
from hearth.api.face_demo import create_face_router
from hearth.api.camera_preview import create_preview_router


def create_app(config: MonitorConfig | None = None, *, service: MonitorService | None = None,
               appointments_path: str | Path | None = None, with_simulation: bool = True) -> FastAPI:
    monitor = service or MonitorService(config or MonitorConfig.load(os.environ.get("HEARTH_CONFIG")))
    appointments = Appointments(monitor, appointments_path or (
        ":memory:" if service else os.environ.get("HEARTH_APPOINTMENTS_DB",
        str(Path(__file__).resolve().parents[3] / "data" / "appointments.sqlite"))))

    simulation = None

    async def check_health():
        while True:
            await asyncio.sleep(monitor.config.health_poll_seconds)
            if simulation is not None:
                with simulation.watch() as raised:
                    monitor.check_health()
                simulation.observe(raised)
            else:
                monitor.check_health()
            appointments.check_due()

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(check_health())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if simulation is not None:
                await simulation.close()

    app = FastAPI(title="Hearth Monitoring Core", version="0.1.0", lifespan=lifespan)
    app.state.monitor = monitor
    app.include_router(create_preview_router(monitor))
    app.state.appointments = appointments
    app.include_router(create_face_router(appointments, monitor))

    @app.get("/face-check-in", include_in_schema=False)
    async def face_check_in_page():
        return FileResponse(Path(__file__).with_name("static") / "face-check-in.html")

    def appointment_action(action):
        try:
            return action()
        except KeyError as exc:
            raise HTTPException(404, "Appointment not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/appointments", response_model=Appointment, status_code=201)
    async def create_appointment(body: AppointmentInput):
        return appointment_action(lambda: appointments.create(body))

    @app.get("/appointments", response_model=list[Appointment])
    async def list_appointments():
        appointments.check_due()
        return list(appointments.items.values())

    @app.get("/appointments/{appointment_id}", response_model=Appointment)
    async def get_appointment(appointment_id: str):
        appointments.check_due()
        return appointment_action(lambda: appointments.items[appointment_id])

    @app.put("/appointments/{appointment_id}", response_model=Appointment)
    async def reschedule_appointment(appointment_id: str, body: AppointmentInput):
        return appointment_action(lambda: appointments.update(appointment_id, body))

    @app.post("/appointments/{appointment_id}/cancel", response_model=Appointment)
    async def cancel_appointment(appointment_id: str):
        return appointment_action(lambda: appointments.update(appointment_id))

    @app.post("/appointments/{appointment_id}/check-in", response_model=Appointment)
    async def check_in_appointment(appointment_id: str, body: CheckIn):
        return appointment_action(lambda: appointments.check_in(appointment_id, body))
    if with_simulation:
        # The simulator and evaluator (Person 3) feed this same service through ingest().
        simulation = mount_simulation(app, monitor)
        app.state.sim_controller = simulation.scripted
        scripted = simulation.scripted
        monitor.observation_clock_paused = lambda key: (
            key[1:3] == ("simulation", "sim") and scripted.state == "paused"
            and key[4] == scripted.status().run_id)
    app.add_middleware(CORSMiddleware, allow_origins=monitor.config.cors_origins,
                       allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type"])
    static = Path(__file__).with_name("static")
    app.mount("/assets", StaticFiles(directory=static), name="assets")

    @app.get("/", include_in_schema=False)
    async def home():
        return RedirectResponse("http://127.0.0.1:5173/")

    @app.get("/nurses", include_in_schema=False)
    async def nurse_page():
        return RedirectResponse("http://127.0.0.1:5173/nurses")

    @app.get("/room", include_in_schema=False)
    async def room_page():
        return RedirectResponse("http://127.0.0.1:5173/room")

    @app.get("/rooms")
    async def rooms():
        return {"room_ids": monitor.config.room_ids, "gesture_hold_seconds": monitor.config.gesture_hold_seconds}

    @app.get("/health")
    async def health():
        return {"status": "ok", "room_id": monitor.config.room_id, "schema_version": "1"}

    @app.post("/ingest/observations")
    async def observations(batch: ObservationBatch):
        try:
            return monitor.ingest(batch.observations)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/ingest/status")
    async def ingest_status():
        return {"counts": monitor.ingest_totals.copy(),
                "edge_observation_max_age_seconds": monitor.config.edge_observation_max_age_seconds,
                "edge_future_tolerance_seconds": monitor.config.edge_future_tolerance_seconds,
                "clock": "backend_utc", "scope": "since_backend_start"}

    @app.post("/ingest/heartbeat", response_model=SourceStatus)
    async def heartbeat(body: Heartbeat):
        try:
            monitor.heartbeat(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        from hearth.monitor.detector import source_key
        return monitor.health[source_key(body)].status

    def check_room(room_id: str):
        if room_id not in monitor.config.room_ids:
            raise HTTPException(404, "Room not configured")

    @app.get("/rooms/{room_id}/snapshot", response_model=RoomSnapshot)
    async def snapshot(room_id: str):
        check_room(room_id)
        return monitor.snapshot(room_id)

    @app.get("/sources", response_model=list[SourceStatus])
    async def sources():
        monitor.check_health()
        return [entry.status for entry in monitor.health.values()]

    def matches(record, source: str | None, run_id: str | None):
        return (source is None or record.source == source) and (run_id is None or record.run_id == run_id)

    @app.get("/events", response_model=list[Event])
    async def events(source: str | None = None, run_id: str | None = None):
        monitor.check_health()
        return [e for e in monitor.events if matches(e, source, run_id)]

    @app.get("/events/export")
    async def export(source: str | None = None, run_id: str | None = None):
        monitor.check_health()
        data = "".join(e.model_dump_json() + "\n" for e in monitor.events if matches(e, source, run_id))
        return Response(data, media_type="application/x-ndjson",
                        headers={"Content-Disposition": 'attachment; filename="hearth-events.jsonl"'})

    @app.get("/alerts", response_model=list[Alert])
    async def alerts(source: str | None = None, run_id: str | None = None):
        return [a for a in monitor.alerts.values() if matches(a, source, run_id)]

    def transition(alert_id: str, action: str, nurse_id: str | None = None):
        try:
            return monitor.transition_alert(alert_id, action, nurse_id)
        except KeyError as exc:
            raise HTTPException(404, "Alert not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/alerts/{alert_id}/ack", response_model=Alert)
    async def acknowledge(alert_id: str):
        return transition(alert_id, "ack")

    @app.post("/alerts/{alert_id}/resolve", response_model=Alert)
    async def resolve(alert_id: str, body: NurseAction | None = None):
        return transition(alert_id, "resolve", body.nurse_id if body else None)

    @app.post("/help-requests", response_model=Alert)
    async def help_request(body: HelpRequest):
        try:
            return monitor.request_help(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/nurse-sessions", response_model=NurseProfile)
    async def create_nurse(body: NursePreferences):
        try:
            return monitor.save_nurse(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.put("/nurse-sessions/{nurse_id}", response_model=NurseProfile)
    async def update_nurse(nurse_id: str, body: NursePreferences):
        try:
            return monitor.save_nurse(body, nurse_id)
        except KeyError as exc:
            raise HTTPException(404, "Nurse session not found; create a new session") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/nurse-sessions/{nurse_id}/snapshot", response_model=NurseSnapshot)
    async def nurse_snapshot(nurse_id: str):
        try:
            return monitor.nurse_snapshot(nurse_id)
        except KeyError as exc:
            raise HTTPException(404, "Nurse session not found; create a new session") from exc

    @app.post("/alerts/{alert_id}/accept", response_model=Alert)
    async def accept_alert(alert_id: str, body: NurseAction):
        try:
            return monitor.accept_alert(alert_id, body.nurse_id)
        except KeyError as exc:
            raise HTTPException(404, "Alert or nurse session not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    async def stream(websocket: WebSocket, queue: asyncio.Queue, initial_payloads: list[dict]):
        # An anyio task group (not raw asyncio tasks) so cancellation composes with
        # Starlette's own cancel scopes when a client disconnects mid-send.
        async def send_messages(group):
            for message in initial_payloads:
                await websocket.send_json(message)
            while True:
                message = await queue.get()
                if message is None:
                    await websocket.close(code=1013, reason="Reconnect to refresh state")
                    break
                await websocket.send_json(message)
            group.cancel_scope.cancel()

        async def receive_disconnect(group):
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
            group.cancel_scope.cancel()

        try:
            with suppress(WebSocketDisconnect):
                async with anyio.create_task_group() as group:
                    group.start_soon(send_messages, group)
                    group.start_soon(receive_disconnect, group)
        finally:
            monitor.unsubscribe(queue)

    @app.websocket("/ws/nurse/{nurse_id}")
    async def nurse_feed(websocket: WebSocket, nurse_id: str):
        if nurse_id not in monitor.nurses:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        snapshot = monitor.nurse_snapshot(nurse_id)
        queue = monitor.subscribe(nurse_id=nurse_id)
        await stream(websocket, queue, [NurseSnapshotMessage(data=snapshot).model_dump(mode="json")])

    @app.websocket("/ws/room/{room_id}")
    async def websocket_feed(websocket: WebSocket, room_id: str):
        if room_id not in monitor.config.room_ids:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        # Snapshot and subscription happen without an await: no event can fall between them.
        snapshot = monitor.snapshot(room_id)
        queue = monitor.subscribe(room_id=room_id)
        initial = ([SourceStatusMessage(data=s) for s in snapshot.sources]
                   + [ObservationMessage(data=o) for o in snapshot.observations]
                   + [EventMessage(data=e) for e in snapshot.events]
                   + [AlertMessage(data=a) for a in snapshot.alerts])
        initial_payloads = [m.model_dump(mode="json") for m in initial]

        await stream(websocket, queue, initial_payloads)

    return app


app = create_app()

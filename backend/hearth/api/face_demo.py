"""Loopback-only, explicit opt-in camera demo endpoints."""
from pathlib import Path
from datetime import timedelta
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from hearth.monitor.appointments import CheckIn
from hearth.monitor.patient_camera import PatientCamera
from hearth_schema import Observation


class Enroll(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    alias: str = Field(min_length=1, max_length=40)
    consent: bool
    identify_alerts: bool = False
    frames: list[str] = Field(min_length=3, max_length=3)
    appointment_id: str | None = None


class CameraSession(BaseModel):
    room_id: str


class CameraFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    frame: str = Field(max_length=1_100_000)


class CameraObservation(CameraFrame):
    observation: Observation


def create_face_router(appointments, monitor, demo=None):
    setup_error = "Disabled. Enable HEARTH_FACE_DEMO=1 and provide HEARTH_FACE_KEY to use patient enrollment."
    if demo is None and os.environ.get("HEARTH_FACE_DEMO") == "1":
        try:
            from hearth.monitor.face_demo import FaceDemo
            from hearth.perception.face_demo import FaceExtractor
            key = os.environ.get("HEARTH_FACE_KEY")
            if not key:
                raise ValueError("Encryption key is required")
            demo = FaceDemo(Path(__file__).resolve().parents[3] / "data" / "face-demo.sqlite",
                            key, FaceExtractor())
        except Exception:
            # Do not expose key material or decrypted records through errors.
            setup_error = "Face demo unavailable: check optional dependencies, downloaded models, and the original encryption key."

    camera = PatientCamera(demo, monitor.config) if demo is not None else None

    def local_only(request: Request):
        if (request.client is None or request.client.host not in {"127.0.0.1", "::1"}
                or request.url.hostname not in {"localhost", "127.0.0.1", "::1"}):
            raise HTTPException(403, "Face demo is available only on the local laptop")
        if request.headers.get("origin") not in {None, str(request.base_url).rstrip("/")}:
            raise HTTPException(403, "Use the same local page for the face demo")
        if request.method != "GET" and request.headers.get("x-hearth-face-demo") != "1":
            raise HTTPException(403, "Use the face demo controls")

    def active():
        if demo is None:
            raise HTTPException(503, setup_error)
        return demo

    router = APIRouter(prefix="/face-demo", tags=["patient face demo"], dependencies=[Depends(local_only)])

    def action(fn):
        try:
            return fn()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/status")
    async def status():
        return {"enabled": demo is not None, "message": "Ready for consenting demo patients" if demo else setup_error}

    @router.get("/profiles")
    async def profiles():
        service = active()
        return [service.public(r) for r in service.profiles()]

    @router.post("/enroll")
    async def enroll(body: Enroll):
        service = active()
        token = None
        if any(len(frame) > 1_100_000 for frame in body.frames):
            raise HTTPException(413, "Enrollment frames are too large")
        if body.appointment_id:
            item = appointments.items.get(body.appointment_id)
            if item is None or not item.demo or item.status not in {"scheduled", "overdue"}:
                raise HTTPException(422, "Link an open fictional demo appointment only")
            token = item.check_in_token
        return action(lambda: service.enroll(body.alias, body.frames, body.consent, body.appointment_id, token, identify_alerts=body.identify_alerts))

    @router.delete("/profiles/{profile_id}")
    async def remove(profile_id: str):
        active().delete(profile_id)
        return {"deleted": True}

    @router.post("/sessions")
    async def session(body: CameraSession):
        action(lambda: monitor.validate_room(body.room_id))
        return {"session_id": action(lambda: active().start(body.room_id))}

    @router.delete("/sessions/{session_id}")
    async def stop(session_id: str):
        active().sessions.pop(session_id, None)
        return {"stopped": True}

    def check_arrival(result):
        if result["status"] != "recognized":
            return
        record = next((r for r in active().profiles() if r["profile_id"] == result["profile"]["profile_id"]), None)
        if record is None:
            return
        links = dict(record.get("appointment_links", {}))
        if record.get("appointment_id"):
            links.setdefault(record["appointment_id"], record.get("appointment_token"))
        checked = []
        for appointment_id, token in links.items():
            item = appointments.items.get(appointment_id)
            if (item and item.demo and item.check_in_token == token and item.room_id == result["room_id"]
                    and item.status in {"scheduled", "overdue"}
                    and monitor.utcnow() >= item.scheduled_at - timedelta(minutes=15)):
                appointments.check_in(item.appointment_id, CheckIn(method="token", token=token), verified_demo_face=True)
                checked.append(item.appointment_id)
        if checked:
            result["checked_in"] = checked[0]
            result["checked_in_appointments"] = checked

    @router.post("/profiles/{profile_id}/appointments/{appointment_id}")
    async def link_appointment(profile_id: str, appointment_id: str):
        item = appointments.items.get(appointment_id)
        if item is None or not item.demo or item.status not in {"scheduled", "overdue"}:
            raise HTTPException(422, "Link an open demo appointment only")
        action(lambda: active().link_appointment(profile_id, appointment_id, item.check_in_token))
        return {"linked": True}

    @router.post("/recognize")
    async def recognize(body: CameraFrame):
        result = action(lambda: active().recognize(body.session_id, body.frame))
        check_arrival(result)
        return result

    @router.post("/observe")
    async def observe(body: CameraObservation):
        active()
        obs = body.observation
        action(lambda: monitor.validate_room(obs.room_id))
        age = (monitor.utcnow() - obs.timestamp).total_seconds()
        if not -monitor.config.edge_future_tolerance_seconds <= age <= monitor.config.edge_observation_max_age_seconds:
            raise HTTPException(422, "Camera frame is outside live timestamp limits")
        result = action(lambda: camera.process(body.session_id, obs, body.frame))
        labels = {}
        if "profile" in result:
            labels[obs.obs_id] = dict(resident_label=result["profile"]["alias"],
                                     patient_profile_id=result["profile"]["profile_id"],
                                     identity_method="face_match" if result["status"] == "recognized" else "recent_face_body_track",
                                     identity_demo=True)
        counts = monitor.ingest([obs], patient_labels=labels)
        if counts["accepted"]:
            check_arrival(result)
        return {**counts, "identity": result}

    return router

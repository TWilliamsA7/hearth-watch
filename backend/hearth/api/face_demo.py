"""Loopback-only patient recognition endpoints: opt-in enrollment, recognition, Jetson templates."""
from pathlib import Path
from datetime import timedelta
import os
import asyncio
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from hearth.monitor.appointments import CheckIn
from hearth.monitor.edge_identity import EdgeFaceTemplate, EdgeIdentities
from hearth.monitor.patient_camera import PatientCamera, TrackedBody
from starlette.concurrency import run_in_threadpool
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
    body: TrackedBody | None = None


DATA = Path(__file__).resolve().parents[3] / "data"


def face_key(path: Path = DATA / "face.key") -> str:
    """HEARTH_FACE_KEY, else the key kept in data/face.key, made on first use.

    Enrollments are encrypted with it: keep the file, as a new key cannot read old ones."""
    if key := os.environ.get("HEARTH_FACE_KEY"):
        return key.strip()
    if path.is_file():
        return path.read_text(encoding="ascii").strip()
    from cryptography.fernet import Fernet
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key().decode()
    # Readable by this user only where the platform supports it.
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="ascii") as out:
        out.write(key + "\n")
    return key


def load_recognition(database: Path = DATA / "face-demo.sqlite", key_file: Path = DATA / "face.key", *,
                     extractor_factory=None):
    """(FaceDemo, None) when recognition can run, else (None, what to do about it).

    On by default; HEARTH_FACE_DEMO=0 turns it off. Messages never include key material."""
    if os.environ.get("HEARTH_FACE_DEMO") == "0":
        return None, "Recognition is turned off on this server (HEARTH_FACE_DEMO=0)."
    try:
        import cryptography, cv2, numpy, PIL  # noqa: F401  (the face-demo extra)
    except ImportError:
        return None, ("Recognition needs the face-demo packages. Run "
                      ".venv/Scripts/python.exe -m pip install -e \"./backend[face-demo]\" "
                      "(scripts/start.ps1 does this), then restart.")
    from hearth.monitor.face_demo import FaceDemo
    from hearth.perception.face_demo import FaceExtractor
    try:
        extractor = (extractor_factory or FaceExtractor)()
    except ValueError as exc:  # missing or corrupted models
        return None, f"{exc} (scripts/start.ps1 downloads them), then restart."
    except Exception:  # an OpenCV build without YuNet/SFace
        return None, "OpenCV could not load the face models; reinstall the face-demo packages."
    try:
        key = face_key(key_file)
    except (OSError, ValueError):
        return None, "The recognition key in data/face.key could not be read or created."
    try:
        return FaceDemo(database, key, extractor), None
    except Exception:
        return None, ("The recognition key does not match the existing enrollments in data/face-demo.sqlite. "
                      "Restore the original key (data/face.key or HEARTH_FACE_KEY), or delete that database to start over.")


def create_face_router(appointments, monitor, demo=None, edge_token=None, *, load=load_recognition):
    setup_error = None
    if demo is None:
        demo, setup_error = load()

    camera = PatientCamera(demo, monitor.config) if demo is not None else None
    # An empty token is no token: the edge route stays closed.
    edge_token = (edge_token if edge_token is not None else os.environ.get("HEARTH_EDGE_FACE_TOKEN")) or None
    edge = EdgeIdentities(demo, monitor) if demo is not None else None

    def local_only(request: Request):
        if (request.client is None or request.client.host not in {"127.0.0.1", "::1"}
                or request.url.hostname not in {"localhost", "127.0.0.1", "::1"}):
            raise HTTPException(403, "Recognition is available only on the local care computer")
        if request.headers.get("origin") not in {None, str(request.base_url).rstrip("/")}:
            raise HTTPException(403, "Use the same local page for recognition")
        if request.method != "GET" and request.headers.get("x-hearth-face-demo") != "1":
            raise HTTPException(403, "Use the recognition controls")

    def active():
        if demo is None:
            raise HTTPException(503, setup_error)
        return demo

    inference_lock = asyncio.Lock()

    async def serialize_face_access(request: Request):
        # Bound work rather than queueing old camera frames behind a busy recognizer.
        # All enrollment/session access uses the same guard as the worker thread.
        if inference_lock.locked() and request.url.path.endswith(("/observe", "/recognize", "/edge-embeddings")):
            raise HTTPException(503, "Recognition busy; keep monitoring without identity and retry a fresh frame")
        async with inference_lock:
            yield

    router = APIRouter(prefix="/face-demo", tags=["patient face demo"],
                       dependencies=[Depends(local_only), Depends(serialize_face_access)])

    def edge_only(request: Request):
        # The Jetson is not on loopback, so it proves itself with a shared token instead.
        # It sends face templates only, never frames. No browser page may call this.
        supplied = request.headers.get("authorization", "")
        if (edge_token is None or request.headers.get("origin") is not None
                or not secrets.compare_digest(supplied.encode(), f"Bearer {edge_token}".encode())):
            raise HTTPException(403, "Edge face templates need HEARTH_EDGE_FACE_TOKEN")

    edge_router = APIRouter(prefix="/face-demo", tags=["patient face demo"],
                            dependencies=[Depends(edge_only), Depends(serialize_face_access)])

    def action(fn):
        try:
            return fn()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/status")
    async def status():
        return {"enabled": demo is not None, "message": "Ready to enroll residents" if demo else setup_error,
                "body_detectors": ["yolo", "posenet"],
                "transport": "loopback_only" + ("+edge_templates" if edge_token else ""),
                "identity_ttl_seconds": PatientCamera.IDENTITY_TTL}

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
            if item is None or item.status not in {"scheduled", "overdue"}:
                raise HTTPException(422, "Link an open appointment only")
            token = item.check_in_token
        return await run_in_threadpool(action, lambda: service.enroll(body.alias, body.frames, body.consent, body.appointment_id, token, identify_alerts=body.identify_alerts))

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
        if result["status"] != "recognized" or "profile" not in result:
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
            if (item and item.check_in_token == token and item.room_id == result["room_id"]
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
        if item is None or item.status not in {"scheduled", "overdue"}:
            raise HTTPException(422, "Link an open appointment only")
        action(lambda: active().link_appointment(profile_id, appointment_id, item.check_in_token))
        return {"linked": True}

    @router.delete("/profiles/{profile_id}/appointments/{appointment_id}")
    async def unlink_appointment(profile_id: str, appointment_id: str):
        action(lambda: active().unlink_appointment(profile_id, appointment_id))
        return {"unlinked": True}

    @router.post("/recognize")
    async def recognize(body: CameraFrame):
        result = await run_in_threadpool(action, lambda: active().recognize(body.session_id, body.frame))
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
        result = await run_in_threadpool(action, lambda: camera.process(body.session_id, obs, body.frame, body=body.body))
        check_in_profile = result.pop("_check_in_profile", None)
        labels = {}
        if "profile" in result:
            labels[obs.obs_id] = dict(resident_label=result["profile"]["alias"],
                                     patient_profile_id=result["profile"]["profile_id"],
                                     identity_method="face_match" if result["status"] == "recognized" else "recent_face_body_track",
                                     identity_demo=True)
            box = body.body.bbox if body.body is not None else (obs.posture_prediction.bbox if obs.posture_prediction else None)
            hand = obs.hand_gesture_prediction
            hand_box = hand.bbox if hand is not None else None
            labels[obs.obs_id]["_hand_matches_body"] = bool(box is not None and hand_box is not None
                and box.x1 <= (hand_box.x1 + hand_box.x2)/2 <= box.x2
                and box.y1 <= (hand_box.y1 + hand_box.y2)/2 <= box.y2)
        counts = monitor.ingest([obs], patient_labels=labels)
        if counts["accepted"]:
            arrival = ({"status":"recognized", "profile":check_in_profile, "room_id":obs.room_id}
                       if check_in_profile is not None else result)
            check_arrival(arrival)
            for key in ("checked_in", "checked_in_appointments"):
                if key in arrival:
                    result[key] = arrival[key]
        return {**counts, "identity": result}

    @edge_router.post("/edge-embeddings")
    async def edge_embedding(body: EdgeFaceTemplate):
        active()
        age = (monitor.utcnow() - body.timestamp).total_seconds()
        if not -monitor.config.edge_future_tolerance_seconds <= age <= monitor.config.edge_observation_max_age_seconds:
            raise HTTPException(422, "Face template is outside live timestamp limits")
        result = await run_in_threadpool(action, lambda: edge.submit(body))
        check_arrival(result)
        # The edge needs only progress, not who it is: a name goes no further than incidents.
        return {key: result[key] for key in ("status", "matches", "checked_in") if key in result}

    outer = APIRouter()
    outer.include_router(router)
    outer.include_router(edge_router)
    return outer

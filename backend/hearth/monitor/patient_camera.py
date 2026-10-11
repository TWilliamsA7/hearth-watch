"""Single-person staged camera demo; identity never changes fall classification."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from hearth_schema import BoundingBox
from .face_demo import FaceNotVisible


class TrackedBody(BaseModel):
    """Optional same-frame poseNet evidence, separate from the shared sensor schema."""
    model_config = ConfigDict(extra="forbid")
    detector: Literal["posenet"]
    obs_id: str = Field(min_length=1, max_length=200)
    track_id: str = Field(min_length=1, max_length=100)
    detection_count: int = Field(ge=0, le=100)
    bbox: BoundingBox | None = None

    @model_validator(mode="after")
    def require_box(self):
        if self.detection_count == 1 and self.bbox is None:
            raise ValueError("A single tracked body requires a normalized bounding box")
        return self


def overlap(a, b):
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    union = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - intersection
    return intersection / union if union else 0


class PatientCamera:
    # Explicit demo limits, not a multi-person identity tracker.
    IDENTITY_TTL = 8.0
    MIN_BODY_IOU = .10

    def __init__(self, faces, config):
        self.faces, self.config = faces, config

    def process(self, session_id, observation, frame, *, body=None):
        faces = self.faces
        records = {r["profile_id"]: r for r in faces.profiles()}
        session = faces.sessions.get(session_id)
        if session is None or session["expires"] <= faces.clock():
            raise ValueError("Start a new camera recognition session")
        if observation.source_type != "edge" or not observation.run_id or observation.room_id != session["room_id"]:
            raise ValueError("Use a live camera run in this session's room")
        if body is not None and body.obs_id != observation.obs_id:
            raise ValueError("Body evidence must belong to this exact observation")
        context = (observation.source, observation.run_id)
        if session.get("camera_context", context) != context:
            raise ValueError("Camera source or run changed; start a new session")
        session["camera_context"] = context
        if session.get("last_observation_at") is not None and observation.timestamp <= session["last_observation_at"]:
            raise ValueError("Camera observations must arrive in timestamp order")
        session["last_observation_at"] = observation.timestamp
        track = (observation.person_id, body.track_id if body is not None else None)
        if session.get("camera_track", track) != track:
            session.pop("body", None)
            session.pop("patient", None)
            session["profile_id"], session["count"] = None, 0
        session["camera_track"] = track
        previous = session.get("body")
        prediction = observation.posture_prediction
        if body is not None:
            # Never manufacture a YOLO prediction or a posture confidence for poseNet.
            reliable = (observation.person_detected and body.detection_count == 1
                        and body.bbox is not None and bool(observation.keypoints)
                        and observation.posture_prediction is None)
            box = body.bbox
        else:
            reliable = (observation.person_detected and prediction is not None and prediction.detection_count == 1
                        and prediction.bbox is not None and prediction.confidence is not None
                        and prediction.confidence >= self.config.model_confidence_min
                        and prediction.model_id == self.config.fall_model_id)
            box = prediction.bbox if prediction is not None else None
        now = faces.clock()
        continuous = (reliable and previous and previous["box"] is not None
                      and 0 <= now - previous["received"] <= self.config.max_observation_gap_seconds
                      and (observation.timestamp - previous["timestamp"]).total_seconds() <= self.config.max_observation_gap_seconds
                      and overlap(previous["box"], box) >= self.MIN_BODY_IOU)
        if not continuous:
            session.pop("patient", None)
            session["profile_id"], session["count"] = None, 0
        session["body"] = dict(timestamp=observation.timestamp, received=now,
                               box=box if reliable else None)
        if not reliable:
            return {"status": "identity_unconfirmed"}
        try:
            result = faces.recognize(session_id, frame, body_box=box)
        except FaceNotVisible:
            result = {"status": "face_not_visible"}
        except ValueError as exc:
            session.pop("patient", None)
            return {"status": "identity_unconfirmed", "reason": str(exc)}
        if result["status"] == "recognized":
            session["patient"] = dict(profile_id=result["profile"]["profile_id"], until=now+self.IDENTITY_TTL)
        elif result["status"] != "face_not_visible":
            # Unknown/ambiguous faces or a new candidate may not inherit an old name.
            session.pop("patient", None)
        patient = session.get("patient")
        record = records.get(patient["profile_id"]) if patient and patient["until"] > now else None
        if record is None or not record.get("identify_alerts") or record.get("consent_version") != faces.CONSENT_VERSION:
            session.pop("patient", None)
            response = {"status": "identity_unconfirmed" if result["status"] == "recognized" else result["status"]}
            # A fresh match may confirm a linked appointment without consenting to
            # names on incidents. This private value is removed by the API route.
            if record is not None and result["status"] == "recognized":
                response["_check_in_profile"] = faces.public(record)
            return response
        return {"status": "recognized" if result["status"] == "recognized" else "tracked",
                "profile": faces.public(record), "room_id": observation.room_id,
                "identity_expires_at": patient["until"]}

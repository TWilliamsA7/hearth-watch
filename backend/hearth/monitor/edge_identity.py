"""Opt-in names for the Jetson's anonymous people, from face templates it computes itself.

The Jetson runs SFace only when res10 finds a face inside one tracked person's head box, and
sends that person's 128-value template (never pixels) with their anonymous person_id. Matching,
consent and enrollment stay here. A name belongs to one track, (room, source, run_id,
person_id): a new person_id, the track's "not detected" end, or the source going offline clears
it. The live edge sends no run_id (it would change the source on every restart), and its
person ids restart at p1, so the offline clear is what stops a restarted camera's p1 inheriting
the previous run's name: a restart always outlasts the source timeout.

The laptop camera rules apply per track: three consecutive matching checks confirm a name, an
unknown or ambiguous face clears it, it lasts IDENTITY_TTL after the last match (so a fall
with the face turned away can still be named), and only enrollments that allowed named alerts
are shown. One enrollment names at most one live track: if two tracks match the same face,
neither keeps the name. Identity never changes fall or gesture classification.
"""
import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from hearth_schema import BoundingBox

from .patient_camera import PatientCamera


class EdgeFaceTemplate(BaseModel):
    """One face check on the Jetson, for one tracked person. Not part of the sensor schema."""
    model_config = ConfigDict(extra="forbid")
    detector: Literal["res10+sface"]
    source: str = Field(min_length=1, max_length=100)
    room_id: str
    run_id: str | None = Field(default=None, min_length=1, max_length=100)  # as on its observations
    person_id: str = Field(min_length=1, max_length=100)
    obs_id: str = Field(min_length=1, max_length=200)  # the frame the face was checked on
    timestamp: datetime
    face_box: BoundingBox  # normalized to the full frame; inside the person's head box
    embedding: list[float] = Field(min_length=128, max_length=128)


class EdgeIdentities:
    IDENTITY_TTL = PatientCamera.IDENTITY_TTL
    CONTINUE_SECONDS = 3.0   # a gap longer than this restarts the three-match count

    def __init__(self, faces, monitor):
        self.faces, self.monitor = faces, monitor
        self.tracks = {}     # (room, source, run_id, person_id) -> state dict
        faces.listeners.append(self.tracks.clear)
        monitor.observation_listeners.append(self.observe)
        monitor.event_listeners.append(self.source_event)
        monitor.incident_labeler = self.label
        monitor.identity_of = self.identity

    @staticmethod
    def key(record):
        return (record.room_id, record.source, record.run_id, record.person_id)

    def observe(self, obs):
        """A person's track ending on the edge ends any name it had."""
        if obs.source_type == "edge" and obs.person_id is not None and not obs.person_detected:
            self.tracks.pop(self.key(obs), None)

    def source_event(self, event):
        if event.event_type == "source_offline":
            for key in [k for k in self.tracks if k[:3] == (event.room_id, event.source, event.run_id)]:
                del self.tracks[key]

    def submit(self, item: EdgeFaceTemplate) -> dict:
        norm = math.sqrt(sum(v * v for v in item.embedding))
        if not all(math.isfinite(v) for v in item.embedding) or not .9 <= norm <= 1.1:
            raise ValueError("Send an L2-normalized SFace template")
        feature = [v / norm for v in item.embedding]
        self.monitor.validate_room(item.room_id)
        latest = self.monitor.observations.get(
            (item.room_id, "edge", item.source, None, item.run_id, item.person_id))
        gap = self.monitor.config.max_observation_gap_seconds
        # The face check runs a little behind the camera and is sent apart from the
        # observation batches, so allow one observation gap either way.
        if (latest is None or not latest.person_detected
                or abs((item.timestamp - latest.timestamp).total_seconds()) > gap):
            return {"status": "track_unknown"}
        key = self.key(item)
        state = self.tracks.get(key)
        if state is not None and item.timestamp <= state["timestamp"]:
            raise ValueError("Face templates must arrive in timestamp order for each person")
        now = self.faces.clock()
        records = self.faces.profiles()
        best = self.faces.match(feature, records)
        if best is None:
            self.tracks[key] = dict(timestamp=item.timestamp, candidate=None, count=0, at=now, named=None)
            return {"status": "unknown", "matches": 0}
        score, record = best
        continuing = (state is not None and state["candidate"] == record["profile_id"]
                      and now - state["at"] <= self.CONTINUE_SECONDS)
        count = state["count"] + 1 if continuing else 1
        named = state["named"] if continuing else None
        if count >= self.faces.REQUIRED_MATCHES:
            others = [k for k, s in self.tracks.items() if k != key and s["named"]
                      and s["named"]["profile_id"] == record["profile_id"] and s["named"]["until"] > now]
            if others:
                for other in others:
                    self.tracks[other]["named"] = None
                self.tracks[key] = dict(timestamp=item.timestamp, candidate=None, count=0, at=now, named=None)
                return {"status": "ambiguous_track", "matches": 0}
            # Only what labelling needs, so naming an incident never touches the database.
            named = dict(profile_id=record["profile_id"], alias=record["alias"], until=now + self.IDENTITY_TTL,
                         expires_at=record["expires_at"], show=bool(record.get("identify_alerts"))
                         and record.get("consent_version") == self.faces.CONSENT_VERSION)
        self.tracks[key] = dict(timestamp=item.timestamp, candidate=record["profile_id"], count=count,
                                at=now, named=named)
        if count < self.faces.REQUIRED_MATCHES:
            return {"status": "confirming", "matches": count}
        return {"status": "recognized", "matches": count, "similarity": round(score, 3),
                "profile": self.faces.public(record), "room_id": item.room_id, "show_name": named["show"]}

    def identity(self, obs) -> dict:
        """Whether this observation's person is recognized now, and their name if they agreed to be named."""
        if obs.source_type != "edge" or obs.person_id is None:
            return {"recognized": False, "name": None}
        state = self.tracks.get(self.key(obs))
        named = state and state["named"]
        now = self.faces.clock()
        if not named or named["until"] <= now or named["expires_at"] <= now:
            return {"recognized": False, "name": None}
        return {"recognized": True, "name": named["alias"] if named["show"] else None}

    def label(self, obs):
        """Incident metadata naming this observation's person, or None."""
        if obs.source_type != "edge" or obs.person_id is None:
            return None
        state = self.tracks.get(self.key(obs))
        named = state and state["named"]
        now = self.faces.clock()
        if not named or not named["show"] or named["until"] <= now or named["expires_at"] <= now:
            return None
        return dict(resident_label=named["alias"], patient_profile_id=named["profile_id"],
                    identity_method="edge_face_track", identity_demo=True)

"""Opt-in volunteer recognition. Stores encrypted templates, never camera frames."""
import json
import math
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


def similarity(a, b):
    if len(a) != 128 or len(b) != 128:
        raise ValueError("Invalid SFace template")
    return sum(x * y for x, y in zip(a, b))


class FaceNotVisible(ValueError):
    """No face in this frame; a continuous body track may retain a recent match."""


class FaceDemo:
    CONSENT_VERSION = "demo-patient-v2"
    THRESHOLD = .55
    MARGIN = .10
    REQUIRED_MATCHES = 3
    RETENTION_SECONDS = 24 * 60 * 60

    def __init__(self, path, key, extractor, *, clock=time.time):
        from cryptography.fernet import Fernet
        self.cipher = Fernet(key)
        self.extractor, self.clock = extractor, clock
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.execute("CREATE TABLE IF NOT EXISTS face_enrollments (id TEXT PRIMARY KEY, payload BLOB NOT NULL)")
        self.sessions = {}
        self.profiles()  # Reject a wrong key before allowing any enrollment.

    def profiles(self):
        profiles = []
        for profile_id, payload in self.db.execute("SELECT id, payload FROM face_enrollments").fetchall():
            record = json.loads(self.cipher.decrypt(payload))
            if record["expires_at"] <= self.clock():
                self.delete(profile_id)
            else:
                profiles.append(record)
        return profiles

    @staticmethod
    def public(record):
        return {k: v for k, v in record.items() if k not in {"template", "appointment_token", "appointment_links"}}

    def enroll(self, alias, frames, consent, appointment_id=None, appointment_token=None, *, identify_alerts=False):
        if consent is not True:
            raise ValueError("Explicit patient consent is required")
        records = self.profiles()
        if len(records) >= 20:
            raise ValueError("Demo enrollment limit reached; delete unused profiles")
        if len(frames) != 3:
            raise ValueError("Capture three enrollment views")
        if len(set(frames)) != 3:
            raise ValueError("Capture three separate camera frames")
        vectors = [self.extractor.extract(frame) for frame in frames]
        if any(similarity(vectors[0], v) < self.THRESHOLD for v in vectors[1:]):
            raise ValueError("Enrollment views do not agree; retry with one person")
        average = [sum(values) / len(vectors) for values in zip(*vectors)]
        norm = math.sqrt(sum(value * value for value in average))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid enrollment template")
        template = [value / norm for value in average]
        if any(similarity(template, record["template"]) >= self.THRESHOLD for record in records):
            raise ValueError("This face appears enrolled already; delete the previous enrollment first")
        now = self.clock()
        record = dict(profile_id=str(uuid4()), alias=alias, consent_version=self.CONSENT_VERSION,
                      consented_at=now, expires_at=now + self.RETENTION_SECONDS,
                      appointment_id=appointment_id, appointment_token=appointment_token,
                      identify_alerts=identify_alerts, template=template)
        with self.db:
            self.db.execute("INSERT INTO face_enrollments VALUES (?, ?)",
                            (record["profile_id"], self.cipher.encrypt(json.dumps(record).encode())))
        self.sessions.clear()
        return self.public(record)

    def delete(self, profile_id):
        with self.db:
            self.db.execute("DELETE FROM face_enrollments WHERE id=?", (profile_id,))
        self.sessions.clear()

    def link_appointment(self, profile_id, appointment_id, token):
        record = next((r for r in self.profiles() if r["profile_id"] == profile_id), None)
        if record is None:
            raise ValueError("Patient enrollment not found")
        record.setdefault("appointment_links", {})[appointment_id] = token
        with self.db:
            self.db.execute("UPDATE face_enrollments SET payload=? WHERE id=?",
                            (self.cipher.encrypt(json.dumps(record).encode()), profile_id))

    def start(self, room_id):
        self.sessions = {key: value for key, value in self.sessions.items() if value["expires"] > self.clock()}
        if len(self.sessions) >= 10:
            raise ValueError("Too many camera sessions; stop an existing session")
        token = str(uuid4())
        self.sessions[token] = dict(room_id=room_id, expires=self.clock() + 1800,
                                    last=0., profile_id=None, count=0, frame=None)
        return token

    def recognize(self, session_id, frame, *, body_box=None):
        import hashlib
        state = self.sessions.get(session_id)
        if state is None or state["expires"] <= self.clock():
            self.sessions.pop(session_id, None)
            raise ValueError("Start a new camera recognition session")
        # Purging an expired enrollment also invalidates existing recognition evidence.
        records = self.profiles()
        if session_id not in self.sessions:
            raise ValueError("Enrollment changed; restart recognition")
        now = self.clock()
        if state["last"] and now - state["last"] < .2:
            raise ValueError("Wait between camera frames")
        digest = hashlib.sha256(frame.encode()).hexdigest()
        repeated = state["frame"] == digest
        previous = state["last"]
        state["last"], state["frame"] = now, digest
        try:
            feature = (self.extractor.extract(frame, body_box=body_box) if body_box is not None
                       else self.extractor.extract(frame))
        except ValueError:
            state["profile_id"], state["count"] = None, 0
            raise
        scored = sorted(((similarity(feature, r["template"]), r) for r in records), key=lambda x: x[0], reverse=True)
        if (not scored or scored[0][0] < self.THRESHOLD or
                (len(scored) > 1 and scored[0][0] - scored[1][0] < self.MARGIN)):
            state["profile_id"], state["count"] = None, 0
            return dict(status="unknown", matches=0)
        score, record = scored[0]
        if repeated:
            state["profile_id"], state["count"] = None, 0
            return dict(status="new_frame_required", matches=0)
        continuing = state["profile_id"] == record["profile_id"] and now - previous <= 3
        state["count"] = state["count"] + 1 if continuing else 1
        state["profile_id"] = record["profile_id"]
        if state["count"] < self.REQUIRED_MATCHES:
            return dict(status="confirming", matches=state["count"])
        return dict(status="recognized", matches=state["count"], similarity=round(score, 3),
                    profile=self.public(record), room_id=state["room_id"])

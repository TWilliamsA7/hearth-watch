"""Hearth shared contract (schema version 1).

Single source of truth for every message that crosses a component boundary.
JSON Schema files and frontend TypeScript types are generated from these models
(see shared/scripts/export_schema.py). Change the contract here, then regenerate.

Three separate record kinds — never mix them:
  * Observation  — what a sensor (Jetson or simulated camera) could see.
  * Event        — what the monitor concluded from observations.
  * GroundTruth  — what actually happened in a simulated scenario. Evaluation
                   only; the monitor must never read it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"

SourceType = Literal["edge", "simulation", "replay"]
"""edge = live Jetson; simulation = simulator; replay = recorded trace played back."""

# poseNet resnet18-body / COCO-18 keypoint names.
KeypointName = Literal[
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
]

EventType = Literal[
    "possible_fall",
    "recovered",
    "person_visible",
    "person_not_visible",
    "face_present",
    "face_absent",
    "source_offline",
    "source_online",
    "alert_acknowledged",
]

TruthLabel = Literal[
    "walking",
    "standing",
    "sitting",
    "bending",
    "lying_intentional",
    "fall",
    "occluded",
    "camera_outage",
    "absent",
]

Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class _Model(BaseModel):
    # Serialization schema marks defaulted fields required: the server always sends them.
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Keypoint(_Model):
    """Coordinates are normalized to the frame: (0, 0) top-left, (1, 1) bottom-right."""

    name: KeypointName
    x: Unit
    y: Unit
    confidence: Unit


class PostureFeatures(_Model):
    """Per-frame posture features. Every field is optional: missing = not measurable this frame."""

    torso_angle_deg: Annotated[float, Field(ge=0.0, le=180.0)] | None = Field(
        default=None, description="Neck-to-hip angle from vertical. 0 = upright, 90 = horizontal."
    )
    head_y: Unit | None = Field(default=None, description="Normalized head height (0 = top of frame).")
    hip_y: Unit | None = Field(default=None, description="Normalized mid-hip height (0 = top of frame).")
    bbox_aspect: Annotated[float, Field(ge=0.0)] | None = Field(
        default=None, description="Person bounding box width / height."
    )
    vertical_velocity: float | None = Field(
        default=None, description="Change in hip_y per second; positive = moving down."
    )


class Observation(_Model):
    schema_version: Literal["1"] = SCHEMA_VERSION
    obs_id: str
    timestamp: datetime
    source: str = Field(description="Device or generator id, e.g. 'jetson-01' or 'sim'.")
    source_type: SourceType
    room_id: str
    scenario_id: str | None = None
    person_id: str | None = Field(
        default=None,
        description="Anonymous per-session track id when a source reports several people "
        "(e.g. 'p1'). Not an identity. None = single-person source.",
    )
    person_detected: bool
    keypoints: list[Keypoint] | None = None
    features: PostureFeatures | None = None
    face_present: bool | None = Field(
        default=None, description="None = face detection did not run on this frame."
    )
    face_confidence: Unit | None = None


class Event(_Model):
    schema_version: Literal["1"] = SCHEMA_VERSION
    event_id: str
    timestamp: datetime
    source: str
    source_type: SourceType
    room_id: str
    scenario_id: str | None = None
    person_id: str | None = Field(
        default=None,
        description="Anonymous per-session track id of the person the event concerns "
        "(e.g. 'p1'). Not an identity. None = single-person source.",
    )
    event_type: EventType
    confidence: Unit | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroundTruth(_Model):
    """One labelled segment of a simulated scenario. Times are seconds from scenario start."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    scenario_id: str
    seed: int
    segment_id: str
    person_id: str | None = Field(
        default=None, description="Which simulated resident this segment describes. None = single-resident scenario."
    )
    t_start: Annotated[float, Field(ge=0.0)]
    t_end: Annotated[float, Field(ge=0.0)]
    truth_label: TruthLabel
    expects_alert: bool


class Heartbeat(_Model):
    """Sent by every source about once per second; also carries edge performance metrics."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    timestamp: datetime
    source: str
    source_type: SourceType
    room_id: str
    fps: Annotated[float, Field(ge=0.0)] | None = None
    inference_ms: Annotated[float, Field(ge=0.0)] | None = None
    mem_used_mb: Annotated[float, Field(ge=0.0)] | None = None


class ObservationBatch(_Model):
    """Body of POST /ingest/observations."""

    observations: list[Observation]


class SourceStatus(_Model):
    source: str
    source_type: SourceType
    room_id: str
    online: bool
    last_seen: datetime | None = None
    fps: float | None = None
    inference_ms: float | None = None
    mem_used_mb: float | None = None


# --- WebSocket messages, server -> dashboard (WS /ws/room/{room_id}) ---


class ObservationMessage(_Model):
    type: Literal["observation"] = "observation"
    data: Observation


class EventMessage(_Model):
    type: Literal["event"] = "event"
    data: Event


class SourceStatusMessage(_Model):
    type: Literal["source_status"] = "source_status"
    data: SourceStatus


ServerMessage = Annotated[
    Union[ObservationMessage, EventMessage, SourceStatusMessage],
    Field(discriminator="type"),
]

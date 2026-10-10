"""Demo parameters, to be calibrated against staged traces; not clinical thresholds."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Zone(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str
    kind: str = "bed"
    # Camera-frame coordinates, NOT simulator room meters.
    polygon: list[tuple[float, float]] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_points(self):
        if self.kind not in {"bed", "chair"}:
            raise ValueError("Only bed/chair zones can downgrade alerts")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.polygon):
            raise ValueError("Zone coordinates must be normalized to the observation frame")
        return self


class MonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    room_id: str = "room-01"
    additional_room_ids: list[str] = Field(default_factory=list)
    gesture_hold_seconds: float = Field(default=2, gt=0)
    gesture_release_seconds: float = Field(default=1, gt=0)
    gesture_cooldown_seconds: float = Field(default=10, ge=0)
    gesture_keypoint_confidence_min: float = Field(default=.5, gt=0, le=1)
    gesture_raise_margin: float = Field(default=.08, gt=0, le=1)
    floor_hold_seconds: float = Field(default=3, gt=0)
    posture_mode: Literal["model", "pose_fallback"] = "model"
    fall_model_id: str = "melihuzunoglu/human-fall-detection"
    model_confidence_min: float = Field(default=.65, gt=0, le=1)
    model_standing_hold_seconds: float = Field(default=1, gt=0)
    max_observation_gap_seconds: float = Field(default=1, gt=0)
    source_timeout_seconds: float = Field(default=3, gt=0)
    edge_observation_max_age_seconds: float = Field(default=3, gt=0)
    edge_future_tolerance_seconds: float = Field(default=5, ge=0)
    health_poll_seconds: float = Field(default=.25, gt=0)
    upright_angle_max_deg: float = Field(default=30, ge=0, le=180)
    floor_angle_min_deg: float = Field(default=60, ge=0, le=180)
    floor_angle_max_deg: float = Field(default=120, ge=0, le=180)
    floor_head_y_min: float = Field(default=.65, ge=0, le=1)
    floor_hip_y_min: float = Field(default=.65, ge=0, le=1)
    floor_bbox_aspect_min: float = Field(default=1, gt=0)
    descending_velocity_min: float = Field(default=.15, gt=0)
    zone_keypoint_confidence_min: float = Field(default=.5, gt=0, le=1)
    event_capacity: int = Field(default=10000, gt=0)
    dedupe_capacity: int = Field(default=20000, gt=0)
    websocket_queue_capacity: int = Field(default=512, gt=0)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    zones: list[Zone] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_angles(self):
        if not self.upright_angle_max_deg < self.floor_angle_min_deg <= self.floor_angle_max_deg:
            raise ValueError("Upright and floor angle ranges must not overlap")
        return self

    @property
    def room_ids(self) -> list[str]:
        return list(dict.fromkeys([self.room_id, *self.additional_room_ids]))

    @classmethod
    def load(cls, path: str | Path | None = None):
        return cls.model_validate_json(Path(path or Path(__file__).with_name("config.json")).read_text())

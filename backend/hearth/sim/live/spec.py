"""EngineSpec: everything that configures a live run. Same spec and seed, same run."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IncidentKind = Literal["fall", "inactivity", "pacing", "exit_loiter"]
FallMotion = Literal["trip", "faint", "collapse", "sit_miss", "stand_up"]
CameraPreset = Literal["corner_left", "corner_right", "front", "ceiling", "jetson"]
Lighting = Literal["day", "evening", "dim"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class RoomSpec(_Model):
    width: Annotated[float, Field(ge=4.0, le=10.0)] | None = Field(
        default=None, description="Metres, wall to wall across the camera's view. None = drawn from the seed."
    )
    depth: Annotated[float, Field(ge=4.0, le=9.0)] | None = Field(default=None, description="Metres, front to back.")
    height: Annotated[float, Field(ge=2.3, le=3.5)] = 2.7
    furniture_density: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.6, description="0 = a bare room with a chair or two, 1 = as much as fits."
    )
    doors: Annotated[int, Field(ge=1, le=3)] = 2
    window: bool = True


class CameraSpec(_Model):
    preset: CameraPreset = "corner_left"
    height: Annotated[float, Field(ge=1.0, le=3.4)] = 2.4
    fov: Annotated[float, Field(ge=30.0, le=110.0)] = Field(default=62.0, description="Vertical field of view, degrees.")
    aspect: Annotated[float, Field(gt=0.5, le=3.0)] = 4 / 3
    fps: Annotated[float, Field(gt=0.0, le=30.0)] = 10.0
    tilt: Annotated[float, Field(ge=-15.0, le=60.0)] = Field(
        default=12.0, description="Preset 'jetson': degrees the camera looks down (edge/DEMO.md: 10-15)."
    )
    reach: Annotated[float, Field(gt=0.5, le=12.0)] = Field(
        default=3.5, description="Preset 'jetson': metres into the room where the view centre is aimed."
    )

    @model_validator(mode="after")
    def _jetson_lens(self) -> CameraSpec:
        # The Jetson's Logitech C270 sees about 43 degrees vertically (55 diagonal at 4:3), not 62.
        if self.preset == "jetson" and "fov" not in self.model_fields_set:
            object.__setattr__(self, "fov", 43.0)
        return self


class IncidentSpec(_Model):
    """One incident, at a time (scheduled) or now (manual)."""

    kind: IncidentKind
    t: Annotated[float, Field(ge=0.0)] = Field(default=0.0, description="Scenario seconds (scheduled incidents).")
    actor: str | None = Field(default=None, description="person_id; None = someone suitable, chosen by the seed.")
    motion: FallMotion | None = Field(default=None, description="Falls: how. None = what suits the posture.")
    duration: Annotated[float, Field(gt=0.0)] | None = Field(
        default=None, description="Inactivity: how long. Falls: time on the floor before getting up."
    )
    recover: bool = Field(default=True, description="Falls: get up after `duration` (else stay down until reset).")
    repeats: Annotated[int, Field(ge=1, le=20)] = Field(default=4, description="Pacing legs or exit visits.")
    in_view: bool = Field(
        default=True,
        description="Falls: stage it where the camera has a clear, side-on view (as hand-written scenarios do). "
        "False = wherever there's room, seen or not: the unbiased setting for validation.",
    )


class EngineSpec(_Model):
    room: RoomSpec = RoomSpec()
    actors: Annotated[int, Field(ge=0, le=20)] = 4
    camera: CameraSpec = CameraSpec()
    lighting: Lighting = "day"
    activity_frequency: Annotated[float, Field(ge=0.25, le=4.0)] = Field(
        default=1.0, description="Scales how often people change activity (2 = dwell half as long)."
    )
    social_frequency: Annotated[float, Field(ge=0.0, le=4.0)] = Field(
        default=1.0, description="Scales how often people start conversations (0 = never)."
    )
    incidents: list[IncidentSpec] = Field(default_factory=list, description="Scheduled incidents.")
    incident_rates: dict[IncidentKind, Annotated[float, Field(ge=0.0)]] = Field(
        default_factory=dict, description="Procedural incidents per actor-hour, by kind. Empty = none (the default)."
    )
    noise: bool = Field(default=True, description="Perception noise on the camera's keypoints.")
    room_id: str = "room-01"

"""Scenario files: what the simulated resident does, and the ground truth derived from it.

A scenario is an ordered list of segments. Each segment becomes exactly one GroundTruth
record, so the script and the truth can never drift apart. All scene coordinates are
normalized to the simulated camera frame: (0, 0) top-left, (1, 1) bottom-right.

The file holds nominal values. Each seed realizes its own variant (Scenario.realize):
segment timing, transition speed and positions vary within the scenario's declared
bounds, so evaluating several seeds tests several different performances of the same
script, not just different sensor noise on one.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Annotated, Literal

import yaml
from hearth_schema.models import GroundTruth, TruthLabel
from pydantic import BaseModel, ConfigDict, Field, model_validator

Unit = Annotated[float, Field(ge=0.0, le=1.0)]

Facing = Literal["camera", "left", "right"]
PoseName = Literal["standing", "sitting", "bending", "lying", "fallen"]

# Labels whose segments alert by default. A segment can override this explicitly.
ALERT_LABELS: frozenset[str] = frozenset({"fall"})

DEFAULT_SCENARIOS_DIR = Path(
    os.environ.get("HEARTH_SCENARIOS_DIR", Path(__file__).resolve().parents[3] / "scenarios")
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SceneObject(_Model):
    """A piece of furniture drawn in the room view. Rectangle in frame coordinates."""

    kind: Literal["bed", "chair", "mat", "occluder"]
    x: Unit
    y: Unit
    w: Unit
    h: Unit


class Scene(_Model):
    floor_y: Unit = Field(default=0.92, description="Frame y of the floor line under the person.")
    person_height: Unit = Field(default=0.62, description="Standing height as a fraction of frame height.")
    frame_aspect: float = Field(default=4 / 3, gt=0.0, description="Simulated camera width / height.")
    objects: list[SceneObject] = Field(default_factory=list)


class Variation(_Model):
    """How much each seed may deviate from the nominal script (all +/-)."""

    duration: Unit = Field(default=0.10, description="Fraction of each segment's duration.")
    transition: Unit = Field(default=0.20, description="Fraction of each transition's length.")
    position: Unit = Field(default=0.03, description="Frame x for on-screen positions.")

    @classmethod
    def none(cls) -> Variation:
        return cls(duration=0.0, transition=0.0, position=0.0)


class Segment(_Model):
    label: TruthLabel
    duration: float = Field(gt=0.0, description="Seconds.")
    transition: float = Field(
        default=0.8, ge=0.0, description="Seconds spent moving from the previous pose into this one."
    )
    pose: PoseName | None = Field(
        default=None, description="Override the pose implied by the label (e.g. occluded while sitting)."
    )
    x: float | None = Field(
        default=None, description="Target horizontal position of the person (frame x; may be off-frame)."
    )
    facing: Facing | None = Field(
        default=None, description="Defaults to the walking direction, else the previous facing."
    )
    surface_y: Unit | None = Field(
        default=None, description="Frame y of the surface the person rests on (bed top). Defaults to the floor."
    )
    occluder_top: Unit = Field(
        default=0.55, description="For 'occluded': keypoints below this frame y are hidden."
    )
    expects_alert: bool | None = Field(default=None, description="Defaults to label == 'fall'.")

    @model_validator(mode="after")
    def _transition_fits(self) -> Segment:
        if self.transition > self.duration:
            raise ValueError(f"transition ({self.transition}s) is longer than duration ({self.duration}s)")
        return self

    @property
    def alerts(self) -> bool:
        return self.label in ALERT_LABELS if self.expects_alert is None else self.expects_alert


class Scenario(_Model):
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str
    description: str = ""
    seed: int = 0
    fps: float = Field(default=12.0, gt=0.0, le=30.0)
    room_id: str = "room-01"
    start_x: float = 0.5
    start_facing: Facing = "camera"
    scene: Scene = Field(default_factory=Scene)
    variation: Variation = Field(default_factory=Variation)
    segments: list[Segment] = Field(min_length=1)

    def realize(self, seed: int | None = None) -> Scenario:
        """The concrete variant for one seed, with variation already applied (and zeroed)."""
        seed = self.seed if seed is None else seed
        v = self.variation
        if v == Variation.none():
            return self.model_copy(update={"seed": seed})
        # String seeds hash deterministically (unlike hash()), and keep this stream
        # independent of the generator's perception-noise stream.
        rng = random.Random(f"{self.scenario_id}/{seed}")

        def scale(value: float, amount: float) -> float:
            return round(value * (1.0 + rng.uniform(-amount, amount)), 3)

        def shift(x: float | None) -> float | None:
            if x is None or not 0.0 <= x <= 1.0:  # off-screen entry/exit points stay put
                return x
            return round(min(max(x + rng.uniform(-v.position, v.position), 0.0), 1.0), 4)

        segments = []
        for seg in self.segments:
            duration = scale(seg.duration, v.duration)
            transition = min(scale(seg.transition, v.transition), duration)
            segments.append(seg.model_copy(update={"duration": duration, "transition": transition, "x": shift(seg.x)}))
        return self.model_copy(
            update={"seed": seed, "start_x": shift(self.start_x), "segments": segments, "variation": Variation.none()}
        )

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def segment_times(self) -> list[tuple[float, float]]:
        """(t_start, t_end) of every segment, in seconds from scenario start."""
        times, t = [], 0.0
        for seg in self.segments:
            times.append((t, t + seg.duration))
            t += seg.duration
        return times

    def ground_truth(self, seed: int | None = None) -> list[GroundTruth]:
        """Ground truth for one seed's realized variant."""
        realized = self.realize(seed)
        return realized._segment_truth()

    def _segment_truth(self) -> list[GroundTruth]:
        seed = self.seed
        return [
            GroundTruth(
                scenario_id=self.scenario_id,
                seed=seed,
                segment_id=f"seg-{i + 1:02d}",
                t_start=round(t0, 6),
                t_end=round(t1, 6),
                truth_label=seg.label,
                expects_alert=seg.alerts,
            )
            for i, (seg, (t0, t1)) in enumerate(zip(self.segments, self.segment_times()))
        ]


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return Scenario.model_validate(yaml.safe_load(f))


def scenario_paths(directory: str | Path = DEFAULT_SCENARIOS_DIR) -> list[Path]:
    return sorted(Path(directory).glob("*.yaml"))


def load_all(directory: str | Path = DEFAULT_SCENARIOS_DIR) -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}
    for path in scenario_paths(directory):
        sc = load_scenario(path)
        if sc.scenario_id in scenarios:
            raise ValueError(f"duplicate scenario_id {sc.scenario_id!r} in {path}")
        if sc.scenario_id != path.stem:
            raise ValueError(f"{path.name}: scenario_id {sc.scenario_id!r} must match the file name")
        scenarios[sc.scenario_id] = sc
    return scenarios


def find_scenario(name_or_path: str, directory: str | Path = DEFAULT_SCENARIOS_DIR) -> Scenario:
    """Load by scenario_id from the scenarios directory, or by explicit file path."""
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return load_scenario(path)
    return load_scenario(Path(directory) / f"{name_or_path}.yaml")

"""Scenario files: what the simulated residents do, and the ground truth derived from it.

A scenario scripts one person (`segments`) or several at once (`residents`, each with
their own segments). Each segment becomes exactly one GroundTruth record, so the script
and the truth can never drift apart. All scene coordinates are normalized to the
simulated camera frame: (0, 0) top-left, (1, 1) bottom-right. Depth runs from 0 (front,
nearest the camera) to 1 (back wall): people further back stand higher in the frame and
appear smaller.

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
ObjectKind = Literal["bed", "chair", "couch", "mat", "occluder"]
TagValue = str | int | float | bool

# Labels whose segments alert by default. A segment can override this explicitly.
ALERT_LABELS: frozenset[str] = frozenset({"fall"})

DEFAULT_SCENARIOS_DIR = Path(
    os.environ.get("HEARTH_SCENARIOS_DIR", Path(__file__).resolve().parents[3] / "scenarios")
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class SceneObject(_Model):
    """A piece of furniture drawn in the room view. Rectangle in frame coordinates."""

    kind: ObjectKind
    x: Unit
    y: Unit
    w: Unit
    h: Unit


class Scene(_Model):
    floor_y: Unit = Field(default=0.92, description="Frame y of the floor line under a person at the front.")
    person_height: Unit = Field(
        default=0.62, description="Standing height at the front, as a fraction of frame height."
    )
    frame_aspect: float = Field(default=4 / 3, gt=0.0, description="Simulated camera width / height.")
    depth_rise: Unit = Field(
        default=0.14, description="How far the floor line rises in the frame for a person at depth 1 (back wall)."
    )
    depth_shrink: Unit = Field(
        default=0.3, description="How much smaller a person at depth 1 appears than one at depth 0."
    )
    objects: list[SceneObject] = Field(default_factory=list)

    def floor_at(self, depth: float) -> float:
        return self.floor_y - depth * self.depth_rise

    def height_at(self, depth: float, height: float = 1.0) -> float:
        return self.person_height * height * (1.0 - depth * self.depth_shrink)


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
    depth: Unit | None = Field(
        default=None, description="Target distance into the room: 0 = front (nearest the camera), 1 = back wall."
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
    tags: dict[str, TagValue] = Field(
        default_factory=dict, description="Free-form factors (e.g. fall kind) for breaking down evaluation results."
    )

    @model_validator(mode="after")
    def _transition_fits(self) -> Segment:
        if self.transition > self.duration:
            raise ValueError(f"transition ({self.transition}s) is longer than duration ({self.duration}s)")
        return self

    @property
    def alerts(self) -> bool:
        return self.label in ALERT_LABELS if self.expects_alert is None else self.expects_alert


class Resident(_Model):
    """One simulated person and their script. A single-person scenario has one, with no id."""

    resident_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
        description="Anonymous id reported as Observation.person_id (e.g. 'p1'). None = single-person scenario.",
    )
    height: float = Field(default=1.0, gt=0.5, le=1.3, description="Relative to Scene.person_height.")
    start_x: float = 0.5
    start_depth: Unit = 0.0
    start_facing: Facing = "camera"
    segments: list[Segment] = Field(min_length=1)

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

    def vary(self, rng: random.Random, v: Variation) -> Resident:
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
        return self.model_copy(update={"start_x": shift(self.start_x), "segments": segments})


class Scenario(_Model):
    """Give either `segments` (one person) or `residents` (several people at once)."""

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
    tags: dict[str, TagValue] = Field(
        default_factory=dict, description="Free-form factors (e.g. generator settings) for breaking down results."
    )
    segments: list[Segment] | None = Field(default=None, min_length=1)
    residents: list[Resident] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _one_cast(self) -> Scenario:
        if (self.segments is None) == (self.residents is None):
            raise ValueError("give exactly one of 'segments' (one person) or 'residents' (several)")
        if self.residents is not None:
            ids = [r.resident_id for r in self.residents]
            if None in ids or len(set(ids)) != len(ids):
                raise ValueError("every resident needs a unique resident_id")
        return self

    @property
    def multi_person(self) -> bool:
        return self.residents is not None

    def cast(self) -> list[Resident]:
        """The people in this scenario. A single-person scenario has one resident with id None."""
        if self.residents is not None:
            return list(self.residents)
        return [Resident(start_x=self.start_x, start_facing=self.start_facing, segments=self.segments or [])]

    def realize(self, seed: int | None = None) -> Scenario:
        """The concrete variant for one seed, with variation already applied (and zeroed)."""
        seed = self.seed if seed is None else seed
        v = self.variation
        if v == Variation.none():
            return self.model_copy(update={"seed": seed})
        # String seeds hash deterministically (unlike hash()), and keep these streams
        # independent of the generator's perception-noise stream and of each other.
        update: dict = {"seed": seed, "variation": Variation.none()}
        if self.residents is None:
            varied = self.cast()[0].vary(random.Random(f"{self.scenario_id}/{seed}"), v)
            update |= {"start_x": varied.start_x, "segments": varied.segments}
        else:
            update["residents"] = [
                r.vary(random.Random(f"{self.scenario_id}/{seed}/{r.resident_id}"), v) for r in self.residents
            ]
        return self.model_copy(update=update)

    @property
    def duration(self) -> float:
        return max(r.duration for r in self.cast())

    def segment_times(self) -> list[tuple[float, float]]:
        """Single-person scenarios only; use cast()[i].segment_times() otherwise."""
        if self.multi_person:
            raise ValueError("segment_times() is per resident in a multi-person scenario")
        return self.cast()[0].segment_times()

    def ground_truth(self, seed: int | None = None) -> list[GroundTruth]:
        """Ground truth for one seed's realized variant."""
        return self.realize(seed)._segment_truth()

    def _segment_truth(self) -> list[GroundTruth]:
        total = self.duration
        truth = []
        for res in self.cast():
            prefix = f"{res.resident_id}-" if res.resident_id else ""
            times = res.segment_times()
            for i, (seg, (t0, t1)) in enumerate(zip(res.segments, times)):
                if i == len(times) - 1:
                    t1 = total  # a resident whose script ends early holds their last activity
                truth.append(
                    GroundTruth(
                        scenario_id=self.scenario_id,
                        seed=self.seed,
                        segment_id=f"{prefix}seg-{i + 1:02d}",
                        person_id=res.resident_id,
                        t_start=round(t0, 6),
                        t_end=round(t1, 6),
                        truth_label=seg.label,
                        expects_alert=seg.alerts,
                    )
                )
        return truth


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

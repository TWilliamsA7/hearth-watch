"""Headless live runs: the live engine as fast as it will go, written like any other run.

A LiveRun has the same shape the scripted runner's SimRun offers to write_run() and to
the evaluator (name, items, ground_truth, start, info(), world()), so a live run is
written to the same four files, replays in the room-view dev page, and scores the same
way. Live specs can be kept as YAML under scenarios/live/:

    title: Day room, six residents, one fall
    seed: 3
    duration: 300
    spec:
      actors: 6
      incidents: [{kind: fall, t: 120}]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from hearth_schema.models import GroundTruth, Heartbeat, Observation
from pydantic import BaseModel, ConfigDict, Field

from hearth.sim.generator import SIM_EPOCH
from hearth.sim.runner import RunInfo
from hearth.sim.scenario import DEFAULT_SCENARIOS_DIR
from hearth.sim.world import WorldTrack

from hearth.sim.live.engine import World, run_world
from hearth.sim.live.incidents import IncidentRecord
from hearth.sim.live.spec import EngineSpec

LIVE_SUBDIR = "live"


class LiveScenario(BaseModel):
    """A live run, saved: everything needed to reproduce it."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = ""
    title: str = "Live engine run"
    seed: int = 0
    duration: float = Field(default=300.0, gt=0)
    spec: EngineSpec = EngineSpec()


@dataclass
class _Scenario:
    """The bits of a Scenario the evaluator's drive() reads."""

    scenario_id: str
    duration: float


@dataclass(frozen=True)
class LiveRun:
    scenario: _Scenario
    title: str
    seed: int
    start: datetime
    items: list[Observation | Heartbeat]
    ground_truth: list[GroundTruth]
    incidents: list[IncidentRecord]
    world_track: WorldTrack | None
    _world: World

    @property
    def name(self) -> str:
        return f"{self.scenario.scenario_id}-s{self.seed}"

    @property
    def duration(self) -> float:
        return self.scenario.duration

    @property
    def observations(self) -> list[Observation]:
        return [i for i in self.items if isinstance(i, Observation)]

    def world(self) -> WorldTrack:
        if self.world_track is None:
            raise RuntimeError("run with frames=True to keep the world track")
        return self.world_track

    def info(self) -> RunInfo:
        n_obs = len(self.observations)
        w = self._world
        return RunInfo(run_id=self.name, scenario_id=self.scenario.scenario_id, title=self.title, seed=self.seed,
                       start=self.start, duration=self.scenario.duration, fps=w.spec.camera.fps, room_id=w.spec.room_id,
                       scene=w.scene(), residents=[a.person_id for a in w.actors], observations=n_obs,
                       heartbeats=len(self.items) - n_obs)  # fmt: skip


def run_live(sc: LiveScenario, *, start: datetime = SIM_EPOCH, frames: bool = True) -> LiveRun:
    scenario_id = sc.scenario_id or f"live-{sc.seed}"
    w, items, track = run_world(sc.spec, sc.seed, sc.duration, start=start, scenario_id=scenario_id, frames=frames)
    return LiveRun(_Scenario(scenario_id, round(w.t, 3)), sc.title, sc.seed, start, items, w.ground_truth(),
                   list(w.incidents.records), track, w)  # fmt: skip


def load_live_scenarios(directory: Path = DEFAULT_SCENARIOS_DIR / LIVE_SUBDIR) -> dict[str, LiveScenario]:
    out: dict[str, LiveScenario] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        sc = LiveScenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        if not sc.scenario_id:
            sc = sc.model_copy(update={"scenario_id": path.stem})
        out[sc.scenario_id] = sc
    return out

"""Headless runs: generate a whole scenario as fast as possible and write it to disk.

A run writes four files, kept apart on purpose:
  <scenario>-s<seed>.stream.jsonl  observations + heartbeats, what the monitor may see
  <scenario>-s<seed>.truth.jsonl   GroundTruth segments, for evaluation only
  <scenario>-s<seed>.run.json      run metadata (virtual start time, fps, duration)
  <scenario>-s<seed>.world.json    every body in 3D, for the dashboard's 3D view only
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hearth_schema.models import GroundTruth, Heartbeat, Observation
from pydantic import BaseModel, ConfigDict

from hearth.sim.generator import SIM_EPOCH, NoiseConfig, ObservationGenerator
from hearth.sim.scenario import Scenario, Scene
from hearth.sim.world import WorldTrack

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[3] / "recordings" / "sim"


class RunInfo(BaseModel):
    """Run metadata (<run>.run.json). The dashboard uses it to draw the room and place
    events on the scenario clock (t = timestamp - start)."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    run_id: str
    scenario_id: str
    title: str
    seed: int
    start: datetime
    duration: float
    fps: float
    room_id: str
    scene: Scene
    residents: list[str]  # person_ids in a multi-person run; empty for a single-person run
    observations: int
    heartbeats: int


@dataclass(frozen=True)
class SimRun:
    scenario: Scenario  # realized for this seed: its timings match ground_truth
    seed: int
    start: datetime
    items: list[Observation | Heartbeat]
    ground_truth: list[GroundTruth]

    @property
    def name(self) -> str:
        return f"{self.scenario.scenario_id}-s{self.seed}"

    @property
    def observations(self) -> list[Observation]:
        return [i for i in self.items if isinstance(i, Observation)]

    def world(self) -> WorldTrack:
        """The run's bodies in 3D. Computed on demand: batch evaluation never needs it."""
        return ObservationGenerator(self.scenario, self.seed, start=self.start).world_track()

    def info(self) -> RunInfo:
        sc = self.scenario
        n_obs = len(self.observations)
        return RunInfo(run_id=self.name, scenario_id=sc.scenario_id, title=sc.title, seed=self.seed,
                       start=self.start, duration=sc.duration, fps=sc.fps, room_id=sc.room_id, scene=sc.scene,
                       residents=[r.resident_id for r in sc.cast() if r.resident_id],
                       observations=n_obs, heartbeats=len(self.items) - n_obs)  # fmt: skip


def run_headless(
    scenario: Scenario, seed: int | None = None, *, start: datetime = SIM_EPOCH, noise: NoiseConfig | None = None
) -> SimRun:
    gen = ObservationGenerator(scenario, seed, start=start, noise=noise)
    return SimRun(gen.scenario, gen.seed, start, list(gen.stream()), scenario.ground_truth(gen.seed))


def stream_line(item: Observation | Heartbeat) -> str:
    kind = "observation" if isinstance(item, Observation) else "heartbeat"
    return json.dumps({"type": kind, "data": item.model_dump(mode="json")})


def write_run(run: SimRun, out_dir: str | Path = DEFAULT_OUT_DIR) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stream": out_dir / f"{run.name}.stream.jsonl",
        "truth": out_dir / f"{run.name}.truth.jsonl",
        "run": out_dir / f"{run.name}.run.json",
        "world": out_dir / f"{run.name}.world.json",
    }
    paths["stream"].write_text("".join(stream_line(i) + "\n" for i in run.items), encoding="utf-8")
    paths["truth"].write_text("".join(gt.model_dump_json() + "\n" for gt in run.ground_truth), encoding="utf-8")
    paths["run"].write_text(run.info().model_dump_json(indent=2) + "\n", encoding="utf-8")
    paths["world"].write_text(run.world().model_dump_json() + "\n", encoding="utf-8")
    return paths

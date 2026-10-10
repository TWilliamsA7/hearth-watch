"""Headless runs: generate a whole scenario as fast as possible and write it to disk.

A run writes three files, kept apart on purpose:
  <scenario>-s<seed>.stream.jsonl  observations + heartbeats, what the monitor may see
  <scenario>-s<seed>.truth.jsonl   GroundTruth segments, for evaluation only
  <scenario>-s<seed>.run.json      run metadata (virtual start time, fps, duration)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hearth_schema.models import GroundTruth, Heartbeat, Observation

from hearth.sim.generator import SIM_EPOCH, NoiseConfig, ObservationGenerator
from hearth.sim.scenario import Scenario

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[3] / "recordings" / "sim"


@dataclass(frozen=True)
class SimRun:
    scenario: Scenario
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


def run_headless(
    scenario: Scenario, seed: int | None = None, *, start: datetime = SIM_EPOCH, noise: NoiseConfig | None = None
) -> SimRun:
    gen = ObservationGenerator(scenario, seed, start=start, noise=noise)
    return SimRun(scenario, gen.seed, start, list(gen.stream()), scenario.ground_truth(gen.seed))


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
    }
    paths["stream"].write_text("".join(stream_line(i) + "\n" for i in run.items), encoding="utf-8")
    paths["truth"].write_text("".join(gt.model_dump_json() + "\n" for gt in run.ground_truth), encoding="utf-8")
    meta = {
        "scenario_id": run.scenario.scenario_id,
        "title": run.scenario.title,
        "seed": run.seed,
        "start": run.start.isoformat().replace("+00:00", "Z"),
        "duration": run.scenario.duration,
        "fps": run.scenario.fps,
        "room_id": run.scenario.room_id,
        "scene": run.scenario.scene.model_dump(mode="json"),
        "observations": len(run.observations),
        "heartbeats": len(run.items) - len(run.observations),
    }
    paths["run"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return paths

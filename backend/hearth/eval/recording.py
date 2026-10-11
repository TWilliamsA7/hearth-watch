"""Recorded observation streams, real or simulated, and their hand-labelled ground truth.

One loader reads every recording format in the project:

  * edge recordings (edge/hearth_edge/recorder.py, run_edge.py --record): one bare
    Observation per line, source_type "edge", no heartbeats;
  * simulator runs (python -m hearth.sim run -> <run>.stream.jsonl) and fall-model
    recordings (hearth.perception.cli --record): {"type": "observation" | "heartbeat",
    "data": {...}} lines;
  * a mix of bare Observations and Heartbeats.

Other message types (events, alerts, source status) are skipped, so a captured
WebSocket log loads too.

Ground truth for a clip lives next to it: <name>.truth.jsonl, one GroundTruth per line,
t_start / t_end in seconds from the clip's first record, scenario_id = <name>, seed 0.
<name> is the file name without ".jsonl" and a trailing ".stream", so simulator runs
find their own <run>.truth.jsonl unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hearth_schema.models import GroundTruth, Heartbeat, Observation

Record = Observation | Heartbeat

# Clip-name prefixes used by edge/tools/record_clips.sh, and what they stage.
PREFIX_LABELS: dict[str, tuple[str, bool]] = {
    "lie_down": ("fall", True),  # staged falls / lying on the floor: an alert is expected
    "fall": ("fall", True),
    "stand": ("standing", False),
    "walk": ("walking", False),
    "sit": ("sitting", False),
    "bend": ("bending", False),
    "lying_intentional": ("lying_intentional", False),
}


def clip_name(path: Path) -> str:
    name = path.name.removesuffix(".jsonl")
    return name.removesuffix(".stream")


def truth_path(path: Path) -> Path:
    return path.with_name(f"{clip_name(path)}.truth.jsonl")


def parse_record(line: dict) -> Record | None:
    if "type" in line and "data" in line:
        kind, data = line["type"], line["data"]
        if kind == "observation":
            return Observation.model_validate(data)
        if kind == "heartbeat":
            return Heartbeat.model_validate(data)
        return None
    if "obs_id" in line:
        return Observation.model_validate(line)
    if "timestamp" in line and "source" in line and "event_type" not in line and "alert_id" not in line:
        return Heartbeat.model_validate(line)
    return None


@dataclass(frozen=True)
class Recording:
    path: Path
    items: list[Record]

    @property
    def name(self) -> str:
        return clip_name(self.path)

    @property
    def start(self) -> datetime:
        return self.items[0].timestamp

    @property
    def duration(self) -> float:
        return (self.items[-1].timestamp - self.start).total_seconds()

    @property
    def observations(self) -> list[Observation]:
        return [i for i in self.items if isinstance(i, Observation)]

    @property
    def source(self) -> str:
        """The camera source the clip came from (the first observation's)."""
        obs = self.observations
        return (obs[0] if obs else self.items[0]).source

    @property
    def people(self) -> list[str]:
        return sorted({o.person_id for o in self.observations if o.person_id})

    def truth(self) -> list[GroundTruth] | None:
        """Hand-labelled truth next to the clip, or None if it hasn't been labelled yet."""
        return load_truth(truth_path(self.path))


def load_recording(path: str | Path) -> Recording:
    path = Path(path)
    items: list[Record] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = parse_record(json.loads(raw))
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: {exc}") from exc
        if record is not None:
            items.append(record)
    if not any(isinstance(i, Observation) for i in items):
        raise ValueError(f"{path}: no observations")
    items.sort(key=lambda i: i.timestamp)  # stable: a frame's people keep their order
    return Recording(path, items)


def find_recordings(paths: list[str | Path]) -> list[Path]:
    """Files as given, and every *.jsonl in given directories, minus truth/report files."""
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(f for f in p.rglob("*.jsonl")
                            if not f.name.endswith((".truth.jsonl", ".events.jsonl")))  # fmt: skip
        else:
            found.append(p)
    return found


def load_truth(path: Path) -> list[GroundTruth] | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [GroundTruth.model_validate_json(line) for line in lines] or None


def guess_label(name: str) -> tuple[str, bool]:
    """(truth_label, expects_alert) from a clip name such as lie_down-20261010-063944."""
    for prefix in sorted(PREFIX_LABELS, key=len, reverse=True):
        if name.startswith(prefix):
            return PREFIX_LABELS[prefix]
    return ("standing", False)


def label_template(rec: Recording) -> list[GroundTruth]:
    """One segment per person covering the whole clip, labelled from the clip name.

    A starting point for hand-timing: split it into segments (e.g. standing 0-4.2, fall
    4.2-9.0) using python -m hearth.eval timeline <clip>."""
    label, alerts = guess_label(rec.name)
    end = round(rec.duration, 2) or 0.01
    return [GroundTruth(scenario_id=rec.name, seed=0, segment_id=f"{person or 'p'}-1", person_id=person,
                        t_start=0.0, t_end=end, truth_label=label, expects_alert=alerts)
            for person in (rec.people or [None])]  # fmt: skip


def write_truth(path: Path, truth: list[GroundTruth]) -> None:
    path.write_text("".join(gt.model_dump_json() + "\n" for gt in truth), encoding="utf-8")

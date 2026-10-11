"""Score recorded clips (Jetson, fall-model video, captured sessions) against hand-labelled truth.

The same monitor, driver and scorer as simulated batch runs: a clip is replayed through
the real MonitorService on its own record time, and evaluate() compares the events with
the clip's <name>.truth.jsonl. Clips without truth are still replayed and their events
listed, so you can see what the monitor did before labelling them.

  python -m hearth.eval real recordings/real/jetson
  python -m hearth.eval timeline recordings/real/jetson/lie_down-20261010-063944.jsonl
  python -m hearth.eval label-template recordings/real/jetson
"""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hearth_schema.models import Event, Observation

from hearth.eval.batch import DEFAULT_OUT_DIR, MonitorFactory, PerPersonMonitor, drive, summary_markdown
from hearth.eval.real_monitor import OVERRIDES_ENV, ServiceMonitor, create_monitor, load_config
from hearth.eval.recording import Recording
from hearth.eval.report import EvaluationReport, EvaluationSummary
from hearth.eval.scoring import EvalConfig, evaluate, summarize
from hearth.monitor.config import MonitorConfig
from hearth.monitor.detector import track_key

DEFAULT_REAL_DIR = DEFAULT_OUT_DIR / "real"
REAL_NOTE = ("Recorded clips scored against hand-labelled truth. A handful of staged clips from one "
             "camera position: evidence about this setup, not real-world or clinical accuracy.")


@dataclass
class ClipResult:
    recording: Recording
    events: list[Event]
    report: EvaluationReport | None  # None: not labelled yet

    @property
    def alerts(self) -> list[Event]:
        return [e for e in self.events if e.event_type == "possible_fall"]


def evaluate_recording(rec: Recording, factory: MonitorFactory = create_monitor,
                       config: EvalConfig = EvalConfig()) -> ClipResult:
    events = drive(rec, PerPersonMonitor(factory), tail=max(config.alert_tolerance, config.check_tolerance) + 1.0)
    truth = rec.truth()
    if truth is None:
        return ClipResult(rec, events, None)
    report = evaluate(events, truth, rec.start, source=rec.source, config=config)
    return ClipResult(rec, events, report.model_copy(update={"run_id": rec.name}))


def evaluate_recordings(recs: list[Recording], factory: MonitorFactory = create_monitor,
                        config: EvalConfig = EvalConfig()) -> list[ClipResult]:
    return [evaluate_recording(r, factory, config) for r in recs]


def real_markdown(results: list[ClipResult], monitor: str) -> str:
    scored = [r.report for r in results if r.report is not None]
    lines: list[str] = []
    if scored:
        lines.append(summary_markdown(summarize(scored), scored, monitor, title="Recorded clip results",
                                      note=REAL_NOTE, unit="Clip"))
    else:
        lines += ["# Recorded clip results", "", "No clip has a .truth.jsonl yet, so nothing is scored.", ""]
    unlabelled = [r for r in results if r.report is None]
    if unlabelled:
        lines += ["## Not labelled yet", "",
                  "Events the monitor raised; add `<clip>.truth.jsonl` to score them "
                  "(`python -m hearth.eval label-template <clip>`).", "",
                  "| Clip | Duration (s) | Observations | possible_fall at (s) |", "| --- | ---: | ---: | --- |"]
        for r in unlabelled:
            at = ", ".join(f"{(e.timestamp - r.recording.start).total_seconds():.1f}" for e in r.alerts) or "-"
            lines.append(f"| {r.recording.name} | {r.recording.duration:.1f} | {len(r.recording.observations)} | {at} |")
        lines.append("")
    lines += feature_table(results)
    return "\n".join(lines)


def write_results(results: list[ClipResult], out: Path, monitor: str) -> str:
    (out / "clips").mkdir(parents=True, exist_ok=True)
    for r in results:
        (out / "clips" / f"{r.recording.name}.events.jsonl").write_text(
            "".join(e.model_dump_json() + "\n" for e in r.events), encoding="utf-8")
        if r.report is not None:
            (out / "clips" / f"{r.recording.name}.report.json").write_text(r.report.model_dump_json(indent=2),
                                                                            encoding="utf-8")
    scored = [r.report for r in results if r.report is not None]
    if scored:
        (out / "summary.json").write_text(summarize(scored).model_dump_json(indent=2), encoding="utf-8")
    markdown = real_markdown(results, monitor)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    return markdown


# --- features per truth label: the sim-to-real gap ---

FEATURES = ("torso_angle_deg", "head_y", "hip_y", "bbox_aspect", "vertical_velocity")


def feature_rows(results: list[ClipResult]) -> dict[str, dict[str, list[float]]]:
    """truth label -> feature -> values, from every labelled observation."""
    rows: dict[str, dict[str, list[float]]] = {}
    for r in results:
        truth = r.recording.truth()
        if truth is None:
            continue
        for o in r.recording.observations:
            if o.features is None:
                continue
            t = (o.timestamp - r.recording.start).total_seconds()
            seg = next((g for g in truth if g.person_id == o.person_id and g.t_start <= t < g.t_end), None)
            if seg is None:
                continue
            per = rows.setdefault(seg.truth_label, {})
            for name in FEATURES:
                if (v := getattr(o.features, name)) is not None:
                    per.setdefault(name, []).append(v)
    return rows


def feature_table(results: list[ClipResult]) -> list[str]:
    rows = feature_rows(results)
    if not rows:
        return []
    lines = ["## Features by true activity", "",
             "Median (10th-90th percentile). Compare with simulated runs of the same activity "
             "(`python -m hearth.eval real recordings/sim`) to see where the simulator differs.", "",
             "| Activity | Frames | " + " | ".join(FEATURES) + " |", "| --- | ---: |" + " ---: |" * len(FEATURES)]
    for label in sorted(rows):
        per = rows[label]
        frames = max((len(v) for v in per.values()), default=0)
        cells = []
        for name in FEATURES:
            vs = sorted(per.get(name, []))
            if not vs:
                cells.append("-")
                continue
            q = (lambda f: vs[min(len(vs) - 1, int(f * len(vs)))])
            cells.append(f"{statistics.median(vs):.2f} ({q(0.1):.2f}-{q(0.9):.2f})")
        lines.append(f"| {label} | {frames} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


# --- labelling aid ---


@dataclass
class Bucket:
    t: float
    frames: int = 0
    people: int = 0
    states: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    fails: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def floor_fails(obs: Observation, c: MonitorConfig) -> list[str]:
    """Why a frame is not floor-level under the pose rules (empty = floor-level)."""
    f = obs.features
    if f is None:
        return ["no features"]
    fails = []
    if f.torso_angle_deg is None:
        fails.append("no torso")
    elif not c.floor_angle_min_deg <= f.torso_angle_deg <= c.floor_angle_max_deg:
        fails.append("angle")
    if f.head_y is None:
        fails.append("no head")
    elif f.head_y < c.floor_head_y_min:
        fails.append("head high")
    if f.hip_y is None:
        fails.append("no hips")
    elif f.hip_y < c.floor_hip_y_min:
        fails.append("hips high")
    if f.bbox_aspect is not None and f.bbox_aspect < c.floor_bbox_aspect_min:
        fails.append("narrow box")
    return fails


def timeline(rec: Recording, config: MonitorConfig | None = None, bucket_s: float = 0.5) -> list[Bucket]:
    """The clip in bucket_s steps: the detector's posture state, features, failing floor rules
    and events. Use it to find where each truth segment starts and ends."""
    config = config or load_config()
    monitor = ServiceMonitor(config)
    buckets: dict[int, Bucket] = {}

    def at(t: float) -> Bucket:
        i = int(t // bucket_s)
        return buckets.setdefault(i, Bucket(round(i * bucket_s, 3)))

    for item in rec.items:
        t = (item.timestamp - rec.start).total_seconds()
        events = monitor.ingest(item)
        b = at(t)
        for e in events:
            b.events.append(e.event_type + (f"[{e.person_id}]" if e.person_id else ""))
        if not isinstance(item, Observation):
            continue
        b.frames += 1
        if not item.person_detected:
            continue
        b.people += 1
        state = monitor.service.detector.states.get(track_key(item))
        if state is not None:
            b.states.append(state.posture)
        if item.features is not None:
            b.features = {k: v for k, v in item.features.model_dump().items() if v is not None}
        b.fails = floor_fails(item, config)
    return [buckets[i] for i in sorted(buckets)]


def timeline_text(rec: Recording, rows: list[Bucket]) -> str:
    head = (f"{rec.name}: {rec.duration:.1f} s, {len(rec.observations)} observations, source {rec.source}\n"
            f"{'t':>6}  {'state':<18} {'angle':>6} {'head':>5} {'hip':>5} {'box':>5} {'vel':>6}  "
            "not floor because / events")
    lines = [head]
    for b in rows:
        state = max(set(b.states), key=b.states.count) if b.states else ("nobody" if b.frames else "-")
        f = b.features

        def num(name: str, width: int, fmt: str = ".2f") -> str:
            return f"{f[name]:{width}{fmt}}" if name in f else " " * (width - 1) + "-"

        notes = ", ".join(b.fails) if b.people and b.fails else ("FLOOR" if b.people else "")
        if b.events:
            notes += ("  << " if notes else "<< ") + ", ".join(b.events)
        lines.append(f"{b.t:6.1f}  {state:<18} {num('torso_angle_deg', 6, '.1f')} {num('head_y', 5)} "
                     f"{num('hip_y', 5)} {num('bbox_aspect', 5)} {num('vertical_velocity', 6)}  {notes}")
    return "\n".join(lines)


# --- threshold sweeps ---


@dataclass
class SweepRow:
    value: object
    summary: EvaluationSummary


def sweep(recs: list[Recording], key: str, values: list, config: EvalConfig = EvalConfig()) -> list[SweepRow]:
    """Score the labelled clips once per value of one monitor setting (on top of --set overrides)."""
    labelled = [r for r in recs if r.truth() is not None]
    if not labelled:
        raise ValueError("a sweep needs labelled clips (<clip>.truth.jsonl)")
    base = json.loads(os.environ.get(OVERRIDES_ENV) or "{}")
    rows = []
    for value in values:
        monitor_config = load_config(overrides={**base, key: value})
        factory: Callable[[], ServiceMonitor] = lambda c=monitor_config: ServiceMonitor(c)  # noqa: E731
        reports = [evaluate_recording(r, factory, config).report for r in labelled]
        rows.append(SweepRow(value, summarize([r for r in reports if r is not None])))
    return rows


def sweep_markdown(key: str, rows: list[SweepRow]) -> str:
    def fmt(x: float | None) -> str:
        return "-" if x is None else f"{x:.2f}"

    lines = [f"## Sweep: {key}", "", "| Value | Detected | Missed | False alerts | Median delay (s) | Max delay (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        c = row.summary.counts
        lines.append(f"| {row.value} | {c.true_positives}/{c.expected_alerts} | {c.missed} | {c.false_positives} "
                     f"| {fmt(row.summary.delay_median)} | {fmt(row.summary.delay_max)} |")
    return "\n".join(lines + [""])

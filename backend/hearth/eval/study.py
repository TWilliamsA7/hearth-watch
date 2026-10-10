"""Studies: many generated scenarios through the monitor, with results broken down by factor.

    python -m hearth.eval study --count 300 --people 1-4 --fall-rate 0.6 --jobs 8

Each run is a generated scenario (hearth.sim.generate) scored as a batch run is
(evaluate_run). What makes a study informative is the breakdown: every fall carries its
tags (kind, outcome, depth, persona, facing) and every run its own (people, room,
responder, outage), so the report shows *where* the monitor misses falls and raises
false alerts, not just how often. Rates come with 95% Wilson intervals, so a cell with
four falls in it doesn't read like one with four hundred.

The worst runs are saved as scenario YAML with the command that replays them, so any
row of the report can be watched in the room-view dev page.

Simulated results only, never accuracy claims (README, Claims and privacy).
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from hearth.eval.batch import DEFAULT_OUT_DIR, evaluate_run, load_monitor_factory
from hearth.eval.report import (
    Counts,
    EvaluationReport,
    FalseAlertBreakdown,
    FallBreakdown,
    Rate,
    StudyReport,
    StudyRun,
)
from hearth.eval.scoring import ALERT_EVENT, EvalConfig
from hearth.sim.behavior import FALL_KINDS
from hearth.sim.generate import GenSpec, generate
from hearth.sim.scenario import Scenario, save_scenario

DEFAULT_STUDY_DIR = DEFAULT_OUT_DIR / "study"
FALL_FACTORS = ("fault", "outcome", "depth_bin", "persona", "facing")
"""Tags on each fall segment."""
RUN_FACTORS = ("people", "room", "responder", "outage", "collision")
"""Tags on each generated scenario."""


def plan_study(
    base: GenSpec, *, count: int, people: tuple[int, int] = (1, 1), rooms: Iterable[str] = ("day-room",),
    seed: int = 0, sweep: bool = False,
) -> list[tuple[GenSpec, int]]:  # fmt: skip
    """The (spec, seed) pairs to run.

    Random: each run draws its number of people and its room. Sweep: every combination of
    people x room x fall kind gets an equal share of the runs, so no cell is left thin.
    """
    rooms = list(rooms)
    if sweep:
        cells = list(itertools.product(range(people[0], people[1] + 1), rooms, FALL_KINDS))
        per_cell = max(1, math.ceil(count / len(cells)))
        return [
            (base.model_copy(update={"people": n, "room": room, "fall_kinds": {kind: 1.0}}), seed + k)
            for n, room, kind in cells
            for k in range(per_cell)
        ]
    out = []
    for i in range(count):
        rng = random.Random(f"study/{seed}/{i}")
        out.append((base.model_copy(update={"people": rng.randint(*people), "room": rng.choice(rooms)}), seed + i))
    return out


@dataclass(frozen=True)
class Outcome:
    """One scored run, with what's needed to break it down."""

    spec: GenSpec
    scenario: Scenario
    report: EvaluationReport

    def segment_tags(self) -> dict[str, dict]:
        """segment_id -> tags, matching GroundTruth ids (see Scenario._segment_truth)."""
        out = {}
        for r in self.scenario.cast():
            prefix = f"{r.resident_id}-" if r.resident_id else ""
            for i, seg in enumerate(r.segments):
                out[f"{prefix}seg-{i + 1:02d}"] = seg.tags
        return out


def _init_worker(path: list[str]) -> None:
    sys.path[:] = path  # so the monitor module resolves in workers as it does here (spawned on Windows)


def run_one(spec: GenSpec, seed: int, monitor: str, config: EvalConfig = EvalConfig()) -> Outcome:
    scenario = generate(spec, seed)
    report, _ = evaluate_run(scenario, seed, load_monitor_factory(monitor), config)
    return Outcome(spec, scenario, report)


def run_study(
    plan: list[tuple[GenSpec, int]], monitor: str, config: EvalConfig = EvalConfig(), *, jobs: int = 1
) -> list[Outcome]:
    load_monitor_factory(monitor)  # fail here, clearly, rather than once per worker
    if jobs <= 1:
        return [run_one(spec, seed, monitor, config) for spec, seed in plan]
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(list(sys.path),)) as pool:
        futures = [pool.submit(run_one, spec, seed, monitor, config) for spec, seed in plan]
        return [f.result() for f in futures]


# --- breakdown ---


def wilson(hits: int, n: int, z: float = 1.96) -> Rate:
    if n == 0:
        return Rate(n=0, hits=0, rate=None, low=None, high=None)
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return Rate(n=n, hits=hits, rate=round(p, 4), low=round(max(0.0, centre - half), 4),
                high=round(min(1.0, centre + half), 4))  # fmt: skip


def _p90(xs: list[float]) -> float | None:
    if not xs:
        return None
    return round(statistics.quantiles(xs, n=10, method="inclusive")[-1] if len(xs) > 1 else xs[0], 3)


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 3) if xs else None


def _activity(label: str | None, tags: dict) -> str:
    if label is None:
        return "unknown"
    if tags.get("responder"):
        return f"{label} (helping)"
    return f"{label} ({tags['hard_negative']})" if "hard_negative" in tags else label


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_report(outcomes: list[Outcome], monitor: str, *, worst: int = 10, reproduce_dir: Path | None = None) -> StudyReport:
    falls: list[tuple[dict[str, str], bool, float | None]] = []  # (factor values, detected, delay)
    run_fa: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0])  # (factor, value) -> [alerts, hours]
    act_fa: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    runs: list[StudyRun] = []

    for o in outcomes:
        r, sc = o.report, o.scenario
        seg_tags = o.segment_tags()
        run_factors = {f: str(sc.tags.get(f, "-")) for f in RUN_FACTORS}
        hours = r.duration / 3600
        for f, v in run_factors.items():
            run_fa[(f, v)][0] += r.counts.false_positives
            run_fa[(f, v)][1] += hours
        for seg in r.segments:
            tags = seg_tags.get(seg.segment_id, {})
            act_fa[_activity(seg.truth_label, tags)][1] += (seg.t_end - seg.t_start) / 3600
            if seg.expected_event == ALERT_EVENT:
                factors = {f: str(tags.get(f, "-")) for f in FALL_FACTORS} | run_factors
                falls.append((factors, bool(seg.detected), seg.delay))
        for a in r.alerts:
            if a.outcome == "false_positive":
                act_fa[_activity(a.truth_label, seg_tags.get(a.segment_id or "", {}))][0] += 1
        runs.append(StudyRun(run_id=r.run_id, scenario_id=sc.scenario_id, seed=sc.seed, title=sc.title,
                             duration=round(r.duration, 3), tags=sc.tags, expected_alerts=r.counts.expected_alerts,
                             missed=r.counts.missed, false_alerts=r.counts.false_positives,
                             delays=r.detection_delays, reproduce=""))  # fmt: skip

    falls_by = []
    for factor in FALL_FACTORS + RUN_FACTORS:
        groups: dict[str, list[tuple[bool, float | None]]] = defaultdict(list)
        for values, hit, delay in falls:
            groups[values[factor]].append((hit, delay))
        for value in sorted(groups):
            g = groups[value]
            delays = [d for hit, d in g if hit and d is not None]
            falls_by.append(FallBreakdown(factor=factor, value=value, detected=wilson(sum(h for h, _ in g), len(g)),
                                          delay_median=_median(delays), delay_p90=_p90(delays)))  # fmt: skip

    def per_hour(n: float, h: float) -> float | None:
        return round(n / h, 2) if h > 0 else None

    false_alerts_by = [
        FalseAlertBreakdown(factor=f, value=v, false_alerts=int(n), hours=round(h, 4), per_hour=per_hour(n, h))
        for (f, v), (n, h) in sorted(run_fa.items())
    ] + [
        FalseAlertBreakdown(factor="activity", value=a, false_alerts=int(n), hours=round(h, 4), per_hour=per_hour(n, h))
        for a, (n, h) in sorted(act_fa.items())
    ]

    bad = sorted((x for x in runs if x.missed or x.false_alerts), key=lambda x: (-x.missed, -x.false_alerts, x.run_id))
    bad = bad[:worst]
    by_id = {o.report.run_id: o for o in outcomes}
    for i, x in enumerate(bad):
        sc = by_id[x.run_id].scenario
        if reproduce_dir is not None:
            path = save_scenario(sc, Path(reproduce_dir) / f"{sc.scenario_id}.yaml")
            bad[i] = x.model_copy(update={"reproduce": f"python -m hearth.sim run {_shown(path)}"})

    delays = [d for o in outcomes for d in o.report.detection_delays]
    hours = sum(o.report.duration for o in outcomes) / 3600
    counts = Counts(**{k: sum(getattr(o.report.counts, k) for o in outcomes) for k in Counts.model_fields})
    return StudyReport(
        monitor=monitor, runs=len(outcomes), simulated_hours=round(hours, 4), counts=counts,
        detected=wilson(counts.true_positives, counts.expected_alerts),
        delay_median=_median(delays), delay_p90=_p90(delays),
        false_alerts_per_hour=per_hour(counts.false_positives, hours),
        falls_by=falls_by, false_alerts_by=false_alerts_by, worst=bad,
    )  # fmt: skip


def _pct(rate: Rate) -> str:
    if rate.rate is None:
        return "-"
    return f"{rate.rate:.0%} ({rate.low:.0%}-{rate.high:.0%})"


def _num(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def study_markdown(report: StudyReport) -> str:
    c = report.counts
    lines = [
        "# Simulated study results",
        "",
        f"Monitor: `{report.monitor}`. {report.runs} generated runs, {report.simulated_hours * 60:.1f} simulated minutes.",
        "",
        "Simulated scenarios only. These numbers describe behaviour on synthetic observations, "
        "not real-world or clinical accuracy. Ranges are 95% Wilson intervals.",
        "",
        "| Falls | Detected | Missed | Median delay (s) | p90 delay (s) | False alerts | False alerts / hour |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        f"| {c.expected_alerts} | {_pct(report.detected)} | {c.missed} | {_num(report.delay_median)} "
        f"| {_num(report.delay_p90)} | {c.false_positives} | {_num(report.false_alerts_per_hour)} |",
        "",
        "## Falls detected, by factor",
        "",
        "| Factor | Value | Falls | Detected | Median delay (s) | p90 delay (s) |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for b in report.falls_by:
        lines.append(f"| {b.factor} | {b.value} | {b.detected.n} | {_pct(b.detected)} "
                     f"| {_num(b.delay_median)} | {_num(b.delay_p90)} |")  # fmt: skip
    lines += [
        "",
        "## False alerts",
        "",
        "Per run factor (room-hours), then per activity the alerted person was doing (person-hours).",
        "",
        "| Factor | Value | False alerts | Hours | Per hour |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for b in report.false_alerts_by:
        lines.append(f"| {b.factor} | {b.value} | {b.false_alerts} | {b.hours:.3f} | {_num(b.per_hour)} |")
    if report.worst:
        lines += ["", "## Worst runs", "", "| Run | Missed | False alerts | What happened | Replay |",
                  "| --- | ---: | ---: | --- | --- |"]  # fmt: skip
        for w in report.worst:
            lines.append(f"| {w.run_id} | {w.missed} | {w.false_alerts} | {w.title} | `{w.reproduce or '-'}` |")
    return "\n".join(lines) + "\n"


def write_study(report: StudyReport, outcomes: list[Outcome], out: Path = DEFAULT_STUDY_DIR) -> Path:
    out = Path(out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for o in outcomes:
        (out / "runs" / f"{o.report.run_id}.report.json").write_text(o.report.model_dump_json(indent=2), encoding="utf-8")
    (out / "study.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (out / "study.md").write_text(study_markdown(report), encoding="utf-8")
    return out


"""Batch evaluation: run scenarios headless through a fresh monitor and score the result.

The monitor seam
----------------
The real monitor (hearth.monitor, owned by the monitoring core) is plugged in through a
factory, by default "hearth.monitor:create_monitor". Whatever it returns must satisfy
MonitorLike:

  * ingest(item)     - one Observation or Heartbeat, in timestamp order; returns any
                       Events it produced.
  * advance_to(now)  - time has passed with no input; returns Events that are due (the
                       3 s source_offline check needs this when a source goes silent).

Time is always taken from the records, never the wall clock, so a 30 s scenario scores
in milliseconds and gives the same answer every time. Each run gets its own monitor,
so no state leaks between runs.
"""

from __future__ import annotations

import importlib
import statistics
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from typing import Protocol

from hearth_schema.models import Event, Heartbeat, Observation

from hearth.eval.report import EvaluationReport, EvaluationSummary
from hearth.eval.scoring import EvalConfig, evaluate
from hearth.sim.runner import SimRun, run_headless
from hearth.sim.scenario import Scenario

DEFAULT_MONITOR = "hearth.monitor:create_monitor"
TICK_SECONDS = 0.25


class MonitorLike(Protocol):
    def ingest(self, item: Observation | Heartbeat) -> Iterable[Event]: ...

    def advance_to(self, now: datetime) -> Iterable[Event]: ...


MonitorFactory = Callable[[], MonitorLike]


def load_monitor_factory(spec: str = DEFAULT_MONITOR) -> MonitorFactory:
    """Resolve "package.module:callable" to a monitor factory."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"monitor spec must look like 'module:callable', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"cannot load monitor {spec!r} ({exc}). The monitor must expose a factory returning an "
            "object with ingest(item) and advance_to(now); see hearth/eval/batch.py."
        ) from exc


def drive(run: SimRun, monitor: MonitorLike, *, tail: float = 5.0, tick: float = TICK_SECONDS) -> list[Event]:
    """Feed a run to the monitor in timestamp order, advancing its clock through any silence.

    tail: seconds to keep advancing after the scenario ends, so late events are captured.
    """
    events: list[Event] = []
    step = timedelta(seconds=tick)
    clock = run.start
    for item in run.items:
        while clock + step <= item.timestamp:
            clock += step
            events.extend(monitor.advance_to(clock))
        events.extend(monitor.ingest(item))
        clock = max(clock, item.timestamp)
    end = run.start + timedelta(seconds=run.scenario.duration + tail)
    while clock + step <= end:
        clock += step
        events.extend(monitor.advance_to(clock))
    return events


def evaluate_run(
    scenario: Scenario, seed: int | None, factory: MonitorFactory, config: EvalConfig = EvalConfig()
) -> tuple[EvaluationReport, list[Event]]:
    run = run_headless(scenario, seed)
    events = drive(run, factory(), tail=max(config.alert_tolerance, config.check_tolerance) + 1.0)
    return evaluate(events, run.ground_truth, run.start, config=config), events


def seeds_for(scenario: Scenario, count: int) -> list[int]:
    """The scenario's own seed first, then consecutive ones: stable across machines and runs."""
    return [scenario.seed + k for k in range(count)]


def run_batch(
    scenarios: Iterable[Scenario], seeds: int, factory: MonitorFactory, config: EvalConfig = EvalConfig()
) -> list[EvaluationReport]:
    return [
        evaluate_run(sc, seed, factory, config)[0] for sc in scenarios for seed in seeds_for(sc, seeds)
    ]


def summary_markdown(summary: EvaluationSummary, reports: list[EvaluationReport], monitor: str) -> str:
    """Human-readable results table. Wording matches the README's claims rules."""
    rows: dict[str, list[EvaluationReport]] = {}
    for r in reports:
        rows.setdefault(r.scenario_id, []).append(r)

    def fmt(x: float | None) -> str:
        return "-" if x is None else f"{x:.2f}"

    lines = [
        "# Simulated evaluation results",
        "",
        f"Monitor: `{monitor}`. {summary.runs} runs over {summary.scenarios} scenarios.",
        "",
        "Simulated scenarios only. These numbers describe behaviour on synthetic observations, "
        "not real-world or clinical accuracy.",
        "",
        "| Scenario | Runs | Expected alerts | Detected | Missed | False alerts | Median delay (s) | Max delay (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sid, rs in rows.items():
        delays = [d for r in rs for d in r.detection_delays]
        median = statistics.median(delays) if delays else None
        lines.append(
            f"| {sid} | {len(rs)} | {sum(r.counts.expected_alerts for r in rs)} "
            f"| {sum(r.counts.true_positives for r in rs)} | {sum(r.counts.missed for r in rs)} "
            f"| {sum(r.counts.false_positives for r in rs)} | {fmt(median)} | {fmt(max(delays) if delays else None)} |"
        )
    c = summary.counts
    lines += [
        f"| **Total** | {summary.runs} | {c.expected_alerts} | {c.true_positives} | {c.missed} "
        f"| {c.false_positives} | {fmt(summary.delay_median)} | {fmt(summary.delay_max)} |",
        "",
    ]
    if summary.false_positives_by_label:
        labels = ", ".join(f"{label}: {n}" for label, n in summary.false_positives_by_label.items())
        lines += [f"False alerts by true activity: {labels}.", ""]
    return "\n".join(lines)

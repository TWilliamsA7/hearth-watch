"""Scoring: match monitor Events against GroundTruth segments.

Pure functions over contract records. The events can come from anywhere the monitor ran:
a headless batch, or an event log captured from the live API during a simulated run.

Rules:
  * A possible_fall alert matches an alert-expected segment if it is raised between the
    segment's start and its end plus a tolerance. The first match is a true positive and
    its delay is measured from segment start; later matches are duplicates (not false
    positives, but reported).
  * Any other possible_fall alert is a false positive, attributed to the segment it
    happened in, so results read as "1 false alert during lying_intentional".
  * An alert-expected segment with no match is a missed event.
  * Non-alert labels can carry a secondary expectation (camera_outage -> source_offline,
    absent -> person_not_visible), scored the same way but kept out of the alert counts.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from hearth_schema.models import Event, EventType, GroundTruth

from hearth.eval.report import (
    AlertResult,
    Counts,
    EvaluationReport,
    EvaluationSummary,
    SegmentResult,
    TimelineEvent,
)

ALERT_EVENT: EventType = "possible_fall"

SECONDARY_EXPECTATIONS: dict[str, EventType] = {
    "camera_outage": "source_offline",
    "absent": "person_not_visible",
}


@dataclass(frozen=True)
class EvalConfig:
    alert_tolerance: float = 3.0
    """Seconds after an alert-expected segment ends in which an alert still counts."""
    check_tolerance: float = 4.0
    """Same, for secondary expectations. Covers the backend's 3 s offline threshold."""


def _seconds(ts: datetime, start: datetime) -> float:
    return round((ts - start).total_seconds(), 6)


def _segment_at(truth: list[GroundTruth], t: float) -> GroundTruth | None:
    for gt in truth:
        if gt.t_start <= t < gt.t_end:
            return gt
    return truth[-1] if truth and t >= truth[-1].t_end else None


def evaluate(
    events: Iterable[Event],
    truth: list[GroundTruth],
    start: datetime,
    *,
    source: str = "sim",
    config: EvalConfig = EvalConfig(),
) -> EvaluationReport:
    """Score one run. start = the run's virtual start time (scenario t = 0)."""
    if not truth:
        raise ValueError("no ground truth segments")
    scenario_id, seed = truth[0].scenario_id, truth[0].seed
    truth = sorted(truth, key=lambda gt: gt.t_start)

    relevant = sorted(
        (e for e in events if e.source == source and e.scenario_id in (None, scenario_id)),
        key=lambda e: e.timestamp,
    )
    timed = [(_seconds(e.timestamp, start), e) for e in relevant]

    # Primary: possible_fall alerts.
    matched: dict[str, tuple[str, float]] = {}  # segment_id -> (event_id, delay)
    alerts: list[AlertResult] = []
    fp_by_segment: Counter[str] = Counter()
    for t, e in timed:
        if e.event_type != ALERT_EVENT:
            continue
        target = next(
            (gt for gt in truth if gt.expects_alert and gt.t_start <= t <= gt.t_end + config.alert_tolerance),
            None,
        )
        if target is None:
            where = _segment_at(truth, t)
            if where is not None:
                fp_by_segment[where.segment_id] += 1
            alerts.append(AlertResult(event_id=e.event_id, t=t, outcome="false_positive",
                                      segment_id=where and where.segment_id,
                                      truth_label=where and where.truth_label))  # fmt: skip
        elif target.segment_id in matched:
            alerts.append(AlertResult(event_id=e.event_id, t=t, outcome="duplicate",
                                      segment_id=target.segment_id, truth_label=target.truth_label))  # fmt: skip
        else:
            delay = round(t - target.t_start, 3)
            matched[target.segment_id] = (e.event_id, delay)
            alerts.append(AlertResult(event_id=e.event_id, t=t, outcome="true_positive", delay=delay,
                                      segment_id=target.segment_id, truth_label=target.truth_label))  # fmt: skip

    # Per segment, including secondary expectations.
    segments: list[SegmentResult] = []
    for gt in truth:
        if gt.expects_alert:
            expected: EventType | None = ALERT_EVENT
            hit = matched.get(gt.segment_id)
        else:
            expected = SECONDARY_EXPECTATIONS.get(gt.truth_label)
            hit = None
            if expected is not None:
                hit = next(
                    ((e.event_id, round(t - gt.t_start, 3)) for t, e in timed
                     if e.event_type == expected and gt.t_start <= t <= gt.t_end + config.check_tolerance),
                    None,
                )  # fmt: skip
        segments.append(
            SegmentResult(
                segment_id=gt.segment_id,
                truth_label=gt.truth_label,
                t_start=gt.t_start,
                t_end=gt.t_end,
                expected_event=expected,
                detected=None if expected is None else hit is not None,
                delay=hit[1] if hit else None,
                matched_event_id=hit[0] if hit else None,
                false_alerts=fp_by_segment[gt.segment_id],
            )
        )

    expected_alerts = sum(1 for gt in truth if gt.expects_alert)
    outcomes = Counter(a.outcome for a in alerts)
    return EvaluationReport(
        run_id=f"{scenario_id}-s{seed}",
        scenario_id=scenario_id,
        seed=seed,
        source=source,
        start=start,
        duration=truth[-1].t_end,
        alert_tolerance=config.alert_tolerance,
        counts=Counts(
            expected_alerts=expected_alerts,
            true_positives=outcomes["true_positive"],
            false_positives=outcomes["false_positive"],
            missed=expected_alerts - outcomes["true_positive"],
            duplicates=outcomes["duplicate"],
        ),
        detection_delays=[delay for _, delay in matched.values()],
        segments=segments,
        alerts=alerts,
        timeline=[TimelineEvent(event_id=e.event_id, t=t, event_type=e.event_type) for t, e in timed],
    )


def summarize(reports: list[EvaluationReport]) -> EvaluationSummary:
    delays = [d for r in reports for d in r.detection_delays]
    fp_labels: Counter[str] = Counter(
        a.truth_label or "unknown" for r in reports for a in r.alerts if a.outcome == "false_positive"
    )
    return EvaluationSummary(
        runs=len(reports),
        scenarios=len({r.scenario_id for r in reports}),
        counts=Counts(
            expected_alerts=sum(r.counts.expected_alerts for r in reports),
            true_positives=sum(r.counts.true_positives for r in reports),
            false_positives=sum(r.counts.false_positives for r in reports),
            missed=sum(r.counts.missed for r in reports),
            duplicates=sum(r.counts.duplicates for r in reports),
        ),
        delay_median=round(statistics.median(delays), 3) if delays else None,
        delay_max=max(delays) if delays else None,
        false_positives_by_label=dict(sorted(fp_labels.items())),
        missed_runs=[r.run_id for r in reports if r.counts.missed],
        false_positive_runs=[r.run_id for r in reports if r.counts.false_positives],
    )

"""Evaluation report models.

These are the after-action data the dashboard renders, so they follow the shared-contract
conventions (extra fields forbidden, times in seconds from scenario start). They are
candidates to move into shared/ once the dashboard consumes them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from hearth_schema.models import EventType, TruthLabel
from pydantic import BaseModel, ConfigDict, Field

REPORT_VERSION = "1"

AlertOutcome = Literal["true_positive", "duplicate", "false_positive"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class SegmentResult(_Model):
    segment_id: str
    person_id: str | None = Field(default=None, description="Resident the record concerns; None in single-person runs.")
    truth_label: TruthLabel
    t_start: float
    t_end: float
    expected_event: EventType | None = Field(description="What the monitor should have reported, if anything.")
    detected: bool | None = Field(description="None when nothing was expected.")
    delay: float | None = Field(default=None, description="Seconds from segment start to the first matching event.")
    matched_event_id: str | None = None
    false_alerts: int = Field(default=0, description="Unexpected possible_fall alerts raised during this segment.")


class AlertResult(_Model):
    event_id: str
    t: float
    outcome: AlertOutcome
    person_id: str | None = Field(default=None, description="Resident the record concerns; None in single-person runs.")
    segment_id: str | None
    truth_label: TruthLabel | None
    delay: float | None = None


class TimelineEvent(_Model):
    """Every monitor event of the run, placed on the scenario clock for the after-action view."""

    event_id: str
    t: float
    event_type: EventType
    person_id: str | None = Field(default=None, description="Resident the record concerns; None in single-person runs.")


class Counts(_Model):
    expected_alerts: int
    true_positives: int
    false_positives: int
    missed: int
    duplicates: int


class EvaluationReport(_Model):
    report_version: Literal["1"] = REPORT_VERSION
    run_id: str
    scenario_id: str
    seed: int
    source: str
    start: datetime
    duration: float
    people: list[str] = Field(default_factory=list, description="Resident ids; empty for single-person runs.")
    alert_tolerance: float
    counts: Counts
    detection_delays: list[float]
    segments: list[SegmentResult]
    alerts: list[AlertResult]
    timeline: list[TimelineEvent]


class EvaluationSummary(_Model):
    """Totals across many runs. These are simulated results, not real-world accuracy."""

    report_version: Literal["1"] = REPORT_VERSION
    runs: int
    scenarios: int
    counts: Counts
    delay_median: float | None
    delay_max: float | None
    false_positives_by_label: dict[str, int]
    missed_runs: list[str]
    false_positive_runs: list[str]

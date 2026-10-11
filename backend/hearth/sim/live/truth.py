"""Ground truth for a live run, written as it happens.

Each actor's truth label (Actor.label: walking, standing, sitting, bending,
lying_intentional, fall, absent) is checked every tick. When it changes, the open
segment closes and a new one opens, so segments per person are contiguous and cover the
whole run, just as a scripted scenario's do. A fall segment expects an alert. Labels that
flicker for less than MIN_SEGMENT_S (a half-step while turning) are folded into the
segment around them, so the record says what a person watching would say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hearth_schema.models import GroundTruth

from hearth.sim.scenario import ALERT_LABELS

MIN_SEGMENT_S = 0.4


@dataclass
class _Open:
    label: str
    t_start: float
    pending: str | None = None  # a new label not yet held long enough
    pending_since: float = 0.0


@dataclass
class TruthRecorder:
    scenario_id: str
    seed: int
    segments: list[GroundTruth] = field(default_factory=list)
    _open: dict[str, _Open] = field(default_factory=dict)
    _count: dict[str, int] = field(default_factory=dict)

    def update(self, t: float, labels: dict[str, str]) -> None:
        for pid, label in labels.items():
            cur = self._open.get(pid)
            if cur is None:
                self._open[pid] = _Open(label, t)
                continue
            if label == cur.label:
                cur.pending = None
                continue
            if cur.pending != label:
                cur.pending, cur.pending_since = label, t
            # Falls and leaving the room count from the first tick; others must hold a moment.
            if label in ALERT_LABELS or label == "absent" or t - cur.pending_since >= MIN_SEGMENT_S:
                at = cur.pending_since
                self._close(pid, cur, at)
                self._open[pid] = _Open(label, at)

    def _close(self, pid: str, cur: _Open, t_end: float) -> None:
        if t_end <= cur.t_start:
            return
        n = self._count.get(pid, 0) + 1
        self._count[pid] = n
        self.segments.append(GroundTruth(scenario_id=self.scenario_id, seed=self.seed, segment_id=f"{pid}-{n:03d}",
                                         person_id=pid, t_start=round(cur.t_start, 3), t_end=round(t_end, 3),
                                         truth_label=cur.label, expects_alert=cur.label in ALERT_LABELS))  # type: ignore[arg-type]

    def now(self) -> dict[str, tuple[str, str]]:
        """person_id -> (segment id the open segment will get, label)."""
        return {pid: (f"{pid}-{self._count.get(pid, 0) + 1:03d}", o.label) for pid, o in self._open.items()}

    def all(self, t: float) -> list[GroundTruth]:
        """Every segment so far, the open ones closed at t (without ending them)."""
        out = list(self.segments)
        for pid, cur in self._open.items():
            if t > cur.t_start:
                n = self._count.get(pid, 0) + 1
                out.append(GroundTruth(scenario_id=self.scenario_id, seed=self.seed, segment_id=f"{pid}-{n:03d}",
                                       person_id=pid, t_start=round(cur.t_start, 3), t_end=round(t, 3),
                                       truth_label=cur.label, expects_alert=cur.label in ALERT_LABELS))  # type: ignore[arg-type]
        return sorted(out, key=lambda g: (g.person_id or "", g.t_start))

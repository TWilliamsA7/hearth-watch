"""Live scenario controller: plays a simulated run through the real ingestion pipeline.

The controller pushes a scenario's Observations and Heartbeats into an injected
`ingest(items) -> events` function (the monitoring core's ingestion hook, see
docs/interfaces.md section 2.2) in real time, so the monitor, alerts and dashboard
treat it like a camera feed. It never sends ground truth; that stays here for the
dashboard's truth strip and the after-action report.

Time
----
Records are stamped on one continuous clock anchored at first play:
    timestamp = anchor + scenario_t
so the monitor sees the scenario's own timing at any playback speed and across pauses.
Pausing mid-fall and resuming must not change what the monitor concludes, so a pause
inserts no gap in record time. The cost: after a pause or at speed != 1, record time
drifts from the wall clock, so the backend must judge liveness (offline after 3 s) by
arrival time, not record timestamps (docs/interfaces.md section 2.4). Events map back
to scenario time as timestamp - anchor.

* Paused: no observations, but a heartbeat every second (stamped a millisecond after the
  last record), so a paused simulation is not reported as a camera outage. Inside a
  scripted camera_outage it stays silent.
* Ended: the source goes quiet (truthfully); SimStatus.state = "ended" tells the
  dashboard why.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from hearth_schema.models import Event, Heartbeat, Observation, TruthLabel
from pydantic import BaseModel, ConfigDict, Field

from hearth.eval.report import EvaluationReport
from hearth.eval.scoring import EvalConfig, evaluate, segment_at
from hearth.sim.generator import SIM_EPOCH, SOURCE
from hearth.sim.runner import RunInfo, SimRun, run_headless
from hearth.sim.scenario import Scenario, Scene, load_all
from hearth.sim.world import WorldTrack

Record = Observation | Heartbeat
IngestFn = Callable[[Sequence[Record]], Awaitable[Iterable[Event]]]
SimState = Literal["idle", "loaded", "playing", "paused", "ended"]

DEFAULT_LIVE_DIR = Path(__file__).resolve().parents[3] / "recordings" / "eval" / "live"
STAMP_STEP = timedelta(milliseconds=1)


class Clock(Protocol):
    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


class WallClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


# --- API models (exported to backend/schema/sim.schema.json) ---


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class ScenarioInfo(_Model):
    scenario_id: str
    title: str
    description: str
    duration: float = Field(description="Nominal seconds; each seed's run may differ by about 10%.")
    seed: int = Field(description="Default seed.")
    residents: list[str] = Field(description="Resident ids; empty for a single-person scenario.")
    scene: Scene

    @classmethod
    def of(cls, sc: Scenario) -> ScenarioInfo:
        return cls(scenario_id=sc.scenario_id, title=sc.title, description=sc.description.strip(),
                   duration=round(sc.duration, 3), seed=sc.seed,
                   residents=[r.resident_id for r in sc.cast() if r.resident_id], scene=sc.scene)  # fmt: skip


class TruthNow(_Model):
    """Ground truth at this instant, for people watching. Never sent to the monitor."""

    person_id: str | None
    segment_id: str
    truth_label: TruthLabel
    expects_alert: bool


class SimStatus(_Model):
    state: SimState
    scenario_id: str | None = None
    seed: int | None = None
    run_id: str | None = Field(default=None, description="Key for GET /eval/runs/{run_id} once the run ends.")
    t: float = Field(default=0.0, description="Scenario seconds.")
    duration: float | None = None
    speed: float = 1.0
    scene: Scene | None = None
    truth: list[TruthNow] = Field(default_factory=list)


# --- controller ---


@dataclass
class _LiveRun:
    run_id: str
    run: SimRun
    cursor: int = 0
    t: float = 0.0
    anchor: datetime | None = None  # the stamp given to scenario t = 0
    events: list[Event] = field(default_factory=list)
    last_stamp: datetime | None = None
    last_heartbeat: datetime | None = None  # wall time of the last pause heartbeat
    world: WorldTrack | None = None  # built on first request

    def scenario_t(self, ts: datetime) -> float:
        """Map a stamped time back onto the scenario clock."""
        if self.anchor is None:
            return 0.0
        return max(0.0, min((ts - self.anchor).total_seconds(), self.t))


class SimController:
    TICK_SECONDS = 0.05
    PAUSE_HEARTBEAT_SECONDS = 1.0
    MIN_SPEED, MAX_SPEED = 0.25, 4.0

    def __init__(
        self,
        ingest: IngestFn,
        *,
        clock: Clock | None = None,
        scenarios: dict[str, Scenario] | None = None,
        live_dir: Path | None = DEFAULT_LIVE_DIR,
        config: EvalConfig = EvalConfig(),
    ) -> None:
        self._ingest = ingest
        self.clock = clock or WallClock()
        self._scenarios = scenarios if scenarios is not None else load_all()
        self._live_dir = live_dir
        self._config = config
        self.state: SimState = "idle"
        self.speed = 1.0
        self._live: _LiveRun | None = None
        self._finished: dict[str, tuple[RunInfo, EvaluationReport]] = {}
        self._counter = itertools.count(1)
        self._task: asyncio.Task | None = None

    # --- queries ---

    def scenarios(self) -> list[ScenarioInfo]:
        return [ScenarioInfo.of(sc) for sc in self._scenarios.values()]

    def status(self) -> SimStatus:
        live = self._live
        if live is None:
            return SimStatus(state=self.state, speed=self.speed)
        sc = live.run.scenario
        truth = [
            TruthNow(person_id=g.person_id, segment_id=g.segment_id, truth_label=g.truth_label,
                     expects_alert=g.expects_alert)  # fmt: skip
            for pid in {g.person_id for g in live.run.ground_truth}
            if (g := segment_at(live.run.ground_truth, live.t, pid)) is not None
        ]
        return SimStatus(state=self.state, scenario_id=sc.scenario_id, seed=live.run.seed, run_id=live.run_id,
                         t=round(live.t, 3), duration=round(sc.duration, 3), speed=self.speed, scene=sc.scene,
                         truth=sorted(truth, key=lambda x: x.person_id or ""))  # fmt: skip

    def world(self) -> WorldTrack | None:
        """The loaded run's bodies in 3D, for the dashboard's 3D view; sample it at status().t."""
        live = self._live
        if live is None:
            return None
        if live.world is None:
            live.world = live.run.world()
        return live.world

    def runs(self) -> list[RunInfo]:
        return [info for info, _ in self._finished.values()]

    def report(self, run_id: str) -> EvaluationReport:
        return self._finished[run_id][1]

    # --- commands ---

    async def load(self, scenario_id: str, seed: int | None = None) -> SimStatus:
        if scenario_id not in self._scenarios:
            raise KeyError(scenario_id)
        await self._stop_task()
        run = run_headless(self._scenarios[scenario_id], seed)
        self._live = _LiveRun(run_id=f"{run.name}-live{next(self._counter)}", run=run)
        self.state = "loaded"
        return self.status()

    async def play(self, *, autorun: bool = True) -> SimStatus:
        """autorun=False leaves time to tick() (tests and custom loops)."""
        if self._live is None:
            raise RuntimeError("load a scenario first")
        if self.state == "ended":
            await self.load(self._live.run.scenario.scenario_id, self._live.run.seed)
        if self.state in ("loaded", "paused"):
            if self._live.anchor is None:
                self._live.anchor = self.clock.now()
            self.state = "playing"
        if autorun:
            self._ensure_task()
        return self.status()

    async def pause(self) -> SimStatus:
        if self.state == "playing" and self._live:
            self._live.last_heartbeat = None
            self.state = "paused"
        return self.status()

    async def reset(self) -> SimStatus:
        if self._live is not None:
            return await self.load(self._live.run.scenario.scenario_id, self._live.run.seed)
        return self.status()

    async def set_speed(self, speed: float) -> SimStatus:
        if not self.MIN_SPEED <= speed <= self.MAX_SPEED:
            raise ValueError(f"speed must be between {self.MIN_SPEED} and {self.MAX_SPEED}")
        self.speed = speed  # stamps follow scenario time, so nothing else changes
        return self.status()

    def observe(self, events: Iterable[Event]) -> None:
        """Record monitor events raised outside ingest() (e.g. the app's offline ticker)."""
        if self._live is not None:
            self._live.events.extend(e for e in events if e.source == SOURCE)

    # --- time ---

    async def tick(self, wall_seconds: float) -> None:
        """Advance by wall_seconds of real time: send what is due, or a heartbeat while paused."""
        live = self._live
        if live is None:
            return
        if self.state == "paused":
            await self._pause_heartbeat(live)
            return
        if self.state != "playing":
            return
        duration = live.run.scenario.duration
        live.t = min(live.t + wall_seconds * self.speed, duration)
        items = live.run.items
        due: list[Record] = []
        while live.cursor < len(items) and self._t_of(items[live.cursor]) <= live.t + 1e-9:
            due.append(self._stamp(live, items[live.cursor]))
            live.cursor += 1
        if due:
            self.observe(await self._ingest(due))
        if live.t >= duration - 1e-9 and live.cursor >= len(items):
            self.state = "ended"
            self._finish(live)

    def _t_of(self, item: Record) -> float:
        return (item.timestamp - SIM_EPOCH).total_seconds()

    def _stamp(self, live: _LiveRun, item: Record) -> Record:
        assert live.anchor is not None
        # Records in one frame (several people) share a stamp; pause heartbeats sit a millisecond
        # after the last record, so stamps never go backwards.
        ts = live.anchor + timedelta(seconds=self._t_of(item))
        if live.last_stamp is not None and ts < live.last_stamp:
            ts = live.last_stamp
        live.last_stamp = ts
        return item.model_copy(update={"timestamp": ts})

    async def _pause_heartbeat(self, live: _LiveRun) -> None:
        if any(g.truth_label == "camera_outage" for g in self.status().truth):
            return  # a scripted outage stays silent even while paused
        now = self.clock.now()
        if live.last_heartbeat and (now - live.last_heartbeat).total_seconds() < self.PAUSE_HEARTBEAT_SECONDS:
            return
        live.last_heartbeat = now
        ts = live.last_stamp + STAMP_STEP if live.last_stamp else now
        live.last_stamp = ts
        sc = live.run.scenario
        beat = Heartbeat(timestamp=ts, source=SOURCE, source_type="simulation", room_id=sc.room_id, fps=sc.fps)
        self.observe(await self._ingest([beat]))

    def _finish(self, live: _LiveRun) -> None:
        rebased = [e.model_copy(update={"timestamp": SIM_EPOCH + timedelta(seconds=live.scenario_t(e.timestamp))})
                   for e in live.events]  # fmt: skip
        report = evaluate(rebased, live.run.ground_truth, SIM_EPOCH, config=self._config)
        report = report.model_copy(update={"run_id": live.run_id})
        info = live.run.info().model_copy(update={"run_id": live.run_id})
        self._finished[live.run_id] = (info, report)
        if self._live_dir is not None:
            self._live_dir.mkdir(parents=True, exist_ok=True)
            (self._live_dir / f"{live.run_id}.report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # --- background loop ---

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def _loop(self) -> None:
        last = self.clock.now()
        while self.state in ("playing", "paused"):
            await self.clock.sleep(self.TICK_SECONDS)
            now = self.clock.now()
            await self.tick((now - last).total_seconds())
            last = now

    async def _stop_task(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def close(self) -> None:
        await self._stop_task()

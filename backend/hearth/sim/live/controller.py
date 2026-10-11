"""Live engine controller: runs a World in real time through the real ingestion pipeline.

The counterpart of hearth.sim.controller.SimController for the live engine, following
the same rules (docs/interfaces.md section 2.4):

* Records are stamped anchor + scenario t on one continuous clock, so pausing or
  changing speed never inserts a gap the monitor could read as something.
* Paused: no observations, a heartbeat every second, so a pause isn't an outage.
* Observations and heartbeats go through the injected ingest(); ground truth never does.

On top of that it pushes LiveFrames (who is where, doing what) to WebSocket viewers at
FRAME_HZ, independently of the engine's tick rate and the camera's frame rate, and takes
commands: a new room, play, pause, speed, behaviour settings, incidents, resets.

Only one simulated source plays at a time: `on_play` lets the router pause the scripted
controller when the live one starts, and vice versa.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

from hearth_schema.models import Event, Heartbeat

from hearth.eval.report import EvaluationReport
from hearth.eval.scoring import evaluate
from hearth.sim.controller import STAMP_STEP, Clock, IngestFn, Record, WallClock
from hearth.sim.generator import SIM_EPOCH, SOURCE

from hearth.sim.live.engine import World
from hearth.sim.live.incidents import IncidentRecord
from hearth.sim.live.models import ActorDetail, LiveActorInfo, LiveState, LiveStatus, LiveWorld
from hearth.sim.live.spec import EngineSpec, IncidentSpec

FRAME_HZ = 15.0
MAX_CATCH_UP_S = 0.5  # never simulate more than this per tick (a stalled loop doesn't fast-forward)
QUEUE_SIZE = 32


class LiveController:
    TICK_SECONDS = 0.05
    PAUSE_HEARTBEAT_SECONDS = 1.0
    MIN_SPEED, MAX_SPEED = 0.25, 8.0

    def __init__(self, ingest: IngestFn, *, clock: Clock | None = None,
                 on_play: Callable[[], Awaitable[None]] | None = None) -> None:  # fmt: skip
        self._ingest = ingest
        self.clock = clock or WallClock()
        self.on_play = on_play
        self.state: LiveState = "idle"
        self.speed = 1.0
        self.world: World | None = None
        self.run_id: str | None = None
        self._counter = itertools.count(1)
        self._anchor: datetime | None = None
        self._last_stamp: datetime | None = None
        self._last_heartbeat: datetime | None = None
        self._debt = 0.0  # scenario seconds owed to the engine (ticks are whole DT steps)
        self._events: list[Event] = []
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._since_frame = 0.0
        self._task: asyncio.Task | None = None
        # Called every tick, playing or paused (the live camera link: a real person keeps
        # moving while the simulation is paused).
        self.tick_hooks: list[Callable[[], None]] = []

    # --- queries ---

    def status(self) -> LiveStatus:
        w = self.world
        if w is None:
            return LiveStatus(state=self.state, speed=self.speed)
        return LiveStatus(state=self.state, run_id=self.run_id, scenario_id=w.scenario_id, seed=w.seed,
                          t=round(w.t, 3), speed=self.speed, actors=len(w.actors))  # fmt: skip

    def live_world(self) -> LiveWorld | None:
        return self.world.live_world(self.run_id or "") if self.world else None

    def actors(self) -> list[LiveActorInfo]:
        return [self.world.info(a) for a in self.world.actors] if self.world else []

    def detail(self, person_id: str) -> ActorDetail:
        if self.world is None:
            raise KeyError(person_id)
        return self.world.detail(person_id)

    def report(self) -> EvaluationReport:
        """Ground truth so far against the monitor's events so far (simulated results)."""
        if self.world is None or self._anchor is None:
            raise RuntimeError("nothing has played yet")
        anchor = self._anchor
        rebased = [e.model_copy(update={"timestamp": SIM_EPOCH + (e.timestamp - anchor)}) for e in self._events]
        report = evaluate(rebased, self.world.ground_truth(), SIM_EPOCH)
        return report.model_copy(update={"run_id": self.run_id})

    # --- commands ---

    async def new(self, spec: EngineSpec, seed: int) -> LiveWorld:
        """A new room and cast. Ready to play; starts paused."""
        await self._stop_task()
        self.world = World(spec, seed)
        self.run_id = f"{self.world.scenario_id}-live{next(self._counter)}"
        self._anchor = self._last_stamp = self._last_heartbeat = None
        self._debt, self._events = 0.0, []
        self.state = "paused"
        world = self.world.live_world(self.run_id)
        self._broadcast("world", world.model_dump_json())
        self._broadcast_frame()
        if self.tick_hooks:
            self._ensure_task()  # keep ticking while paused, for the live link
        return world

    def announce_world(self) -> None:
        """Send the room again (e.g. after the camera changed) to everyone watching."""
        if self.world is not None:
            self._broadcast("world", self.world.live_world(self.run_id or "").model_dump_json())

    def keep_running(self) -> None:
        """Tick even before play(), so tick hooks run and frames go out (the live link)."""
        if self.world is not None:
            self._ensure_task()

    async def reset(self) -> LiveWorld:
        if self.world is None:
            raise RuntimeError("generate a room first")
        playing = self.state == "playing"
        world = await self.new(self.world.spec, self.world.seed)
        if playing:
            await self.play()
        return world

    async def play(self, *, autorun: bool = True) -> LiveStatus:
        if self.world is None:
            raise RuntimeError("generate a room first")
        if self.on_play is not None:
            await self.on_play()
        if self._anchor is None:
            self._anchor = self.clock.now() - timedelta(seconds=self.world.t)
        self.state = "playing"
        if autorun:
            self._ensure_task()
        return self.status()

    async def pause(self) -> LiveStatus:
        if self.state == "playing":
            self.state = "paused"
            self._last_heartbeat = None
            self._broadcast_frame()
        return self.status()

    async def set_speed(self, speed: float) -> LiveStatus:
        if not self.MIN_SPEED <= speed <= self.MAX_SPEED:
            raise ValueError(f"speed must be between {self.MIN_SPEED} and {self.MAX_SPEED}")
        self.speed = speed
        return self.status()

    def configure(self, *, activity_frequency: float | None = None, social_frequency: float | None = None) -> EngineSpec:
        """Change behaviour settings mid-run (they apply to the next choices people make)."""
        if self.world is None:
            raise RuntimeError("generate a room first")
        update = {k: v for k, v in (("activity_frequency", activity_frequency), ("social_frequency", social_frequency))
                  if v is not None}  # fmt: skip
        spec = EngineSpec.model_validate({**self.world.spec.model_dump(), **update})
        self.world.spec = spec
        return spec

    def trigger(self, spec: IncidentSpec) -> IncidentRecord:
        if self.world is None:
            raise RuntimeError("generate a room first")
        rec = self.world.incidents.trigger(spec, source="manual")
        self._broadcast_frame()
        return rec

    def reset_actor(self, person_id: str) -> ActorDetail:
        if self.world is None:
            raise KeyError(person_id)
        self.world.actor(person_id)  # KeyError if unknown
        self.world.incidents.reset(person_id)
        return self.world.detail(person_id)

    def observe(self, events: list[Event]) -> None:
        """Monitor events raised outside ingest() (the app's offline ticker)."""
        if self.world is not None:
            self._events.extend(e for e in events if e.source == SOURCE and e.scenario_id == self.world.scenario_id)

    # --- viewers ---

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(QUEUE_SIZE)
        if self.world is not None:
            q.put_nowait(_message("world", self.world.live_world(self.run_id or "").model_dump_json()))
            q.put_nowait(_message("frame", self.world.frame(self.state, self.speed).model_dump_json()))
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    def _broadcast(self, kind: str, data_json: str) -> None:
        msg = _message(kind, data_json)
        for q in list(self._subscribers):
            if q.full():  # a slow viewer misses frames, never blocks the simulation
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(msg)

    def _broadcast_frame(self) -> None:
        if self.world is not None and self._subscribers:
            self._broadcast("frame", self.world.frame(self.state, self.speed).model_dump_json())

    # --- time ---

    async def tick(self, wall_seconds: float) -> None:
        w = self.world
        if w is None:
            return
        for hook in self.tick_hooks:
            hook()
        if self.state == "paused":
            await self._pause_heartbeat()
        elif self.state == "playing":
            self._debt += min(wall_seconds, MAX_CATCH_UP_S) * self.speed
            records: list[Record] = []
            while self._debt >= w.dt - 1e-9:
                records.extend(w.step())
                self._debt -= w.dt
            if records:
                events = await self._ingest([self._stamp(r) for r in records])
                self.observe(list(events))
        self._since_frame += wall_seconds
        if self._since_frame >= 1.0 / FRAME_HZ:
            self._since_frame = 0.0
            self._broadcast_frame()

    def _stamp(self, item: Record) -> Record:
        assert self._anchor is not None
        ts = self._anchor + (item.timestamp - SIM_EPOCH)
        if self._last_stamp is not None and ts < self._last_stamp:
            ts = self._last_stamp
        self._last_stamp = ts
        return item.model_copy(update={"timestamp": ts})

    async def _pause_heartbeat(self) -> None:
        if self._anchor is None:
            return  # never played: nothing to keep alive
        now = self.clock.now()
        if self._last_heartbeat and (now - self._last_heartbeat).total_seconds() < self.PAUSE_HEARTBEAT_SECONDS:
            return
        self._last_heartbeat = now
        ts = self._last_stamp + STAMP_STEP if self._last_stamp else now
        self._last_stamp = ts
        assert self.world is not None
        beat = Heartbeat(timestamp=ts, source=SOURCE, source_type="simulation", room_id=self.world.spec.room_id,
                         fps=self.world.spec.camera.fps)  # fmt: skip
        self.observe(list(await self._ingest([beat])))

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def _loop(self) -> None:
        last = self.clock.now()
        while self.world is not None:
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


def _message(kind: str, data_json: str) -> str:
    return '{"type": ' + json.dumps(kind) + ', "data": ' + data_json + "}"

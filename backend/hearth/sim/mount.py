"""Mount the simulator and evaluator on the monitoring core's app (docs/interfaces.md 2.2-2.3).

The app calls mount_simulation(app, monitor) once. Simulated records go through the
same MonitorService as camera data, through ingest() below: observations as
MonitorService.ingest() batches, heartbeats through MonitorService.heartbeat(), in the
order they were produced. The events they caused are collected through the service's
event_listeners hook.

Ground truth never passes here: only Observations and Heartbeats do.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from fastapi import FastAPI
from hearth_schema.models import Event, Heartbeat, Observation

from hearth.eval.api import create_eval_router
from hearth.sim.api import create_sim_router
from hearth.sim.controller import SimController
from hearth.sim.live.controller import LiveController
from hearth.sim.live.link import LiveLink, create_link_router


class Simulation:
    def __init__(self, monitor) -> None:  # monitor: hearth.monitor.service.MonitorService
        self.monitor = monitor
        self.scripted = SimController(self.ingest)
        self.live = LiveController(self.ingest)
        self.link = LiveLink(monitor, self.live)  # a live camera's person, played out in the live room

    @contextmanager
    def watch(self) -> Iterator[list[Event]]:
        """Collect the events the service records inside the block."""
        raised: list[Event] = []
        self.monitor.event_listeners.append(raised.append)
        try:
            yield raised
        finally:
            self.monitor.event_listeners.remove(raised.append)

    async def ingest(self, items: Sequence[Observation | Heartbeat]) -> list[Event]:
        with self.watch() as raised:
            batch: list[Observation] = []
            for item in items:
                if isinstance(item, Observation):
                    batch.append(item)
                    continue
                if batch:
                    self.monitor.ingest(batch)
                    batch = []
                self.monitor.heartbeat(item)
            if batch:
                self.monitor.ingest(batch)
        return raised

    def observe(self, events: list[Event]) -> None:
        """Events raised outside ingest() (the app's health ticker), for scoring live runs."""
        if events:
            self.scripted.observe(events)
            self.live.observe(events)

    async def close(self) -> None:
        await self.scripted.close()
        await self.live.close()


def mount_simulation(app: FastAPI, monitor) -> Simulation:
    sim = Simulation(monitor)
    app.include_router(create_sim_router(sim.scripted, sim.live), prefix="/sim")
    app.include_router(create_link_router(sim.link), prefix="/sim")
    app.include_router(create_eval_router(sim.scripted), prefix="/eval")
    app.state.simulation = sim
    return sim

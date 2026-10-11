"""Demo composition of the team's existing services; no teammate files changed.

From the repository root, with backend dependencies installed:
  python frontend/scripts/demo_backend.py
"""
from contextlib import asynccontextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "shared/python")]

from hearth.api.app import create_app
from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService
from hearth.sim.scenario import load_all
from hearth.sim.controller import SimController
from hearth.sim.api import create_sim_router
from hearth.eval.api import create_eval_router
from hearth_schema import Heartbeat, SourceStatusMessage


class DemoMonitor(MonitorService):
    controller = None

    def record(self, event):
        super().record(event)
        # Include health-ticker events, while excluding events from earlier runs.
        if self.controller and event.source_type == "simulation":
            if event.run_id == self.controller.status().run_id:
                self.controller.observe([event])


def create_demo_app(*, clock=None, utcnow=None):
    options = {}
    if clock is not None:
        options["clock"] = clock
    if utcnow is not None:
        options["utcnow"] = utcnow
    monitor = DemoMonitor(MonitorConfig.load(), **options)

    async def ingest(items):
        status = controller.status()
        # A reset deliberately retires the old run. Its later health ticker must
        # not overwrite the current run's health or invent an outage incident.
        for entry in monitor.health.values():
            old = entry.status
            if old.source_type == "simulation" and old.run_id != status.run_id and old.online:
                old.online = False
                old.observation_stale = True
                old.posture, old.posture_confidence = "unknown", None
                monitor.publish(SourceStatusMessage(data=old))
        for item in items:
            record = item.model_copy(update={"run_id": status.run_id, "scenario_id": status.scenario_id})
            if isinstance(record, Heartbeat):
                monitor.heartbeat(record)
            else:
                monitor.ingest([record])
        # DemoMonitor.record forwards actual events once, including health events.
        return []

    single_person = {key: sc for key, sc in load_all().items()
                     if len(sc.cast()) == 1 and sc.cast()[0].resident_id is None}
    controller = SimController(ingest, scenarios=single_person, live_dir=None)
    monitor.controller = controller
    app = create_app(service=monitor)
    app.state.controller = controller
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with original_lifespan(app):
            try:
                yield
            finally:
                await controller.close()

    app.router.lifespan_context = lifespan
    app.include_router(create_sim_router(controller), prefix="/sim")
    app.include_router(create_eval_router(controller), prefix="/eval")
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_demo_app(), host="127.0.0.1", port=8000)

"""HTTP routes for scenario control (/sim). Mounted by the monitoring core's app:

    controller = SimController(ingest)            # ingest = the app's ingestion function
    app.include_router(create_sim_router(controller), prefix="/sim")
    app.include_router(create_eval_router(controller), prefix="/eval")

See docs/interfaces.md section 3.3. These routes never expose ground truth beyond
SimStatus.truth, which is for people watching the dashboard, not for the monitor.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from hearth.sim.controller import ScenarioInfo, SimController, SimStatus


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    seed: int | None = None


class SpeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speed: float


def create_sim_router(controller: SimController) -> APIRouter:
    router = APIRouter(tags=["simulation"])

    @router.get("/scenarios", response_model=list[ScenarioInfo])
    async def scenarios() -> list[ScenarioInfo]:
        return controller.scenarios()

    @router.get("/status", response_model=SimStatus)
    async def status() -> SimStatus:
        return controller.status()

    @router.post("/load", response_model=SimStatus)
    async def load(body: LoadRequest) -> SimStatus:
        try:
            return await controller.load(body.scenario_id, body.seed)
        except KeyError:
            raise HTTPException(404, f"unknown scenario {body.scenario_id!r}") from None

    @router.post("/play", response_model=SimStatus)
    async def play() -> SimStatus:
        try:
            return await controller.play()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    @router.post("/pause", response_model=SimStatus)
    async def pause() -> SimStatus:
        return await controller.pause()

    @router.post("/reset", response_model=SimStatus)
    async def reset() -> SimStatus:
        return await controller.reset()

    @router.post("/speed", response_model=SimStatus)
    async def speed(body: SpeedRequest) -> SimStatus:
        try:
            return await controller.set_speed(body.speed)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    return router

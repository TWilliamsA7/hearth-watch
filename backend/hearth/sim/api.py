"""HTTP routes for scenario control (/sim). Mounted by the monitoring core's app:

    controller = SimController(ingest)            # ingest = the app's ingestion function
    live = LiveController(ingest)                 # the live engine, optional
    app.include_router(create_sim_router(controller, live), prefix="/sim")
    app.include_router(create_eval_router(controller), prefix="/eval")

The live engine's routes are under /sim/live (see docs/interfaces.md section 3.7),
including WS /sim/live/ws, which pushes {"type": "world" | "frame", "data": ...}.
Only one simulated source plays at a time: playing one pauses the other.

See docs/interfaces.md section 3.3. These routes never expose ground truth beyond
SimStatus.truth and the 3D world track (GET /world), which are for people watching the
dashboard, not for the monitor.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from hearth.eval.report import EvaluationReport
from hearth.sim.controller import ScenarioInfo, SimController, SimStatus
from hearth.sim.generate import GenSpec, generate
from hearth.sim.live.commands import ActorCommand, CommandError
from hearth.sim.live.controller import LiveController
from hearth.sim.live.incidents import IncidentError, IncidentRecord
from hearth.sim.live.models import ActorDetail, LiveActorInfo, LiveStatus, LiveWorld
from hearth.sim.live.spec import EngineSpec, IncidentSpec
from hearth.sim.world import WorldTrack


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    seed: int | None = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: GenSpec = GenSpec()
    seed: int = 0


class SpeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speed: float


class LiveNewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: EngineSpec = EngineSpec()
    seed: int = 0
    play: bool = Field(default=False, description="Start playing straight away.")


class LiveConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activity_frequency: float | None = Field(default=None, ge=0.25, le=4.0)
    social_frequency: float | None = Field(default=None, ge=0.0, le=4.0)


def create_sim_router(controller: SimController, live: LiveController | None = None) -> APIRouter:
    router = APIRouter(tags=["simulation"])
    if live is not None:
        _live_routes(router, controller, live)

    @router.get("/scenarios", response_model=list[ScenarioInfo])
    async def scenarios() -> list[ScenarioInfo]:
        return controller.scenarios()

    @router.get("/status", response_model=SimStatus)
    async def status() -> SimStatus:
        return controller.status()

    @router.get("/world", response_model=WorldTrack)
    async def world() -> WorldTrack:
        """The loaded run's bodies in 3D. Fetch once per load; sample at SimStatus.t."""
        track = controller.world()
        if track is None:
            raise HTTPException(404, "no scenario loaded")
        return track

    @router.post("/generate", response_model=ScenarioInfo)
    async def generate_scenario(body: GenerateRequest) -> ScenarioInfo:
        """Generate a scenario (hearth.sim.generate) and make it loadable; then POST /load it."""
        return controller.register(generate(body.spec, body.seed))

    @router.post("/load", response_model=SimStatus)
    async def load(body: LoadRequest) -> SimStatus:
        try:
            return await controller.load(body.scenario_id, body.seed)
        except KeyError:
            raise HTTPException(404, f"unknown scenario {body.scenario_id!r}") from None

    @router.post("/play", response_model=SimStatus)
    async def play() -> SimStatus:
        if live is not None:
            await live.pause()
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


def _live_routes(router: APIRouter, scripted: SimController, live: LiveController) -> None:
    async def pause_scripted() -> None:
        await scripted.pause()

    live.on_play = pause_scripted

    def need_world() -> None:
        if live.world is None:
            raise HTTPException(409, "generate a room first: POST /sim/live/new")

    @router.post("/live/new", response_model=LiveWorld, tags=["live simulation"])
    async def live_new(body: LiveNewRequest) -> LiveWorld:
        """Generate a room and its residents (same spec and seed, same run)."""
        world = await live.new(body.spec, body.seed)
        if body.play:
            await live.play()
        return world

    @router.get("/live/world", response_model=LiveWorld, tags=["live simulation"])
    async def live_world() -> LiveWorld:
        need_world()
        return live.live_world()  # type: ignore[return-value]

    @router.get("/live/status", response_model=LiveStatus, tags=["live simulation"])
    async def live_status() -> LiveStatus:
        return live.status()

    @router.post("/live/play", response_model=LiveStatus, tags=["live simulation"])
    async def live_play() -> LiveStatus:
        need_world()
        return await live.play()

    @router.post("/live/pause", response_model=LiveStatus, tags=["live simulation"])
    async def live_pause() -> LiveStatus:
        return await live.pause()

    @router.post("/live/reset", response_model=LiveWorld, tags=["live simulation"])
    async def live_reset() -> LiveWorld:
        """Back to t = 0: the same room and residents (same spec and seed)."""
        need_world()
        return await live.reset()

    @router.post("/live/speed", response_model=LiveStatus, tags=["live simulation"])
    async def live_speed(body: SpeedRequest) -> LiveStatus:
        try:
            return await live.set_speed(body.speed)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @router.post("/live/config", response_model=EngineSpec, tags=["live simulation"])
    async def live_config(body: LiveConfigRequest) -> EngineSpec:
        need_world()
        return live.configure(activity_frequency=body.activity_frequency, social_frequency=body.social_frequency)

    @router.get("/live/actors", response_model=list[LiveActorInfo], tags=["live simulation"])
    async def live_actors() -> list[LiveActorInfo]:
        return live.actors()

    @router.get("/live/actors/{person_id}", response_model=ActorDetail, tags=["live simulation"])
    async def live_actor(person_id: str) -> ActorDetail:
        try:
            return live.detail(person_id)
        except KeyError:
            raise HTTPException(404, f"no resident {person_id!r}") from None

    @router.post("/live/actors/{person_id}/reset", response_model=ActorDetail, tags=["live simulation"])
    async def live_reset_actor(person_id: str) -> ActorDetail:
        """End any incident; someone on the floor gets up."""
        try:
            return live.reset_actor(person_id)
        except KeyError:
            raise HTTPException(404, f"no resident {person_id!r}") from None

    @router.post("/live/actors/{person_id}/command", response_model=ActorDetail, tags=["live simulation"])
    async def live_command(person_id: str, body: ActorCommand) -> ActorDetail:
        """Have a resident sit or lie on a slot, or go, kneel, tie a shoe, pick something up, sit
        on the floor or fall at a floor point, now. Their plan is replaced; afterwards they carry
        on as usual."""
        need_world()
        assert live.world is not None
        if person_id not in {a.person_id for a in live.world.actors}:
            raise HTTPException(404, f"no resident {person_id!r}")
        try:
            return live.command(person_id, body)
        except KeyError:
            raise HTTPException(404, f"no slot {body.slot_id!r}") from None
        except CommandError as exc:
            raise HTTPException(409, str(exc)) from None

    @router.post("/live/incidents", response_model=IncidentRecord, tags=["live simulation"])
    async def live_incident(body: IncidentSpec) -> IncidentRecord:
        """Trigger an incident now (body.t is ignored). actor = None picks someone suitable."""
        need_world()
        try:
            return live.trigger(body)
        except KeyError:
            raise HTTPException(404, f"no resident {body.actor!r}") from None
        except IncidentError as exc:
            raise HTTPException(409, str(exc)) from None

    @router.get("/live/incidents", response_model=list[IncidentRecord], tags=["live simulation"])
    async def live_incidents() -> list[IncidentRecord]:
        return live.world.incidents.records if live.world else []

    @router.get("/live/report", response_model=EvaluationReport, tags=["live simulation"])
    async def live_report() -> EvaluationReport:
        """Ground truth so far vs the monitor's events so far. Simulated results only."""
        try:
            return live.report()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    @router.websocket("/live/ws")
    async def live_ws(websocket: WebSocket) -> None:
        """{"type": "world", "data": LiveWorld} on connect and per new room; then
        {"type": "frame", "data": LiveFrame} about 15 times a second."""
        await websocket.accept()
        queue = live.subscribe()

        async def send() -> None:
            while True:
                await websocket.send_text(await queue.get())

        async def receive() -> None:
            with suppress(WebSocketDisconnect):
                while True:
                    await websocket.receive_text()

        sender, receiver = asyncio.create_task(send()), asyncio.create_task(receive())
        try:
            await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            live.unsubscribe(queue)
            for task in (sender, receiver):
                task.cancel()
                with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await task

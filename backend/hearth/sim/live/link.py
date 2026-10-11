"""The live camera link: a real person, seen by a live camera, played out in the simulated room.

Every tick of the live controller (playing or paused), the link reads the monitoring core's
latest observation from the linked source (the Jetson, "jetson-01", or a replay), lifts its
keypoints into the room (hearth.sim.live.lift) using the posture the real monitor concluded,
and places the person there (World.set_external). When the monitor raises a possible fall
for that source, the nearest resident comes to help.

Nothing flows the other way: the simulator never feeds the live person's data back into the
monitor, and ground truth never reaches it.

    POST /sim/live/link {"enabled": true, "source": "jetson-01"}
    GET  /sim/live/link
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException
from hearth_schema.models import Event, Observation
from pydantic import BaseModel, ConfigDict, Field

from hearth.monitor.detector import track_key
from hearth.sim.live.controller import LiveController
from hearth.sim.live.avatar import LiveAvatar
from hearth.sim.live.calibrate import C270_FOV, CalibrationFrame, frame_from, solve
from hearth.sim.live.lift import LiftedBody

# No fresh frame for this long: the person is shown as gone. The Jetson pauses for a few
# seconds now and then (camera stalls); holding the last pose that long avoids flicker.
STALE_S = 4.0

log = logging.getLogger("uvicorn.error")  # shows in the backend's console

# The monitor's internal posture state -> (avatar posture, what to call it).
POSTURES: dict[str, tuple[str, str]] = {
    "upright": ("standing", "standing"),
    "standing_candidate": ("standing", "standing"),
    "descending": ("standing", "going down"),
    "sitting": ("sitting", "sitting"),
    "floor_level": ("floor", "on the floor"),
    "fallen": ("floor", "on the floor"),
    "possible_fall": ("floor", "possible fall"),
}


class LinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    source: str = Field(default="jetson-01", description="Camera source to follow, e.g. jetson-01 or replay-01.")
    height: float = Field(default=1.7, ge=1.2, le=2.1, description="The person's height in metres (sets depth).")


class CameraInfo(BaseModel):
    preset: str
    height: float
    tilt: float
    fov: float


class CalibrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seconds: float = Field(default=12.0, ge=1.0, le=30.0,
                           description="How long to watch the person walking about, upright, in view.")
    height: float = Field(default=1.7, ge=1.2, le=2.1, description="The person's height in metres.")


class CalibrateResult(BaseModel):
    ok: bool
    frames: int
    height: float | None = None
    tilt: float | None = None
    fov: float | None = None
    rms: float | None = Field(default=None, description="Fit error, as a fraction of the frame height.")
    reason: str = ""
    camera: CameraInfo | None = Field(default=None, description="The simulated camera now in use.")


class LinkStatus(BaseModel):
    enabled: bool
    source: str
    state: Literal["off", "waiting", "live", "stale"]
    person_id: str
    last_seen: datetime | None = None
    posture: str | None = None
    label: str | None = None
    alert_open: bool = False
    camera: CameraInfo | None = None


class LiveLink:
    def __init__(self, monitor, controller: LiveController) -> None:  # monitor: MonitorService
        self.monitor, self.controller = monitor, controller
        self.enabled, self.source, self.height = False, "jetson-01", 1.7
        self.alert_open = False
        self._posture, self._label = "standing", "standing"
        self.avatar = LiveAvatar(self.height)  # the person's body in the room, smoothed
        self._last_obs_id: str | None = None
        self._last_seen: datetime | None = None
        self._present = False
        self._person_at: float | None = None  # monitor clock when someone was last made out
        controller.tick_hooks.append(self.update)
        monitor.event_listeners.append(self._on_event)

    @property
    def person_id(self) -> str:
        return f"live-{self.source}"

    def configure(self, req: LinkRequest) -> LinkStatus:
        w = self.controller.world
        if w is not None and (not req.enabled or req.source != self.source):
            w.remove_external(self.source)
        if req.source != self.source or req.height != self.height:
            self.avatar = LiveAvatar(req.height)
        if req.source != self.source:
            self._last_seen = None
            self.alert_open = False
        self.enabled, self.source, self.height = req.enabled, req.source, req.height
        if self.enabled:
            self.controller.keep_running()
        return self.status()

    def status(self) -> LinkStatus:
        state: Literal["off", "waiting", "live", "stale"]
        if not self.enabled:
            state = "off"
        elif self._last_seen is None:
            state = "waiting"
        else:
            state = "live" if self._present else "stale"
        return LinkStatus(enabled=self.enabled, source=self.source, state=state, person_id=self.person_id,
                          last_seen=self._last_seen, posture=self._posture if self._present else None,
                          label=self._label if self._present else None, alert_open=self.alert_open,
                          camera=self._camera())  # fmt: skip

    def _camera(self) -> CameraInfo | None:
        w = self.controller.world
        if w is None:
            return None
        c = w.spec.camera
        return CameraInfo(preset=c.preset, height=c.height, tilt=c.tilt, fov=w.camera.fov)

    async def calibrate(self, req: CalibrateRequest) -> CalibrateResult:
        """Watch the linked person walking about upright, work out the real camera's height and
        tilt, and use them for the simulated camera."""
        loop = asyncio.get_running_loop()
        frames: list[CalibrationFrame] = []
        seen: set[str] = set()
        end = loop.time() + req.seconds
        while loop.time() < end:
            obs = self._latest()
            if obs is not None and obs.obs_id not in seen and obs.keypoints and self._fresh():
                seen.add(obs.obs_id)
                if (f := frame_from([(k.name, k.x, k.y) for k in obs.keypoints])) is not None:
                    frames.append(f)
            await asyncio.sleep(0.05)
        result = await asyncio.to_thread(solve, frames, req.height, fov=C270_FOV)
        out = CalibrateResult(ok=result.ok, frames=result.frames, height=result.height, tilt=result.tilt,
                              fov=result.fov, rms=result.rms, reason=result.reason)  # fmt: skip
        log.warning("calibration from %s: %d frames seen, %d usable -> %s", self.source, len(seen), result.frames,
                    f"height {result.height} m, tilt {result.tilt} deg, fit {result.rms}" if result.ok
                    else f"FAILED: {result.reason}")  # fmt: skip
        w = self.controller.world
        if result.ok and w is not None and result.height is not None and result.tilt is not None:
            spec = w.spec.camera.model_copy(update={"preset": "jetson", "height": result.height,
                                                    "tilt": result.tilt, "fov": result.fov})  # fmt: skip
            w.set_camera(spec)
            self.height = req.height
            self.avatar = LiveAvatar(req.height)  # positions were measured with the old camera
            self.controller.announce_world()
        out.camera = self._camera()
        return out

    # --- every tick ---

    def _latest(self) -> Observation | None:
        mine = [o for o in self.monitor.observations.values() if o.source == self.source]
        return max(mine, key=lambda o: o.timestamp) if mine else None

    def _fresh(self) -> bool:
        now = self.monitor.clock()
        return any(key[2] == self.source and entry.last_observation_receipt is not None
                   and now - entry.last_observation_receipt < STALE_S
                   for key, entry in self.monitor.health.items())  # fmt: skip

    def _monitor_posture(self, obs: Observation) -> tuple[str, str]:
        state = self.monitor.detector.states.get(track_key(obs))
        return POSTURES.get(state.posture if state else "", (self._posture, self._label))

    def update(self) -> None:
        w = self.controller.world
        if not self.enabled or w is None:
            return
        obs = self._latest()
        if obs is not None:
            self._last_seen = obs.timestamp
        now = self.monitor.clock()
        seen = obs is not None and self._fresh() and obs.person_detected and bool(obs.keypoints)
        if not seen:
            # poseNet misses people for a frame or two; only call them gone after STALE_S.
            gone = self._person_at is None or now - self._person_at > STALE_S
            if self._present and gone:
                w.set_external(self.source, None)
                self._present = False
                self.avatar = LiveAvatar(self.height)
            elif self._present:
                self._place(w, now, None)  # keep any posture change animating
            return
        self._person_at = now
        posture, self._label = self._monitor_posture(obs)
        new = obs.obs_id != self._last_obs_id
        self._last_obs_id = obs.obs_id
        self._place(w, now, [(k.name, k.x, k.y) for k in obs.keypoints] if new else None, posture)

    def _place(self, w, now: float, keypoints, posture: str | None = None) -> None:
        a = self.avatar
        a.update(keypoints, posture or a.posture, w.camera, w.room, now)
        if not a.ready:
            return
        self._posture = a.posture
        e = w.set_external(self.source, LiftedBody(a.joints(), a.root, a.heading, a.height), posture=a.posture,
                           label=self._label)  # fmt: skip
        self._present = True
        if self.alert_open and a.posture == "floor":
            w.incidents.respond(e)

    def _on_event(self, event: Event) -> None:
        if event.source != self.source:
            return
        if event.event_type == "possible_fall":
            self.alert_open = True
        elif event.event_type == "recovered":
            self.alert_open = False


def create_link_router(link: LiveLink) -> APIRouter:
    router = APIRouter(tags=["simulation"])

    @router.get("/live/link", response_model=LinkStatus)
    async def link_status() -> LinkStatus:
        return link.status()

    @router.post("/live/link/calibrate", response_model=CalibrateResult)
    async def link_calibrate(body: CalibrateRequest) -> CalibrateResult:
        """Walk slowly around, upright and in view: works out the real camera's height and tilt."""
        if link.controller.world is None:
            raise HTTPException(409, "generate a room first (POST /sim/live/new)")
        if not link.enabled:
            raise HTTPException(409, "link a live source first (POST /sim/live/link)")
        return await link.calibrate(body)

    @router.post("/live/link", response_model=LinkStatus)
    async def link_configure(body: LinkRequest) -> LinkStatus:
        if body.enabled and link.controller.world is None:
            raise HTTPException(409, "generate a room first (POST /sim/live/new)")
        return link.configure(body)

    return router

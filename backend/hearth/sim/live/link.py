"""The live camera link: real people, seen by a live camera, played out in the simulated room.

Every tick of the live controller (playing or paused), the link reads the monitoring core's
latest observation for each person on the linked source (the Jetson, "jetson-01", or a replay),
lifts their keypoints into the room (hearth.sim.live.lift) using the posture the real monitor
concluded for that person, and places them there (World.set_external). A camera that reports
several people tags each with an anonymous person_id ("p1", ...); each becomes their own live
person in the room ("live-jetson-01-p1"). One without person_ids gives one ("live-jetson-01").
The camera's own posture model call on each person goes along for display. When the monitor
raises a possible fall for someone, the nearest resident comes to help them.

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
from hearth.sim.live.actor import external_id
from hearth.sim.live.controller import LiveController
from hearth.sim.live.avatar import LiveAvatar
from hearth.sim.live.calibrate import C270_FOV, CalibrationFrame, frame_from, solve, solve_pan
from hearth.sim.live.lift import LiftedBody
from hearth.sim.live.models import LiveModelCall

# No fresh frame for this long: the person is shown as gone. The Jetson pauses for a few
# seconds now and then (camera stalls); holding the last pose that long avoids flicker.
STALE_S = 4.0
# One person's latest frame this far behind the camera's newest: they're not in view any more
# (the camera still sees others). The edge keeps someone's id through gaps up to 1 s.
PERSON_GAP_S = 1.0

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
    corner: str = "left"
    pan: float = 0.0


class CalibrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: Literal["walk", "centre"] = Field(
        default="walk", description="walk: height and tilt (walk about); centre: pan (stand on the centre mark).")
    seconds: float | None = Field(default=None, ge=1.0, le=30.0,
                                  description="How long to watch; default 12 s for walk, 3 s for centre.")
    height: float = Field(default=1.7, ge=1.2, le=2.1, description="The person's height in metres.")


class CalibrateResult(BaseModel):
    ok: bool
    step: str = "walk"
    frames: int
    height: float | None = None
    tilt: float | None = None
    fov: float | None = None
    rms: float | None = Field(default=None, description="Fit error, as a fraction of the frame height.")
    pan: float | None = Field(default=None, description="Step centre: degrees the camera is turned (+ = right).")
    off_by: float | None = Field(default=None, description="Step centre: metres between where you stood and the mark.")
    reason: str = ""
    camera: CameraInfo | None = Field(default=None, description="The simulated camera now in use.")


class LivePersonStatus(BaseModel):
    person_id: str = Field(description="Their id in the room, e.g. live-jetson-01-p2.")
    track_id: str | None = Field(default=None, description="The camera's anonymous track id (p2); None: one person.")
    state: Literal["live", "stale", "outside"]
    posture: str | None = None
    label: str | None = None
    model: LiveModelCall | None = Field(default=None, description="The camera's own posture model on them.")
    alert_open: bool = False
    recognized: bool = Field(default=False, description="Their face matched an enrollment (Jetson face ID).")
    name: str | None = Field(default=None, description="Their enrolled name, when they agreed to be named.")


class LinkStatus(BaseModel):
    enabled: bool
    source: str
    state: Literal["off", "waiting", "live", "stale", "outside"]
    person_id: str = Field(description="Room id of the first live person (live-<source> when nobody is tracked).")
    last_seen: datetime | None = None
    posture: str | None = None
    label: str | None = None
    alert_open: bool = Field(default=False, description="True when the monitor has a possible fall open for anyone.")
    camera: CameraInfo | None = None
    people: list[LivePersonStatus] = Field(default_factory=list, description="Everyone the camera is tracking.")


class LiveTrack:
    """One real person on the linked camera: their smoothed body in the room, and what the
    monitor (posture) and the camera's own model say about them."""

    def __init__(self, track_id: str | None, height: float) -> None:
        self.track_id = track_id  # the camera's person_id; None for a one-person camera
        self.avatar = LiveAvatar(height)
        self.posture, self.label = "standing", "standing"
        self.recognized, self.name = False, None  # from the monitor's face matching, if any
        self.model: LiveModelCall | None = None
        self.present = False  # in the room right now
        self.seen_at: float | None = None  # monitor clock when they were last made out
        self.last_obs_id: str | None = None

    def state(self) -> Literal["live", "stale", "outside"]:
        return "outside" if self.avatar.outside else "live" if self.present else "stale"


def model_call(obs: Observation) -> LiveModelCall | None:
    p = obs.posture_prediction
    return None if p is None else LiveModelCall(model_id=p.model_id, label=p.label, confidence=p.confidence)


class LiveLink:
    def __init__(self, monitor, controller: LiveController) -> None:  # monitor: MonitorService
        self.monitor, self.controller = monitor, controller
        self.enabled, self.source, self.height = False, "jetson-01", 1.7
        self.tracks: dict[str | None, LiveTrack] = {}  # by the camera's person_id
        self.alerts: set[str | None] = set()  # person_ids with a possible fall open
        self._last_seen: datetime | None = None
        controller.tick_hooks.append(self.update)
        monitor.event_listeners.append(self._on_event)

    @property
    def alert_open(self) -> bool:
        return bool(self.alerts)

    def person_id(self, track_id: str | None = None) -> str:
        return external_id(self.source, track_id)

    def configure(self, req: LinkRequest) -> LinkStatus:
        w = self.controller.world
        if w is not None and (not req.enabled or req.source != self.source):
            w.remove_external(self.source)
        if req.source != self.source or not req.enabled:
            self.tracks = {}
        if req.height != self.height:
            for track in self.tracks.values():
                track.avatar = LiveAvatar(req.height)
        if req.source != self.source:
            self._last_seen = None
            self.alerts = set()
        self.enabled, self.source, self.height = req.enabled, req.source, req.height
        if self.enabled:
            self.controller.keep_running()
        return self.status()

    def status(self) -> LinkStatus:
        people = [LivePersonStatus(person_id=self.person_id(t.track_id), track_id=t.track_id, state=t.state(),
                                   posture=t.posture if t.present else None, label=t.label if t.present else None,
                                   model=t.model, alert_open=t.track_id in self.alerts,
                                   recognized=t.recognized, name=t.name)
                  for _, t in sorted(self.tracks.items(), key=lambda kv: _order(kv[0]))]  # fmt: skip
        state: Literal["off", "waiting", "live", "stale", "outside"]
        if not self.enabled:
            state = "off"
        elif self._last_seen is None:
            state = "waiting"
        else:
            states = {p.state for p in people}
            state = "live" if "live" in states else "outside" if "outside" in states else "stale"
        first = next((p for p in people if p.state == "live"), people[0] if people else None)
        return LinkStatus(enabled=self.enabled, source=self.source, state=state,
                          person_id=first.person_id if first else self.person_id(), last_seen=self._last_seen,
                          posture=first.posture if first else None, label=first.label if first else None,
                          alert_open=self.alert_open, camera=self._camera(), people=people)  # fmt: skip

    def _camera(self) -> CameraInfo | None:
        w = self.controller.world
        if w is None:
            return None
        c = w.spec.camera
        return CameraInfo(preset=c.preset, height=c.height, tilt=c.tilt, fov=w.camera.fov, corner=c.corner, pan=c.pan)

    async def calibrate(self, req: CalibrateRequest) -> CalibrateResult:
        """Step "walk": watch the linked person walking about upright; work out the real camera's
        height and tilt. Step "centre": watch them standing on the area's centre mark; work out
        which way a corner camera points (pan). Each result is applied to the simulated camera."""
        loop = asyncio.get_running_loop()
        frames: list[CalibrationFrame] = []
        seen: set[str] = set()
        crowded = 0  # frames skipped because more than one person was in view
        seconds = req.seconds if req.seconds is not None else (12.0 if req.step == "walk" else 3.0)
        end = loop.time() + seconds
        while loop.time() < end:
            people = [o for o in self._latest_by_person().values() if o.person_detected and o.keypoints]
            if people and self._fresh() and not all(o.obs_id in seen for o in people):
                seen.update(o.obs_id for o in people)
                if len(people) > 1:  # the fit assumes one person of the given height
                    crowded += 1
                elif (f := frame_from([(k.name, k.x, k.y) for k in people[0].keypoints or []])) is not None:
                    frames.append(f)
            await asyncio.sleep(0.05)
        w = self.controller.world
        if w is None:
            raise RuntimeError("no room")
        cam, room = w.spec.camera, w.room
        if req.step == "centre":
            pan = await asyncio.to_thread(solve_pan, frames, req.height, cam, room.width, room.depth)
            out = CalibrateResult(ok=pan.ok, step="centre", frames=pan.frames, pan=pan.pan, off_by=pan.off_by,
                                  reason=pan.reason)  # fmt: skip
            summary = f"pan {pan.pan} deg, {pan.off_by} m off the mark" if pan.ok else f"FAILED: {pan.reason}"
            update = {"pan": pan.pan} if pan.ok else None
        else:
            preset = cam.preset if cam.preset in ("jetson", "jetson_corner") else "jetson"
            spec = cam.model_copy(update={"preset": preset})
            result = await asyncio.to_thread(solve, frames, req.height, fov=C270_FOV, spec=spec, width=room.width,
                                             depth=room.depth)  # fmt: skip
            out = CalibrateResult(ok=result.ok, step="walk", frames=result.frames, height=result.height,
                                  tilt=result.tilt, fov=result.fov, rms=result.rms, reason=result.reason)  # fmt: skip
            summary = (f"height {result.height} m, tilt {result.tilt} deg, fit {result.rms}" if result.ok
                       else f"FAILED: {result.reason}")  # fmt: skip
            update = ({"preset": preset, "height": result.height, "tilt": result.tilt, "fov": result.fov}
                      if result.ok else None)  # fmt: skip
        if not out.ok and crowded:
            out.reason = f"{crowded} frames had more than one person in view: calibrate alone ({out.reason})"
        log.warning("calibration (%s) from %s: %d frames seen, %d usable, %d crowded -> %s", req.step, self.source,
                    len(seen), out.frames, crowded, summary)  # fmt: skip
        if update is not None:
            w.set_camera(w.spec.camera.model_copy(update=update))
            self.height = req.height
            for track in self.tracks.values():  # positions were measured with the old camera
                track.avatar = LiveAvatar(req.height)
            self.controller.announce_world()
        out.camera = self._camera()
        return out

    # --- every tick ---

    def _latest_by_person(self) -> dict[str | None, Observation]:
        """The newest observation of each person on the linked source (by the camera's person_id)."""
        out: dict[str | None, Observation] = {}
        for o in self.monitor.observations.values():
            if o.source == self.source and (o.person_id not in out or o.timestamp > out[o.person_id].timestamp):
                out[o.person_id] = o
        return out

    def _fresh(self) -> bool:
        now = self.monitor.clock()
        return any(key[2] == self.source and entry.last_observation_receipt is not None
                   and now - entry.last_observation_receipt < STALE_S
                   for key, entry in self.monitor.health.items())  # fmt: skip

    def _monitor_posture(self, obs: Observation, track: LiveTrack) -> tuple[str, str]:
        state = self.monitor.detector.states.get(track_key(obs))
        return POSTURES.get(state.posture if state else "", (track.posture, track.label))

    def update(self) -> None:
        w = self.controller.world
        if not self.enabled or w is None:
            return
        latest = self._latest_by_person()
        if latest:
            self._last_seen = max(o.timestamp for o in latest.values())
        now, fresh = self.monitor.clock(), self._fresh()
        # A camera reporting several people sends one person_id-less "nobody in view" record
        # when it sees no one; that record isn't a person.
        several = any(pid is not None for pid in latest)
        for pid, obs in latest.items():
            if (pid is None and several) or not (fresh and obs.person_detected and obs.keypoints):
                continue
            if self._last_seen is not None and (self._last_seen - obs.timestamp).total_seconds() > PERSON_GAP_S:
                continue
            track = self.tracks.get(pid)
            if track is None:
                track = self.tracks[pid] = LiveTrack(pid, self.height)
            track.seen_at = now
            posture, track.label = self._monitor_posture(obs, track)
            track.model = model_call(obs)
            identity = self.monitor.identity_of(obs) if self.monitor.identity_of else {}
            track.recognized, track.name = bool(identity.get("recognized")), identity.get("name")
            new = obs.obs_id != track.last_obs_id
            track.last_obs_id = obs.obs_id
            self._place(w, track, now, [(k.name, k.x, k.y) for k in obs.keypoints] if new else None, posture)
        for pid, track in list(self.tracks.items()):
            if track.seen_at == now:
                continue
            # poseNet misses people for a frame or two; only call them gone after STALE_S.
            if track.seen_at is None or now - track.seen_at > STALE_S:
                w.forget_external(self.source, pid)
                del self.tracks[pid]
            elif track.present:
                self._place(w, track, now, None)  # keep any posture change animating

    def _place(self, w, track: LiveTrack, now: float, keypoints, posture: str | None = None) -> None:
        a = track.avatar
        a.update(keypoints, posture or a.posture, w.camera, w.room, now)
        if not a.ready:
            return
        if a.outside:  # left the tracked area: not in the simulated room
            if track.present:
                w.set_external(self.source, None, person_id=track.track_id)
                track.present = False
            return
        track.posture = a.posture
        e = w.set_external(self.source, LiftedBody(a.joints(), a.root, a.heading, a.height),
                           person_id=track.track_id, posture=a.posture, label=track.label,
                           model=track.model)  # fmt: skip
        track.present = True
        if track.track_id in self.alerts and a.posture == "floor":
            w.incidents.respond(e)

    def _on_event(self, event: Event) -> None:
        if event.source != self.source:
            return
        if event.event_type == "possible_fall":
            self.alerts.add(event.person_id)
        elif event.event_type == "recovered":
            self.alerts.discard(event.person_id)


def _order(track_id: str | None) -> tuple[int, str]:
    """p2 before p10; a one-person camera's track (None) first."""
    if track_id is None:
        return (-1, "")
    digits = "".join(ch for ch in track_id if ch.isdigit())
    return (int(digits) if digits else 0, track_id)


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

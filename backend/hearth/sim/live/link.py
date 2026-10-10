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

import math
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException
from hearth_schema.models import Event, Observation
from pydantic import BaseModel, ConfigDict, Field

from hearth.monitor.detector import track_key
from hearth.sim.live.controller import LiveController
from hearth.sim.live.lift import LiftedBody, lift

STALE_S = 1.5  # no fresh frame for this long: the person is shown as gone
SMOOTH = 0.35  # EMA weight of each new frame on position and heading

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


class LinkStatus(BaseModel):
    enabled: bool
    source: str
    state: Literal["off", "waiting", "live", "stale"]
    person_id: str
    last_seen: datetime | None = None
    posture: str | None = None
    label: str | None = None
    alert_open: bool = False


class LiveLink:
    def __init__(self, monitor, controller: LiveController) -> None:  # monitor: MonitorService
        self.monitor, self.controller = monitor, controller
        self.enabled, self.source, self.height = False, "jetson-01", 1.7
        self.alert_open = False
        self._posture, self._label = "standing", "standing"
        self._root: tuple[float, float] | None = None
        self._heading: float | None = None
        self._last_seen: datetime | None = None
        self._present = False
        controller.tick_hooks.append(self.update)
        monitor.event_listeners.append(self._on_event)

    @property
    def person_id(self) -> str:
        return f"live-{self.source}"

    def configure(self, req: LinkRequest) -> LinkStatus:
        w = self.controller.world
        if w is not None and (not req.enabled or req.source != self.source):
            w.remove_external(self.source)
        if req.source != self.source:
            self._root = self._heading = self._last_seen = None
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
                          label=self._label if self._present else None, alert_open=self.alert_open)  # fmt: skip

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

    def _smooth(self, body: LiftedBody) -> LiftedBody:
        if self._root is None or self._heading is None:
            self._root, self._heading = body.root, body.heading
            return body
        rx = self._root[0] + SMOOTH * (body.root[0] - self._root[0])
        rz = self._root[1] + SMOOTH * (body.root[1] - self._root[1])
        turn = math.remainder(body.heading - self._heading, 2 * math.pi)
        self._root, self._heading = (rx, rz), self._heading + SMOOTH * turn
        dx, dz = rx - body.root[0], rz - body.root[1]
        joints = {k: (p[0] + dx, p[1], p[2] + dz) for k, p in body.joints.items()}
        return LiftedBody(joints, (rx, rz), self._heading, body.height)

    def update(self) -> None:
        w = self.controller.world
        if not self.enabled or w is None:
            return
        obs = self._latest()
        if obs is not None:
            self._last_seen = obs.timestamp
        if obs is None or not self._fresh() or not obs.person_detected or not obs.keypoints:
            if self._present:
                w.set_external(self.source, None)
                self._present = False
                self._root = self._heading = None
            return
        posture, label = self._monitor_posture(obs)
        body = lift([(k.name, k.x, k.y) for k in obs.keypoints], w.camera, w.room, height=self.height,
                    posture=posture)  # fmt: skip
        if body is None:
            return  # keep where they were: one unusable frame isn't someone leaving
        if posture != self._posture and "floor" in (posture, self._posture):
            self._root = None  # lying down or getting up moves the body: don't drag it through the change
        self._posture, self._label = posture, label
        e = w.set_external(self.source, self._smooth(body), posture=posture, label=label)
        self._present = True
        if self.alert_open and posture == "floor":
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

    @router.post("/live/link", response_model=LinkStatus)
    async def link_configure(body: LinkRequest) -> LinkStatus:
        if body.enabled and link.controller.world is None:
            raise HTTPException(409, "generate a room first (POST /sim/live/new)")
        return link.configure(body)

    return router

"""World: the live engine's fixed-step loop.

Every DT seconds of scenario time, in this order:
  1. due incidents fire (scheduled, procedural)
  2. each actor continues its activity, or the Brain gives it a new plan
  3. conversations start, gather and end
  4. ground truth notes any label changes
  5. if a camera frame is due, the camera reports what it sees

All randomness comes from generators seeded by (seed, purpose), consumed in a fixed
order, so the same spec and seed always give the same run, frame for frame. Behaviour
runs at 1/DT Hz; the camera at its own fps; the live controller pushes frames to
viewers at yet another rate. None of them depend on each other.
"""

from __future__ import annotations

import math
import random
from datetime import datetime

from hearth_schema.models import GroundTruth, Heartbeat, Observation

from hearth.sim.generator import SIM_EPOCH, NoiseConfig
from hearth.sim.pose import KEY_POSES
from hearth.sim.scenario import Scene, SceneObject
from hearth.sim.world import (
    PERSON_HEIGHT_M,
    WorldDoor,
    WorldFrame,
    WorldObject,
    WorldRoom,
    WorldRoute,
    WorldTrack,
    WorldWindow,
    world_person,
)

from hearth.sim.live.activities import COOLDOWN_S, Activity, Away, Enter, Hold, Walk, abandon, choose, release, reserve, seat_stance
from hearth.sim.live.actor import LIVE_LOOKS, Actor, ExternalActor, external_profile, make_profile
from hearth.sim.live.animate import world_pose
from hearth.sim.live.camera import Camera, Observer
from hearth.sim.live.incidents import IncidentManager
from hearth.sim.live.lift import LiftedBody
from hearth.sim.live.models import (
    ActorDetail,
    LiveActorInfo,
    LiveAppearance,
    LiveConversation,
    LiveFrame,
    LiveNav,
    LivePerson,
    LivePlace,
    LiveSlot,
    LiveState,
    LiveWorld,
)
from hearth.sim.live.room import LEGACY_KIND, Room, generate_room
from hearth.sim.live.social import Social, waiting
from hearth.sim.live.spec import CameraSpec, EngineSpec
from hearth.sim.live.truth import TruthRecorder

DT = 0.05
Record = Observation | Heartbeat


class World:
    dt = DT

    def __init__(self, spec: EngineSpec, seed: int, *, start: datetime = SIM_EPOCH, scenario_id: str | None = None) -> None:
        self.spec, self.seed = spec, seed
        self.scenario_id = scenario_id or f"live-{seed}"
        self.t = 0.0
        self.room: Room = generate_room(spec.room, seed)
        self.camera = Camera(self.room, spec.camera)
        noise = NoiseConfig() if spec.noise else NoiseConfig.off()
        self.observer = Observer(self.camera, scenario_id=self.scenario_id, seed=seed, room_id=spec.room_id,
                                 noise=noise, start=start)  # fmt: skip
        self.social = Social()
        self.truth = TruthRecorder(self.scenario_id, seed)
        self.actors: list[Actor] = []
        self._by_id: dict[str, Actor] = {}
        self.externals: dict[str, ExternalActor] = {}  # live people, by person_id ("live-<source>")
        self.routes: list[WorldRoute] = []
        self._walks: dict[str, tuple[float, list]] = {}  # person -> (t_start, path) of the walk in progress
        self._spawn()
        self.incidents = IncidentManager(self)
        self.truth.update(0.0, {a.person_id: a.label for a in self.actors})

    # --- people ---

    def _spawn(self) -> None:
        rng = random.Random(f"cast/{self.seed}")
        seats = [s for s in self.room.free_slots("sit") if self.room.pieces[s.piece].kind.name != "bed"]
        rng.shuffle(seats)
        spots = [p.at for p in self.room.places if p.name in ("middle", "window", "browse")]
        rng.shuffle(spots)
        for i in range(self.spec.actors):
            pid = f"p{i + 1}"
            profile, looks = make_profile(rng)
            a = Actor(pid, profile, looks, random.Random(f"actor/{self.seed}/{pid}"), (0.0, -1.0), 0.0)
            a.idle_phase = rng.uniform(0, 2 * math.pi)
            self.actors.append(a)
            self._by_id[pid] = a
            roll = rng.random()
            if roll < 0.15 and self.spec.actors > 2:  # out, coming in a little later
                a.posture = "absent"
                a.pos = self.room.doors[0].outside
                a.plan.extend([Away(rng.uniform(3.0, 25.0)), Enter(rng.choice(self.room.doors).door_id)])
                a.intent = "arrive"
            elif roll < 0.6 and seats:
                seat = seats.pop()
                reserve(a, self, seat)
                a.pos, a.heading, a.posture = seat.at, seat.heading, "sitting"
                a.body = a.rest_pose = seat_stance(seat, a.size).body
                a.plan.append(Hold(rng.uniform(4.0, 45.0), "seated"))
                a.intent = "sit"
            else:
                at = spots.pop() if spots else self._free_point(rng)
                if any(math.dist(at, o.pos) < 0.8 for o in self.actors if o is not a and o.present):
                    at = self._free_point(rng)
                a.pos, a.heading = at, rng.uniform(-math.pi, math.pi)
                a.plan.append(Hold(rng.uniform(1.0, 12.0), "pausing"))
                a.intent = "pause"

    def _free_point(self, rng: random.Random) -> tuple[float, float]:
        nav = self.room.nav
        reach = nav.reachable(self.room.doors[0].inside)
        for _ in range(400):
            p = (rng.uniform(nav.x_min, nav.x_max), rng.uniform(nav.z_min, nav.z_max))
            if nav.free(p) and nav._cell(p) in reach and nav.clearance(p) > 0.3 and \
                    all(math.dist(p, o.pos) > 0.9 for o in self.actors if o.present):  # fmt: skip
                return (round(p[0], 3), round(p[1], 3))
        return self.room.doors[0].inside

    def actor(self, person_id: str) -> Actor:
        return self._by_id[person_id]

    @property
    def bodies(self) -> list[Actor]:
        """Everyone physically in the room, live people included: for steering and space."""
        return [*self.actors, *self.externals.values()]

    def set_external(self, source: str, lifted: LiftedBody | None, *, posture: str = "standing",
                     label: str = "standing", seen_at: float = 0.0) -> ExternalActor:
        """Place (or update) the live person seen by `source`; lifted=None means they're gone."""
        pid = f"live-{source}"
        e = self.externals.get(pid)
        if e is None:
            height = lifted.height if lifted else 1.7
            e = ExternalActor(pid, external_profile(height), LIVE_LOOKS, random.Random(pid), (0.0, -1.0), 0.0,
                              posture="absent", source=source)  # fmt: skip
            e.intent = "live"
            self.externals[pid] = e
        if lifted is None:
            e.posture, e.joints = "absent", {}
            return e
        e.pos, e.heading, e.joints = lifted.root, lifted.heading, dict(lifted.joints)
        e.posture = posture if posture in ("standing", "sitting", "lying", "floor") else "standing"  # type: ignore[assignment]
        e.live_label, e.seen_at = label, seen_at
        return e

    def set_camera(self, spec: CameraSpec) -> None:
        """Swap the simulated camera (e.g. after calibrating it against the real one), in place."""
        self.spec = self.spec.model_copy(update={"camera": spec})
        self.camera = Camera(self.room, spec)
        self.observer.camera = self.camera

    def remove_external(self, source: str) -> None:
        self.externals.pop(f"live-{source}", None)

    def present(self) -> list[Actor]:
        return [a for a in self.actors if a.present]

    def replace_plan(self, a: Actor, plan: list[Activity], intent: str, *, keep_social: bool = False) -> None:
        """Drop what a is doing and do this instead (incidents, being drawn into a conversation)."""
        if a.activity is not None:
            a.activity.end(a, self)
            a.activity = None
        abandon(a, self)
        if not keep_social:
            self.social.leave(self, a)
        a.path, a.destination = [], None
        self._end_walk(a)
        a.plan.extend(plan)
        a.intent, a.intent_since = intent, self.t
        a.history.append((intent, round(self.t, 2)))

    # --- the loop ---

    def step(self) -> list[Record]:
        self.t = round(self.t + DT, 6)
        self.incidents.update()
        for a in self.actors:
            self._update(a)
        self.social.update(self)
        self.truth.update(self.t, {a.person_id: a.label for a in self.actors})
        if self.observer.due(self.t):
            # Live people are in the simulated camera's view too (they hide whoever is behind
            # them), but what it reports about them is dropped: the real camera reports them.
            people = [(a.person_id, world_pose(a), a.heading, 0.0, a.posture in ("lying", "floor"))
                      for a in self.bodies if a.present]  # fmt: skip
            return self.observer.emit(self.t, [p for p in people if p[1]], silent=frozenset(self.externals))
        return []

    def advance(self, seconds: float) -> list[Record]:
        out: list[Record] = []
        for _ in range(max(0, round(seconds / DT))):
            out.extend(self.step())
        return out

    def _update(self, a: Actor) -> None:
        a.steps += 1
        if a.activity is None:
            if not a.plan:
                if a.engaged_by is not None:
                    a.plan.extend(waiting(a, self))
                else:
                    intent, plan = choose(a, self)
                    a.intent, a.intent_since = intent, self.t
                    a.history.append((intent, round(self.t, 2)))
                    a.plan.extend(plan)
            a.activity = a.plan.popleft()
            if isinstance(a.activity, Walk):
                self._walks[a.person_id] = (self.t, [])
        act = a.activity
        if act.update(a, self, DT):
            act.end(a, self)
            a.activity = None
            if isinstance(act, Walk):
                self._end_walk(a, act.path)
            if act.failed:
                abandon(a, self)
                if a.incident:
                    self.incidents.abandoned(a)
                self.social.release_engaged(self, a)
                a.cooldowns[a.intent] = self.t + 5.0
            elif not a.plan and a.intent in COOLDOWN_S:
                a.cooldowns[a.intent] = self.t + COOLDOWN_S[a.intent] * a.rng.uniform(0.7, 1.3)

    def _end_walk(self, a: Actor, path: list | None = None) -> None:
        started = self._walks.pop(a.person_id, None)
        if started is not None and path and len(path) >= 2:
            self.routes.append(WorldRoute(person_id=a.person_id, t_start=round(started[0], 3), t_end=round(self.t, 3),
                                          points=[(round(x, 3), round(z, 3)) for x, z in path]))  # fmt: skip

    # --- views ---

    def world_room(self) -> WorldRoom:
        r, cam = self.room, self.camera
        objects = []
        for p in r.pieces:
            sx, sz = p.size
            h = p.kind.h
            k = p.kind
            objects.append(WorldObject(kind=LEGACY_KIND[k.name], center=(p.at[0], round(h / 2, 3), p.at[1]),  # type: ignore[arg-type]
                                       size=(round(sx, 3), round(h, 3), round(sz, 3)), rotation=round(p.rotation, 4),
                                       variant=k.name, seat_height=k.seat or None, surface_height=k.surface or None,
                                       arm_width=k.arm or None, arm_height=k.arm_h or None,
                                       back_depth=k.back or None))  # fmt: skip
        return WorldRoom(
            width_front=r.width, width_back=r.width, depth=r.depth, height=r.height,
            camera_position=_r3(cam.position), camera_target=_r3(cam.target), camera_fov=cam.fov,
            objects=objects, bounds=r.bounds, camera_aspect=round(cam.aspect, 4), lighting=self.spec.lighting,
            doors=[WorldDoor(door_id=d.door_id, wall=d.wall, center=d.at, width=d.width, heading_in=round(d.heading_in, 4))
                   for d in r.doors],
            windows=[WorldWindow(wall=w.wall, center=w.at, width=w.width, sill=w.sill, height=w.height) for w in r.windows],
        )  # fmt: skip

    def scene(self) -> Scene:
        """The furniture as rectangles in the camera image, for the 2D camera view."""
        objs = []
        for p in sorted(self.room.pieces, key=lambda p: -_dist_to(self.camera.position, p.at)):
            x0, z0, x1, z1 = p.footprint()
            corners = [(x, y, z) for x in (x0, x1) for y in (0.0, p.kind.h) for z in (z0, z1)]
            pts = [q for q in (self.camera.project(c) for c in corners) if q is not None]
            if len(pts) < 8:
                continue
            xs = [min(max(q[0], 0.0), 1.0) for q in pts]
            ys = [min(max(q[1], 0.0), 1.0) for q in pts]
            if max(xs) - min(xs) < 0.005 or max(ys) - min(ys) < 0.005:
                continue
            objs.append(SceneObject(kind=LEGACY_KIND[p.kind.name], x=round(min(xs), 4), y=round(min(ys), 4),  # type: ignore[arg-type]
                                    w=round(max(xs) - min(xs), 4), h=round(max(ys) - min(ys), 4)))  # fmt: skip
        return Scene(frame_aspect=self.camera.aspect, objects=objs)

    def info(self, a: Actor) -> LiveActorInfo:
        p, ap = a.profile, a.appearance
        return LiveActorInfo(person_id=a.person_id, archetype=p.archetype, height=p.height, speed=p.speed,
                             sociability=p.sociability, mobility=p.mobility,
                             appearance=LiveAppearance(skin=ap.skin, top=ap.top, bottom=ap.bottom, hair=ap.hair,
                                                       build=ap.build))  # fmt: skip

    def person(self, a: Actor) -> LivePerson:
        wp = world_person(a.person_id, a.heading, world_pose(a), a.size)
        act = a.activity
        live = isinstance(a, ExternalActor)
        return LivePerson(
            **wp.model_dump(), label=a.label, activity=f"live · {a.source}" if live else act.name if act else "idle",
            intent=a.intent, posture=a.posture, kind="live" if live else "simulated",
            destination=_r2(a.destination) if a.destination else None, path=[_r2(p) for p in a.path[:40]],
            conversation=a.conversation, incident=a.incident, visible=round(self.observer.visible.get(a.person_id, 0.0), 3),
        )  # fmt: skip

    def frame(self, state: LiveState = "playing", speed: float = 1.0) -> LiveFrame:
        return LiveFrame(
            t=round(self.t, 3), state=state, speed=speed,
            people=[self.person(a) for a in self.bodies if a.present],
            conversations=[LiveConversation(conv_id=c.conv_id, members=list(c.members), centre=_r2(c.centre))
                           for c in self.social.conversations.values()],
            incidents=self.incidents.active(),
            slots_taken={s.slot_id: s.occupant for s in self.room.slots if s.occupant},
        )  # fmt: skip

    def live_world(self, run_id: str) -> LiveWorld:
        return LiveWorld(run_id=run_id, scenario_id=self.scenario_id, seed=self.seed, spec=self.spec,
                         room=self.world_room(), scene=self.scene(), actors=[self.info(a) for a in self.actors],
                         slots=self.live_slots(), places=[LivePlace(name=p.name, at=_r2(p.at), heading=round(p.heading, 4))
                                                         for p in self.room.places],
                         nav=self.live_nav())  # fmt: skip

    def live_slots(self) -> list[LiveSlot]:
        out = []
        for s in self.room.slots:
            sit = s.kind == "sit"
            approach = self.room.approach(s, PERSON_HEIGHT_M) if sit else None
            out.append(LiveSlot(slot_id=s.slot_id, piece=s.piece, kind=s.kind, at=_r2(s.at), heading=round(s.heading, 4),
                                height=s.seat if sit else self.room.pieces[s.piece].kind.surface, group=s.group,
                                stand=_r2(s.stand_point(PERSON_HEIGHT_M)) if sit else None,
                                approach=_r2(approach) if approach else None))  # fmt: skip
        return out

    def live_nav(self) -> LiveNav:
        nav = self.room.nav
        rows = ["".join("0" if nav.free(nav._centre((i, j))) else "1" for i in range(nav.nx)) for j in range(nav.nz)]
        return LiveNav(cell=nav.cell, origin=_r2((nav.x_min, nav.z_min)), cols=nav.nx, rows=rows)

    def detail(self, person_id: str) -> ActorDetail:
        a = self.actor(person_id)
        return ActorDetail(info=self.info(a), now=self.person(a), since=round(a.intent_since, 2),
                           plan=[x.name for x in a.plan][:12], history=list(a.history),
                           cooldowns={k: round(v, 1) for k, v in a.cooldowns.items() if v > self.t},
                           incidents=[r for r in self.incidents.records if r.person_id == person_id])  # fmt: skip

    def world_frame(self) -> WorldFrame:
        return WorldFrame(t=round(self.t, 6), people=[world_person(a.person_id, a.heading, world_pose(a), a.size)
                                                       for a in self.actors if a.present])  # fmt: skip

    def ground_truth(self) -> list[GroundTruth]:
        return self.truth.all(self.t)


def _r2(p: tuple[float, float]) -> tuple[float, float]:
    return (round(p[0], 3), round(p[1], 3))


def _r3(p: tuple[float, float, float]) -> tuple[float, float, float]:
    return (round(p[0], 3), round(p[1], 3), round(p[2], 3))


def _dist_to(cam: tuple[float, float, float], p: tuple[float, float]) -> float:
    return math.hypot(cam[0] - p[0], cam[2] - p[1])


def run_world(spec: EngineSpec, seed: int, duration: float, *, start: datetime = SIM_EPOCH,
              scenario_id: str | None = None, frames: bool = True) -> tuple[World, list[Record], WorldTrack | None]:  # fmt: skip
    """Run headless for `duration` seconds: the world at the end, every record, and the world track."""
    w = World(spec, seed, start=start, scenario_id=scenario_id)
    items: list[Record] = []
    track: list[WorldFrame] = []
    every = max(1, round(1.0 / spec.camera.fps / DT))
    n = 0
    while w.t < duration - 1e-9:
        items.extend(w.step())
        n += 1
        if frames and n % every == 0:
            track.append(w.world_frame())
    for a in w.actors:
        w._end_walk(a, a.activity.path if isinstance(a.activity, Walk) else None)
    world = None
    if frames:
        world = WorldTrack(scenario_id=w.scenario_id, seed=seed, fps=spec.camera.fps, duration=round(w.t, 3),
                           room=w.world_room(), frames=track, routes=w.routes)  # fmt: skip
    return w, items, world




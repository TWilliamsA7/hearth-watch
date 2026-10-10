"""Observation generator: turns a scenario + seed into what a camera would have reported.

Output is Observations and Heartbeats only, built as plain dicts the way the edge client
builds them and validated against the shared contract. Ground truth is produced
separately by Scenario.ground_truth() and never travels in this stream.

Several residents: each frame yields one Observation per person the camera can see,
tagged with person_id. People further back are hidden where someone nearer the camera
stands in front of them, as one camera really would. A frame where nobody is visible
yields a single Observation with person_detected = false and no person_id.

Determinism: one random.Random(seed) per run, consumed in a fixed order, and timestamps
from a virtual clock. The same scenario and seed always produce identical output.
"""

from __future__ import annotations

import bisect
import dataclasses
import math
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from hearth_schema.models import Heartbeat, Observation

from hearth.sim.features import MIN_CONFIDENCE, FeatureTracker
from hearth.sim.motion import Key, timeline
from hearth.sim.pose import (
    JOINTS,
    KEY_POSES,
    BodyPose,
    blend_body,
    body_pose,
    gait_amount,
    hips_of,
    smoothstep,
    stride_length,
)
from hearth.sim.scenario import PoseName, Resident, Scenario, Scene, Segment
from hearth.sim.world import (
    HEADING,
    PERSON_HEIGHT_M,
    ImagePose,
    WorldFrame,
    WorldPerson,
    WorldPose,
    WorldRoute,
    WorldTrack,
    body,
    frame_depth,
    frame_x,
    lift,
    room_geometry,
    to_image,
    world_person,
    world_x,
    world_z,
)

SIM_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
"""Default virtual start time for headless runs, so output doesn't depend on the wall clock."""

SOURCE = "sim"

LABEL_POSE: dict[str, PoseName] = {
    "walking": "standing",
    "standing": "standing",
    "sitting": "sitting",
    "bending": "bending",
    "lying_intentional": "lying",
    "fall": "fallen",
}
"""Labels not listed (occluded, absent, camera_outage) hold the previous pose."""

LIES_ALONG: frozenset[str] = frozenset({"lying", "fallen", "supine", "prone", "side_lying"})
"""Poses that, without a heading of their own, are drawn side-on (as hand-written scenarios expect)."""

MIN_VISIBLE_KEYPOINTS = 4

BOX_PADDING = 0.01
"""How far around a nearer person's keypoints their body hides people behind them."""

ACCEL_M = 0.6
"""A walk speeds up over about this distance at the start and slows down over it at the end."""
CORNER_M = 0.3
"""Heading turns from one leg of a walk to the next over this distance either side of the corner."""

Box = tuple[float, float, float, float]  # x0, y0, x1, y1
Floor = tuple[float, float]  # (x, z) metres


@dataclass(frozen=True)
class NoiseConfig:
    """Perception imperfections. NoiseConfig.off() gives clean, ideal observations."""

    jitter: float = 0.004  # keypoint position noise (std dev, normalized units)
    confidence_mean: float = 0.85
    confidence_sd: float = 0.06
    keypoint_dropout: float = 0.03  # chance a visible keypoint is missed
    frame_drop: float = 0.01  # chance a whole frame is lost (heartbeats continue)
    lying_penalty: float = 0.15  # pose nets are less confident on people lying down
    face_rate_front: float = 0.95
    face_rate_side: float = 0.35
    face_rate_lying: float = 0.15
    face_rate_away: float = 0.02
    # How the Jetson edge emits (edge/run_edge.py); all off by default. See edge_like().
    keypoint_confidence: bool = True  # False: drop undetected joints, send no confidence (poseNet)
    face_check_interval: float = 0.0  # > 0: face fields only on check frames, null otherwise
    dropout_bridge: float = 0.0  # > 0: skip (don't send "nobody") when one person vanishes this briefly

    @classmethod
    def edge_like(cls) -> NoiseConfig:
        """Observations shaped like the Jetson's: no keypoint confidence, a face check every
        0.5 s, and brief dropouts skipped rather than sent as "no person"."""
        return cls(keypoint_confidence=False, face_check_interval=0.5, dropout_bridge=0.3)

    @classmethod
    def off(cls) -> NoiseConfig:
        return cls(jitter=0.0, confidence_sd=0.0, keypoint_dropout=0.0, frame_drop=0.0, lying_penalty=0.0,
                   face_rate_front=1.0, face_rate_side=0.0, face_rate_lying=0.0, face_rate_away=0.0)  # fmt: skip


@dataclass(frozen=True)
class Stance:
    """A body at one instant, before it is put in the room: pose, heading, where, how high."""

    body: BodyPose
    heading: float
    x: float  # root on the floor, metres (world x, z)
    z: float
    base: float  # metres above the floor (a bed's surface)

    def flipped(self) -> Stance:
        """The same body described from the opposite heading (no visible change)."""
        return Stance({j: (-f, v, -l) for j, (f, v, l) in self.body.items()}, self.heading + math.pi,
                      self.x, self.z, self.base)  # fmt: skip


def _turn(a: float, b: float) -> float:
    """The short way round from heading a to heading b."""
    return (b - a + math.pi) % (2.0 * math.pi) - math.pi


def blend_stance(a: Stance, b: Stance, alpha: float) -> Stance:
    if alpha <= 0.0:
        return a
    if alpha >= 1.0:
        return b
    return Stance(blend_body(a.body, b.body, alpha), a.heading + _turn(a.heading, b.heading) * alpha,
                  a.x + (b.x - a.x) * alpha, a.z + (b.z - a.z) * alpha, a.base + (b.base - a.base) * alpha)  # fmt: skip


RUN_INTO: frozenset[str] = frozenset({"trip", "collision", "stumble"})
"""Motions a walk runs straight into, without slowing down first."""


def walk_time(length: float, speed: float, *, stops: bool = True) -> float:
    """Seconds to walk `length` metres cruising at `speed`, speeding up first (and slowing
    down at the end if it stops there): the duration that makes a walk cruise at that speed."""
    a = min(ACCEL_M, length / 3)
    return (length + a * (2 if stops else 1)) / speed


@dataclass(frozen=True, eq=False)
class _Walk:
    """A run of consecutive walking segments, walked as one: speeding up, cruising and
    slowing down, unless it runs straight into a trip or a bump (stops=False)."""

    t0: float
    duration: float
    points: list[Floor]
    cum: list[float]  # distance along the route to each point
    stops: bool = True

    @property
    def length(self) -> float:
        return self.cum[-1]

    @property
    def cruise(self) -> float:
        return walk_time(self.length, 1.0, stops=self.stops) / self.duration

    def distance(self, t: float) -> tuple[float, float]:
        """(metres along the route, speed) t seconds into the walk."""
        L, D = self.length, self.duration
        t = min(max(t, 0.0), D)
        if L < 0.05:
            return L * t / D, L / D
        a, v = min(ACCEL_M, L / 3), self.cruise
        ta = 2 * a / v
        if t < ta:
            return 0.5 * v / ta * t * t, v * t / ta
        if self.stops and t > D - ta:
            r = D - t
            return L - 0.5 * v / ta * r * r, v * r / ta
        return a + v * (t - ta), v

    def at(self, s: float) -> tuple[Floor, float | None]:
        """(point, heading) s metres along the route; heading None on a route with no length."""
        pts, cum = self.points, self.cum
        k = max(0, min(bisect.bisect_right(cum, s) - 1, len(pts) - 2))
        a, b = pts[k], pts[k + 1]
        seg = cum[k + 1] - cum[k]
        r = min(max((s - cum[k]) / seg, 0.0), 1.0) if seg > 1e-9 else 1.0
        point = (a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r)
        heading = self._dir(k)
        if heading is None:
            return point, None
        # Round the corners: start turning before one, finish after it.
        for corner, near, far in ((k + 1, k, k + 1), (k, k - 1, k)):
            if not 0 < corner < len(pts) - 1:
                continue
            h0, h1 = self._dir(near), self._dir(far)
            if h0 is None or h1 is None:
                continue
            rad = min(CORNER_M, (cum[corner] - cum[near]) / 2, (cum[far + 1] - cum[corner]) / 2)
            if rad > 1e-6 and abs(s - cum[corner]) < rad:
                return point, h0 + _turn(h0, h1) * smoothstep((s - cum[corner] + rad) / (2 * rad))
        return point, heading

    def _dir(self, k: int) -> float | None:
        a, b = self.points[k], self.points[k + 1]
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-6:
            return None
        return math.atan2(b[0] - a[0], b[1] - a[1])


@dataclass(frozen=True)
class _Plan:
    """Where a segment starts and what it is heading for, resolved once per run."""

    segment: Segment
    t_start: float
    pose: PoseName  # the pose it ends in (for a motion, its last key's)
    heading: float  # radians, the heading it ends with
    start: Floor
    end: Floor
    base: float  # metres above the floor it ends at
    x_start: float  # frame x and depth of start and end, as the scenario gives them
    x_end: float
    depth_start: float
    depth_end: float
    walk: _Walk | None = None
    keys: list[Key] | None = None


@dataclass(frozen=True)
class PersonState:
    """One resident at one instant: the noise-free truth of where their body is."""

    track: ResidentTrack
    plan: _Plan
    pose: ImagePose
    depth: float
    world: WorldPose
    heading: float

    @property
    def label(self) -> str:
        return self.plan.segment.label

    @property
    def lying(self) -> bool:
        """Is the body along the floor (or a bed) rather than upright?"""
        nx, ny, nz = self.world["neck"]
        hx, hy, hz = (sum(c) / 2 for c in zip(self.world["left_hip"], self.world["right_hip"]))
        return abs(ny - hy) < math.hypot(nx - hx, nz - hz)


class ResidentTrack:
    """One resident's motion on the scenario clock. Pure geometry, no randomness."""

    def __init__(self, resident: Resident, scene: Scene) -> None:
        self.resident = resident
        self.scene = scene
        self.size = PERSON_HEIGHT_M * resident.height
        self.plans = self._plan_segments()
        self._starts = [p.t_start for p in self.plans]
        self._start_stances: dict[int, Stance] = {}

    @property
    def person_id(self) -> str | None:
        return self.resident.resident_id

    def _world(self, x: float, depth: float) -> Floor:
        return world_x(self.scene, x, depth), world_z(depth)

    def _plan_segments(self) -> list[_Plan]:
        res = self.resident
        plans: list[_Plan] = []
        pose: PoseName = "standing"
        heading = res.start_heading if res.start_heading is not None else HEADING[res.start_facing]
        x, depth = res.start_x, res.start_depth
        base = 0.0
        for seg, (t0, _) in zip(res.segments, res.segment_times()):
            label_pose = seg.pose or LABEL_POSE.get(seg.label, pose)
            x_end = x if seg.x is None else seg.x
            depth_end = depth if seg.depth is None else seg.depth
            start, end = self._world(x, depth), self._world(x_end, depth_end)
            explicit = seg.heading is not None or seg.facing is not None
            if seg.heading is not None:
                heading = seg.heading
            elif seg.facing is not None:
                heading = HEADING[seg.facing]
            elif seg.label == "walking" and math.hypot(end[0] - start[0], end[1] - start[1]) > 1e-6:
                heading = math.atan2(end[0] - start[0], end[1] - start[1])
            keys = None
            if seg.motion is not None or seg.label == "fall":
                motion = seg.motion or "collapse"
                pace = seg.transition if motion == "collapse" else 1.0
                keys = timeline(motion, seg.variant, pace=pace, fit=seg.duration)
                label_pose = _pose_name(keys[-1].pose) or label_pose
            if label_pose in LIES_ALONG and not explicit and abs(math.sin(heading)) < math.sin(math.pi / 4):
                heading = math.pi / 2  # lying towards or away from the camera: draw it side-on instead
            if seg.label not in ("absent", "camera_outage", "occluded") or seg.surface_y is not None:
                base = lift(self.scene, seg.surface_y, depth_end)
            pose = label_pose
            plans.append(_Plan(seg, t0, pose, heading, start, end, base, x, x_end, depth, depth_end, keys=keys))
            x, depth = x_end, depth_end
        return self._join_walks(plans)

    @staticmethod
    def _join_walks(plans: list[_Plan]) -> list[_Plan]:
        """Consecutive walking segments are one walk."""
        out: list[_Plan] = []
        i = 0
        while i < len(plans):
            if plans[i].segment.label != "walking":
                out.append(plans[i])
                i += 1
                continue
            j = i
            while j < len(plans) and plans[j].segment.label == "walking":
                j += 1
            run = plans[i:j]
            points = [run[0].start] + [p.end for p in run]
            cum = [0.0]
            for a, b in zip(points, points[1:]):
                cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
            after = plans[j].segment if j < len(plans) else None
            stops = after is None or after.motion not in RUN_INTO
            walk = _Walk(run[0].t_start, sum(p.segment.duration for p in run), points, cum, stops)
            out.extend(dataclasses.replace(p, walk=walk) for p in run)
            i = j
        return out

    def segment_index(self, t: float) -> int:
        return max(0, bisect.bisect_right(self._starts, t + 1e-9) - 1)

    def _progress(self, i: int, tau: float) -> float:
        """How far a still segment i has moved towards its target position (0..1)."""
        seg = self.plans[i].segment
        return smoothstep(tau / seg.transition if seg.transition else 1.0)

    def _target(self, i: int, tau: float) -> Stance:
        """Where segment i's own motion puts the body at local time tau, before blending in."""
        p = self.plans[i]
        if p.walk is not None:
            w = p.walk
            s, speed = w.distance(p.t_start + tau - w.t0)
            (x, z), heading = w.at(s)
            stride = stride_length(w.cruise, self.size)
            amount = gait_amount(stride, self.size) * (0.3 + 0.7 * min(speed / w.cruise, 1.0))
            gait = body_pose("standing", 2.0 * math.pi * s / stride, amount)
            return Stance(gait, p.heading if heading is None else heading, x, z, 0.0)
        k = self._progress(i, tau)
        x, z = p.start[0] + (p.end[0] - p.start[0]) * k, p.start[1] + (p.end[1] - p.start[1]) * k
        return Stance(body_pose(p.pose), p.heading, x, z, p.base)

    def _start_stance(self, i: int) -> Stance:
        if i not in self._start_stances:
            if i == 0:
                self._start_stances[i] = self._target(0, 0.0)
            else:
                self._start_stances[i] = self.stance(i - 1, self.plans[i - 1].segment.duration)
        return self._start_stances[i]

    def stance(self, i: int, tau: float) -> Stance:
        """The body during segment i at local time tau."""
        p = self.plans[i]
        seg = p.segment
        tau = min(tau, seg.duration)  # after their script ends, a resident holds their last pose
        if p.keys is not None and i > 0:
            return self._play(i, tau)
        target = self._target(i, tau)
        if i == 0 or seg.transition == 0.0 or tau >= seg.transition:
            return target
        if p.walk is not None and self.plans[i - 1].walk is p.walk:
            return target  # one leg of a walk runs straight on into the next
        return blend_stance(self._start_stance(i), target, smoothstep(tau / seg.transition))

    def _play(self, i: int, tau: float) -> Stance:
        """A motion's keys, from wherever the body was when the segment began."""
        p = self.plans[i]
        assert p.keys is not None
        start = self._start_stance(i)
        if _flat(start.body) and _head_forward(start.body) != _head_forward(p.keys[0].pose):
            start = start.flipped()  # lying the other way round: describe it from the other end
        turn = _turn(start.heading, p.heading)

        def at(key: Key) -> Stance:
            return Stance(key.pose, start.heading + turn * key.turn,
                          p.start[0] + (p.end[0] - p.start[0]) * key.move,
                          p.start[1] + (p.end[1] - p.start[1]) * key.move,
                          start.base + (p.base - start.base) * key.move)  # fmt: skip

        prev_t, prev = 0.0, start
        for key in p.keys:
            if tau < key.at:
                span = key.at - prev_t
                return blend_stance(prev, at(key), key.ease((tau - prev_t) / span) if span > 0 else 1.0)
            prev_t, prev = key.at, at(key)
        return prev

    def _world_pose(self, st: Stance) -> WorldPose:
        return body(st.body, st.heading, st.x, st.z, st.base, self.size)

    def root_at(self, t: float) -> Floor:
        """(x, z) in metres of the body's root at scenario time t."""
        i = self.segment_index(t)
        p = self.plans[i]
        tau = min(t - p.t_start, p.segment.duration)
        if p.walk is not None:
            return p.walk.at(p.walk.distance(t - p.walk.t0)[0])[0]
        if p.keys is not None and i > 0:
            st = self.stance(i, tau)
            return st.x, st.z
        k = self._progress(i, tau)
        return p.start[0] + (p.end[0] - p.start[0]) * k, p.start[1] + (p.end[1] - p.start[1]) * k

    def position_at(self, t: float) -> tuple[float, float]:
        """(frame x, depth) of the body's root at scenario time t."""
        x, z = self.root_at(t)
        depth = frame_depth(z)
        return frame_x(self.scene, x, depth), depth

    def pose_at(self, i: int, tau: float) -> ImagePose:
        """Ideal (noise-free) pose during segment i at local time tau, in the camera frame."""
        return to_image(self.world_pose_at(i, tau), self.scene)

    def world_pose_at(self, i: int, tau: float) -> WorldPose:
        """The same body as pose_at, in metres in the room (see hearth.sim.world)."""
        return self._world_pose(self.stance(i, tau))

    def heading_at(self, i: int, tau: float) -> float:
        """Which way the body faces (radians)."""
        return self.stance(i, tau).heading

    def state_at(self, t: float) -> PersonState:
        i = self.segment_index(t)
        st = self.stance(i, t - self.plans[i].t_start)
        world = self._world_pose(st)
        return PersonState(self, self.plans[i], to_image(world, self.scene), frame_depth(st.z), world, st.heading)

    def routes(self) -> list[WorldRoute]:
        """Each walk as one route on the floor."""
        out: list[WorldRoute] = []
        for p in self.plans:
            w = p.walk
            if w is None or (out and out[-1].t_start == round(w.t0, 6)):
                continue
            out.append(WorldRoute(person_id=self.person_id, t_start=round(w.t0, 6), t_end=round(w.t0 + w.duration, 6),
                                  points=[(round(x, 3), round(z, 3)) for x, z in w.points]))  # fmt: skip
        return out

    def world_at(self, t: float) -> WorldPerson:
        i = self.segment_index(t)
        st = self.stance(i, t - self.plans[i].t_start)
        return world_person(self.person_id, st.heading, self._world_pose(st), self.size)


def _head_forward(pose: BodyPose) -> bool:
    """Is the head ahead of the hips (along f)?"""
    return pose["nose"][0] >= hips_of(pose)[0]


def _flat(pose: BodyPose) -> bool:
    """Is the body lying along the floor?"""
    neck, hips = pose["neck"], hips_of(pose)
    return abs(neck[1] - hips[1]) < abs(neck[0] - hips[0])


def _pose_name(pose: BodyPose) -> PoseName | None:
    return next((name for name, p in KEY_POSES.items() if p is pose), None)  # type: ignore[return-value]


class ObservationGenerator:
    def __init__(
        self,
        scenario: Scenario,
        seed: int | None = None,
        *,
        start: datetime = SIM_EPOCH,
        noise: NoiseConfig | None = None,
        source: str = SOURCE,
    ) -> None:
        self.seed = scenario.seed if seed is None else seed
        self.scenario = scenario.realize(self.seed)  # timings for this seed; ground truth uses the same
        self.start = start
        self.noise = noise or NoiseConfig()
        self.source = source
        self.tracks = [ResidentTrack(r, self.scenario.scene) for r in self.scenario.cast()]

    def frame_times(self) -> Iterator[tuple[int, float]]:
        """(frame index, seconds from start) for every frame slot."""
        fps, total = self.scenario.fps, self.scenario.duration
        n = 0
        while (t := n / fps) < total - 1e-9:
            yield n, t
            n += 1

    def people_at(self, t: float) -> list[PersonState]:
        """Every resident's noise-free state at scenario time t (truth, not observation)."""
        return [track.state_at(t) for track in self.tracks]

    def world_track(self) -> WorldTrack:
        """Every resident's true body in 3D at every frame slot (simulator-only, see hearth.sim.world).

        Absent residents are left out. Everyone else stays in, seen by the camera or not.
        """
        sc = self.scenario
        frames = []
        for _, t in self.frame_times():
            people = [track.world_at(t) for track in self.tracks if track.state_at(t).label != "absent"]
            frames.append(WorldFrame(t=round(t, 6), people=people))
        return WorldTrack(scenario_id=sc.scenario_id, seed=self.seed, fps=sc.fps, duration=sc.duration,
                          room=room_geometry(sc.scene), frames=frames,
                          routes=[r for track in self.tracks for r in track.routes()])  # fmt: skip

    def stream(self) -> Iterator[Observation | Heartbeat]:
        """Every Observation and Heartbeat in timestamp order."""
        rng = random.Random(self.seed)
        next_heartbeat = 0.0
        self._next_face_check = 0.0
        self._last_seen: float | None = None  # single-person: when someone was last made out
        trackers: dict[str | None, FeatureTracker] = {}  # person -> their feature state, as the edge keeps it

        for n, t in self.frame_times():
            people = self.people_at(t)
            if any(p.label == "camera_outage" for p in people):
                # Silence: no observations, no heartbeats. Resume the heartbeat cadence afterwards.
                next_heartbeat = t
                trackers.clear()
                continue
            if t >= next_heartbeat - 1e-9:
                yield self._heartbeat(t)
                next_heartbeat += 1.0
            if rng.random() < self.noise.frame_drop:
                continue
            yield from self._frame(rng, n, t, people, trackers)

    def _frame(
        self, rng: random.Random, n: int, t: float, people: list[PersonState],
        trackers: dict[str | None, FeatureTracker],
    ) -> Iterator[Observation]:  # fmt: skip
        face_check = self._face_check_due(t)
        if not self.scenario.multi_person:
            obs = self._observation(rng, n, t, people[0], [], self._tracker(trackers, None), face_check)
            if obs is not None:
                self._last_seen = t
            elif (self.noise.dropout_bridge > 0 and self._last_seen is not None
                  and t - self._last_seen <= self.noise.dropout_bridge):
                return  # the edge doesn't report a brief dropout as "no person"
            yield obs or self._nobody(n, t, face_check)
            return

        present = [p for p in people if p.label != "absent"]
        emitted = False
        for person in people:  # cast order keeps random draws stable
            if person.label == "absent":
                continue
            in_front = [_body_box(q.pose) for q in present if q.depth < person.depth - 0.05]
            obs = self._observation(rng, n, t, person, in_front, self._tracker(trackers, person.track.person_id),
                                    face_check)
            if obs is None:
                continue
            emitted = True
            yield obs
        if not emitted:
            yield self._nobody(n, t, face_check)

    def _face_check_due(self, t: float) -> bool:
        if self.noise.face_check_interval <= 0:
            return True
        if t >= self._next_face_check - 1e-9:
            self._next_face_check = t + self.noise.face_check_interval
            return True
        return False

    def _tracker(self, trackers: dict[str | None, FeatureTracker], person_id: str | None) -> FeatureTracker:
        if person_id not in trackers:
            trackers[person_id] = FeatureTracker(self.scenario.scene.frame_aspect)
        return trackers[person_id]

    def _timestamp(self, t: float) -> datetime:
        return self.start + timedelta(seconds=round(t, 6))

    def _heartbeat(self, t: float) -> Heartbeat:
        return Heartbeat(
            timestamp=self._timestamp(t),
            source=self.source,
            source_type="simulation",
            room_id=self.scenario.room_id,
            fps=self.scenario.fps,
        )

    def _base(self, n: int, t: float, person_id: str | None) -> dict:
        suffix = f"-{person_id}" if person_id else ""
        return {
            "obs_id": f"sim-{self.scenario.scenario_id}-s{self.seed}-{n:06d}{suffix}",
            "timestamp": self._timestamp(t),
            "source": self.source,
            "source_type": "simulation",
            "room_id": self.scenario.room_id,
            "scenario_id": self.scenario.scenario_id,
            "person_id": person_id,
            "person_detected": False,
            "face_present": False,
        }

    def _nobody(self, n: int, t: float, face_check: bool = True) -> Observation:
        obs = self._base(n, t, None)
        if not face_check:
            obs["face_present"] = None
        return Observation.model_validate(obs)

    def _observation(
        self, rng: random.Random, n: int, t: float, person: PersonState, in_front: list[Box],
        tracker: FeatureTracker,
        face_check: bool = True,
    ) -> Observation | None:  # fmt: skip
        """One person's observation, or None if the camera can't make them out."""
        if person.label == "absent":
            tracker.lost()
            return None
        keypoints = self._keypoints(rng, person, in_front)
        if len(keypoints) < MIN_VISIBLE_KEYPOINTS:
            tracker.lost()
            return None

        features = tracker.update(keypoints, t)
        face_present, face_conf = self._face(rng, keypoints, person)
        if not face_check:
            face_present = face_conf = None  # no face check ran on this frame
        if not self.noise.keypoint_confidence:
            # poseNet reports the joints it detected, with no score.
            keypoints = [dict(k, confidence=None) for k in keypoints if k["confidence"] >= MIN_CONFIDENCE]
        obs = self._base(n, t, person.track.person_id)
        obs.update(
            person_detected=True,
            keypoints=keypoints,
            features=features,
            face_present=face_present,
            face_confidence=face_conf,
        )
        return Observation.model_validate(obs)

    def _keypoints(self, rng: random.Random, person: PersonState, in_front: list[Box]) -> list[dict]:
        nz = self.noise
        plan = person.plan
        lying = person.lying
        jitter = nz.jitter * (1.5 if lying else 1.0)
        conf_mean = nz.confidence_mean - (nz.lying_penalty if lying else 0.0)
        occluder_top = plan.segment.occluder_top if plan.segment.label == "occluded" else None

        out = []
        for joint in JOINTS:
            x, y = person.pose[joint]
            # Draw every random number for every joint, visible or not, so one hidden joint
            # doesn't shift the noise of all the others.
            dx, dy = rng.gauss(0.0, jitter), rng.gauss(0.0, jitter)
            conf = rng.gauss(conf_mean, nz.confidence_sd)
            dropped = rng.random() < nz.keypoint_dropout
            if any(bx0 <= x <= bx1 and by0 <= y <= by1 for bx0, by0, bx1, by1 in in_front):
                continue  # behind someone nearer the camera
            x, y = x + dx, y + dy
            if dropped or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue
            if occluder_top is not None and y > occluder_top:
                continue
            out.append({"name": joint, "x": round(x, 4), "y": round(y, 4),
                        "confidence": round(min(max(conf, 0.05), 0.99), 3)})  # fmt: skip
        return out

    def _face(self, rng: random.Random, keypoints: list[dict], person: PersonState) -> tuple[bool, float | None]:
        nz = self.noise
        facing = _nearest_facing(person.heading)
        if person.lying:
            rate = nz.face_rate_lying
        elif facing == "camera":
            rate = nz.face_rate_front if person.plan.pose != "bending" else nz.face_rate_side
        elif facing == "away":
            rate = nz.face_rate_away
        else:
            rate = nz.face_rate_side
        roll, conf = rng.random(), rng.uniform(0.6, 0.95)
        if not any(k["name"] == "nose" for k in keypoints) or roll >= rate:
            return False, None
        return True, round(conf, 3)


def _nearest_facing(heading: float) -> str:
    return min(HEADING, key=lambda name: abs(_turn(HEADING[name], heading)))


def _body_box(pose: ImagePose) -> Box:
    """The region a person's body covers in the frame (padded keypoint bounding box)."""
    xs = [x for x, _ in pose.values()]
    ys = [y for _, y in pose.values()]
    return (min(xs) - BOX_PADDING, min(ys) - BOX_PADDING, max(xs) + BOX_PADDING, max(ys) + BOX_PADDING)

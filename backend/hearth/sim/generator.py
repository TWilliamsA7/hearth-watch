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
import math
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from hearth_schema.models import Heartbeat, Observation

from hearth.sim.features import compute_features
from hearth.sim.pose import (
    JOINTS,
    SIDE_ONLY_POSES,
    STEP_HZ,
    BodyPose,
    ImagePose,
    blend,
    body_pose,
    ease_in,
    project,
    smoothstep,
)
from hearth.sim.scenario import Facing, PoseName, Resident, Scenario, Scene, Segment
from hearth.sim.world import (
    HEADING,
    WorldFrame,
    WorldPerson,
    WorldPose,
    WorldTrack,
    blend_world,
    place,
    room_geometry,
    world_person,
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

MIN_VISIBLE_KEYPOINTS = 4

BOX_PADDING = 0.01
"""How far around a nearer person's keypoints their body hides people behind them."""

Box = tuple[float, float, float, float]  # x0, y0, x1, y1


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

    @classmethod
    def off(cls) -> NoiseConfig:
        return cls(jitter=0.0, confidence_sd=0.0, keypoint_dropout=0.0, frame_drop=0.0, lying_penalty=0.0,
                   face_rate_front=1.0, face_rate_side=0.0, face_rate_lying=0.0)  # fmt: skip


@dataclass(frozen=True)
class _Plan:
    """Where a segment starts and what it is heading for, resolved once per run."""

    segment: Segment
    t_start: float
    pose: PoseName
    facing: Facing
    x_start: float
    x_end: float
    depth_start: float
    depth_end: float


@dataclass(frozen=True)
class _Placement:
    """Everything needed to put a body in either space: key pose, facing and position."""

    body: BodyPose
    facing: Facing
    x: float
    base_y: float
    depth: float
    height: float


@dataclass(frozen=True)
class _Space:
    """A way of turning a placement into joint positions: the camera frame or the room."""

    name: str
    project: Callable[[_Placement, Scene], dict]
    blend: Callable[[dict, dict, float], dict]


_IMAGE = _Space("image", lambda p, sc: project(p.body, p.facing, p.x, p.base_y, sc, p.height), blend)
_WORLD = _Space("world", lambda p, sc: place(p.body, p.facing, p.x, p.base_y, p.depth, sc, p.height), blend_world)


@dataclass(frozen=True)
class PersonState:
    """One resident at one instant: the noise-free truth of where their body is."""

    track: ResidentTrack
    plan: _Plan
    pose: ImagePose
    depth: float

    @property
    def label(self) -> str:
        return self.plan.segment.label


class ResidentTrack:
    """One resident's motion on the scenario clock. Pure geometry, no randomness."""

    def __init__(self, resident: Resident, scene: Scene) -> None:
        self.resident = resident
        self.scene = scene
        self.plans = self._plan_segments()
        self._starts = [p.t_start for p in self.plans]
        self._start_poses: dict[tuple[str, int], dict] = {}

    @property
    def person_id(self) -> str | None:
        return self.resident.resident_id

    def _plan_segments(self) -> list[_Plan]:
        res = self.resident
        plans: list[_Plan] = []
        pose: PoseName = "standing"
        facing: Facing = res.start_facing
        x, depth = res.start_x, res.start_depth
        for seg, (t0, _) in zip(res.segments, res.segment_times()):
            pose = seg.pose or LABEL_POSE.get(seg.label, pose)
            x_end = x if seg.x is None else seg.x
            depth_end = depth if seg.depth is None else seg.depth
            if seg.facing is not None:
                facing = seg.facing
            elif seg.label == "walking" and x_end != x:
                facing = "right" if x_end > x else "left"
            if pose in SIDE_ONLY_POSES and facing == "camera":
                facing = "right"
            plans.append(_Plan(seg, t0, pose, facing, x, x_end, depth, depth_end))
            x, depth = x_end, depth_end
        return plans

    def segment_index(self, t: float) -> int:
        return max(0, bisect.bisect_right(self._starts, t + 1e-9) - 1)

    def _progress(self, i: int, tau: float) -> float:
        """How far segment i has moved towards its target position (0..1)."""
        seg = self.plans[i].segment
        if seg.label == "walking":
            return min(tau / seg.duration, 1.0)
        return smoothstep(tau / seg.transition if seg.transition else 1.0)

    def depth_at(self, i: int, tau: float) -> float:
        p = self.plans[i]
        return p.depth_start + (p.depth_end - p.depth_start) * self._progress(i, tau)

    def _placement(self, i: int, tau: float) -> _Placement:
        """Where segment i's motion puts the body at local time tau, before blending in."""
        p = self.plans[i]
        seg = p.segment
        k = self._progress(i, tau)
        x = p.x_start + (p.x_end - p.x_start) * k
        depth = self.depth_at(i, tau)
        gait = 2.0 * math.pi * STEP_HZ * tau if seg.label == "walking" else None
        base_y = self.scene.floor_at(depth) if seg.surface_y is None else seg.surface_y
        height = self.scene.height_at(depth, self.resident.height)
        return _Placement(body_pose(p.pose, gait), p.facing, x, base_y, depth, height)

    def _start_pose(self, i: int, space: _Space) -> dict:
        key = (space.name, i)
        if key not in self._start_poses:
            if i == 0:
                self._start_poses[key] = space.project(self._placement(0, 0.0), self.scene)
            else:
                self._start_poses[key] = self._pose_in(i - 1, self.plans[i - 1].segment.duration, space)
        return self._start_poses[key]

    def _pose_in(self, i: int, tau: float, space: _Space) -> dict:
        seg = self.plans[i].segment
        tau = min(tau, seg.duration)  # after their script ends, a resident holds their last pose
        target = space.project(self._placement(i, tau), self.scene)
        if i == 0 or seg.transition == 0.0 or tau >= seg.transition:
            return target
        return space.blend(self._start_pose(i, space), target, self._ease(i, tau))

    def _ease(self, i: int, tau: float) -> float:
        seg = self.plans[i].segment
        return (ease_in if seg.label == "fall" else smoothstep)(tau / seg.transition)

    def pose_at(self, i: int, tau: float) -> ImagePose:
        """Ideal (noise-free) pose during segment i at local time tau, in the camera frame."""
        return self._pose_in(i, tau, _IMAGE)

    def world_pose_at(self, i: int, tau: float) -> WorldPose:
        """The same body as pose_at, in metres in the room (see hearth.sim.world)."""
        return self._pose_in(i, tau, _WORLD)

    def heading_at(self, i: int, tau: float) -> float:
        """Which way the body faces (radians), turning through each transition with the pose."""
        seg = self.plans[i].segment
        target = HEADING[self.plans[i].facing]
        if i == 0 or seg.transition == 0.0 or tau >= seg.transition:
            return target
        start = self.heading_at(i - 1, self.plans[i - 1].segment.duration)
        turn = (target - start + math.pi) % (2.0 * math.pi) - math.pi  # the short way round
        return start + turn * self._ease(i, tau)

    def state_at(self, t: float) -> PersonState:
        i = self.segment_index(t)
        tau = t - self.plans[i].t_start
        return PersonState(self, self.plans[i], self.pose_at(i, tau), self.depth_at(i, min(tau, self.plans[i].segment.duration)))

    def world_at(self, t: float) -> WorldPerson:
        i = self.segment_index(t)
        tau = min(t - self.plans[i].t_start, self.plans[i].segment.duration)
        return world_person(self.person_id, self.heading_at(i, tau), self.world_pose_at(i, tau))


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
                          room=room_geometry(sc.scene), frames=frames)  # fmt: skip

    def stream(self) -> Iterator[Observation | Heartbeat]:
        """Every Observation and Heartbeat in timestamp order."""
        rng = random.Random(self.seed)
        next_heartbeat = 0.0
        prev_hip: dict[str | None, tuple[float, float]] = {}  # person -> (t, hip_y) of their last hips

        for n, t in self.frame_times():
            people = self.people_at(t)
            if any(p.label == "camera_outage" for p in people):
                # Silence: no observations, no heartbeats. Resume the heartbeat cadence afterwards.
                next_heartbeat = t
                prev_hip.clear()
                continue
            if t >= next_heartbeat - 1e-9:
                yield self._heartbeat(t)
                next_heartbeat += 1.0
            if rng.random() < self.noise.frame_drop:
                continue
            yield from self._frame(rng, n, t, people, prev_hip)

    def _frame(
        self, rng: random.Random, n: int, t: float, people: list[PersonState],
        prev_hip: dict[str | None, tuple[float, float]],
    ) -> Iterator[Observation]:  # fmt: skip
        if not self.scenario.multi_person:
            obs = self._observation(rng, n, t, people[0], [], prev_hip.get(None)) or self._nobody(n, t)
            if obs.features is not None and obs.features.hip_y is not None:
                prev_hip[None] = (t, obs.features.hip_y)
            yield obs
            return

        present = [p for p in people if p.label != "absent"]
        emitted = False
        for person in people:  # cast order keeps random draws stable
            if person.label == "absent":
                continue
            in_front = [_body_box(q.pose) for q in present if q.depth < person.depth - 0.05]
            obs = self._observation(rng, n, t, person, in_front, prev_hip.get(person.track.person_id))
            if obs is None:
                continue
            if obs.features is not None and obs.features.hip_y is not None:
                prev_hip[person.track.person_id] = (t, obs.features.hip_y)
            emitted = True
            yield obs
        if not emitted:
            yield self._nobody(n, t)

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

    def _nobody(self, n: int, t: float) -> Observation:
        return Observation.model_validate(self._base(n, t, None))

    def _observation(
        self, rng: random.Random, n: int, t: float, person: PersonState, in_front: list[Box],
        prev_hip: tuple[float, float] | None,
    ) -> Observation | None:  # fmt: skip
        """One person's observation, or None if the camera can't make them out."""
        if person.label == "absent":
            return None
        keypoints = self._keypoints(rng, person, in_front)
        if len(keypoints) < MIN_VISIBLE_KEYPOINTS:
            return None

        prev_y, dt = (prev_hip[1], t - prev_hip[0]) if prev_hip else (None, None)
        features = compute_features(keypoints, self.scenario.scene.frame_aspect, prev_y, dt)
        face_present, face_conf = self._face(rng, keypoints, person.plan)
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
        lying = plan.pose in SIDE_ONLY_POSES
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

    def _face(self, rng: random.Random, keypoints: list[dict], plan: _Plan) -> tuple[bool, float | None]:
        nz = self.noise
        if plan.pose in SIDE_ONLY_POSES:
            rate = nz.face_rate_lying
        elif plan.facing == "camera":
            rate = nz.face_rate_front if plan.pose != "bending" else nz.face_rate_side
        else:
            rate = nz.face_rate_side
        roll, conf = rng.random(), rng.uniform(0.6, 0.95)
        if not any(k["name"] == "nose" for k in keypoints) or roll >= rate:
            return False, None
        return True, round(conf, 3)


def _body_box(pose: ImagePose) -> Box:
    """The region a person's body covers in the frame (padded keypoint bounding box)."""
    xs = [x for x, _ in pose.values()]
    ys = [y for _, y in pose.values()]
    return (min(xs) - BOX_PADDING, min(ys) - BOX_PADDING, max(xs) + BOX_PADDING, max(ys) + BOX_PADDING)

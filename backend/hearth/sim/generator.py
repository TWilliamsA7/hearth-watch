"""Observation generator: turns a scenario + seed into what a camera would have reported.

Output is Observations and Heartbeats only, built as plain dicts the way the edge client
builds them and validated against the shared contract. Ground truth is produced
separately by Scenario.ground_truth() and never travels in this stream.

Determinism: one random.Random(seed) per run, consumed in a fixed order, and timestamps
from a virtual clock. The same scenario and seed always produce identical output.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from hearth_schema.models import Heartbeat, Observation

from hearth.sim.features import compute_features
from hearth.sim.pose import (
    JOINTS,
    SIDE_ONLY_POSES,
    STEP_HZ,
    ImagePose,
    blend,
    body_pose,
    ease_in,
    project,
    smoothstep,
)
from hearth.sim.scenario import Facing, PoseName, Scenario, Segment

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
    base_y: float


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
        self.scenario = scenario
        self.seed = scenario.seed if seed is None else seed
        self.start = start
        self.noise = noise or NoiseConfig()
        self.source = source
        self._plans = self._plan_segments()
        self._start_poses: dict[int, ImagePose] = {}

    # --- scene geometry, no randomness ---

    def _plan_segments(self) -> list[_Plan]:
        sc = self.scenario
        plans: list[_Plan] = []
        pose: PoseName = "standing"
        facing: Facing = sc.start_facing
        x = sc.start_x
        for seg, (t0, _) in zip(sc.segments, sc.segment_times()):
            pose = seg.pose or LABEL_POSE.get(seg.label, pose)
            x_end = x if seg.x is None else seg.x
            if seg.facing is not None:
                facing = seg.facing
            elif seg.label == "walking" and x_end != x:
                facing = "right" if x_end > x else "left"
            if pose in SIDE_ONLY_POSES and facing == "camera":
                facing = "right"
            base_y = sc.scene.floor_y if seg.surface_y is None else seg.surface_y
            plans.append(_Plan(seg, t0, pose, facing, x, x_end, base_y))
            x = x_end
        return plans

    def _target_pose(self, i: int, tau: float) -> ImagePose:
        """Where segment i's motion puts the body at local time tau, before blending in."""
        p = self._plans[i]
        seg = p.segment
        if seg.label == "walking":
            x = p.x_start + (p.x_end - p.x_start) * (tau / seg.duration)
            gait = 2.0 * math.pi * STEP_HZ * tau
        else:
            x = p.x_start + (p.x_end - p.x_start) * smoothstep(tau / seg.transition if seg.transition else 1.0)
            gait = None
        return project(body_pose(p.pose, gait), p.facing, x, p.base_y, self.scenario.scene)

    def _start_pose(self, i: int) -> ImagePose:
        if i not in self._start_poses:
            if i == 0:
                self._start_poses[i] = self._target_pose(0, 0.0)
            else:
                prev = self._plans[i - 1].segment
                self._start_poses[i] = self.pose_at(i - 1, prev.duration)
        return self._start_poses[i]

    def pose_at(self, i: int, tau: float) -> ImagePose:
        """Ideal (noise-free) pose during segment i at local time tau."""
        seg = self._plans[i].segment
        target = self._target_pose(i, tau)
        if i == 0 or seg.transition == 0.0 or tau >= seg.transition:
            return target
        ease = ease_in if seg.label == "fall" else smoothstep
        return blend(self._start_pose(i), target, ease(tau / seg.transition))

    # --- the stream ---

    def frame_times(self) -> Iterator[tuple[int, float, int]]:
        """(frame index, seconds from start, segment index) for every frame slot."""
        fps, total = self.scenario.fps, self.scenario.duration
        i, n = 0, 0
        while (t := n / fps) < total - 1e-9:
            while i + 1 < len(self._plans) and t >= self._plans[i + 1].t_start - 1e-9:
                i += 1
            yield n, t, i
            n += 1

    def stream(self) -> Iterator[Observation | Heartbeat]:
        """Every Observation and Heartbeat in timestamp order."""
        rng = random.Random(self.seed)
        next_heartbeat = 0.0
        prev_hip: tuple[float, float] | None = None  # (t, hip_y) of the last observation with hips

        for n, t, i in self.frame_times():
            plan = self._plans[i]
            if plan.segment.label == "camera_outage":
                # Silence: no observations, no heartbeats. Resume the heartbeat cadence afterwards.
                next_heartbeat = t
                prev_hip = None
                continue
            if t >= next_heartbeat - 1e-9:
                yield self._heartbeat(t)
                next_heartbeat += 1.0
            if rng.random() < self.noise.frame_drop:
                continue
            obs = self._observation(rng, n, t, i, prev_hip)
            if obs.features is not None and obs.features.hip_y is not None:
                prev_hip = (t, obs.features.hip_y)
            yield obs

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

    def _observation(
        self, rng: random.Random, n: int, t: float, i: int, prev_hip: tuple[float, float] | None
    ) -> Observation:
        plan = self._plans[i]
        seg = plan.segment
        obs = {
            "obs_id": f"sim-{self.scenario.scenario_id}-s{self.seed}-{n:06d}",
            "timestamp": self._timestamp(t),
            "source": self.source,
            "source_type": "simulation",
            "room_id": self.scenario.room_id,
            "scenario_id": self.scenario.scenario_id,
            "person_detected": False,
            "face_present": False,
        }
        if seg.label == "absent":
            return Observation.model_validate(obs)

        keypoints = self._keypoints(rng, self.pose_at(i, t - plan.t_start), plan)
        if len(keypoints) < MIN_VISIBLE_KEYPOINTS:
            return Observation.model_validate(obs)

        prev_y, dt = (prev_hip[1], t - prev_hip[0]) if prev_hip else (None, None)
        features = compute_features(keypoints, self.scenario.scene.frame_aspect, prev_y, dt)
        face_present, face_conf = self._face(rng, keypoints, plan)
        obs.update(
            person_detected=True,
            keypoints=keypoints,
            features=features,
            face_present=face_present,
            face_confidence=face_conf,
        )
        return Observation.model_validate(obs)

    def _keypoints(self, rng: random.Random, pose: ImagePose, plan: _Plan) -> list[dict]:
        nz = self.noise
        lying = plan.pose in SIDE_ONLY_POSES
        jitter = nz.jitter * (1.5 if lying else 1.0)
        conf_mean = nz.confidence_mean - (nz.lying_penalty if lying else 0.0)
        occluder_top = plan.segment.occluder_top if plan.segment.label == "occluded" else None

        out = []
        for joint in JOINTS:
            x, y = pose[joint]
            # Draw every random number for every joint, visible or not, so one hidden joint
            # doesn't shift the noise of all the others.
            dx, dy = rng.gauss(0.0, jitter), rng.gauss(0.0, jitter)
            conf = rng.gauss(conf_mean, nz.confidence_sd)
            dropped = rng.random() < nz.keypoint_dropout
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

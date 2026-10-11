"""Body poses for the live engine: gait locked to speed, posture blends, motions and idles.

Everything here builds on hearth.sim.pose and hearth.sim.motion, so a live body is posed
exactly as a scripted one is: bone-by-bone blends that keep limb lengths, planted feet
that stay put while someone sits, and the same choreographed falls and get-ups.

What's new is the idle layer: people who stand, sit or talk are never perfectly still.
They breathe, sway a little, shift their weight, turn their head towards whoever they
are talking to or something in the room, and gesture while talking. These are small
rotations of whole limbs (pose.turn_limb / lean), so bone lengths never change.
"""

from __future__ import annotations

import math

from hearth.sim.generator import Stance, blend_stance
from hearth.sim.motion import Key, timeline
from hearth.sim.pose import (
    F_AXIS,
    HEAD_JOINTS,
    KEY_POSES,
    L_AXIS,
    V_AXIS,
    BodyPose,
    body_pose,
    gait_amount,
    hips_of,
    lean,
    rotate,
    smoothstep,
    stride_length,
    turn_limb,
)
from hearth.sim.world import WorldPose, body

from hearth.sim.live.actor import Actor

STILL_SPEED = 0.15  # m/s: below this the gait fades out, so standing still looks still
MAX_HEAD_YAW = math.radians(65)
HEAD_TURN_RATE = 2.5  # rad/s


def wrap(a: float) -> float:
    """An angle in (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def heading_to(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])


def gait(actor: Actor, speed: float, dt: float, *, stepping: float = 0.0, backwards: bool = False,
         stride: float = 1.0) -> BodyPose:  # fmt: skip
    """A standing body walking at `speed` m/s. The cycle advances with distance covered,
    so the planted foot keeps pace with the floor (and runs the other way when stepping
    backwards); `stepping` adds a little stepping on the spot (turning in place). `stride`
    scales the step length: under 1, shorter quicker steps (a shuffle)."""
    size = actor.size
    stride = stride_length(speed, size) * stride
    ahead = -1.0 if backwards else 1.0
    actor.gait_phase = (actor.gait_phase + 2.0 * math.pi * (ahead * speed * dt / stride + stepping * dt * 0.9)) % (2 * math.pi)
    fade = min(1.0, max(speed / STILL_SPEED, stepping * 1.5))
    if fade <= 0.01:
        return dict(KEY_POSES["standing"])
    return body_pose("standing", actor.gait_phase, gait_amount(stride, size) * smoothstep(fade))


def idle(actor: Actor, pose: BodyPose, t: float, dt: float, *, sway: float = 1.0, talking: bool = False) -> BodyPose:
    """Layer breathing, sway, head turns and talking gestures over a held pose."""
    ph = actor.idle_phase
    out = dict(pose)
    upright = actor.posture in ("standing", "sitting")
    # Breathing: the chest and head rise and fall a few millimetres.
    breath = 0.0035 * math.sin(2 * math.pi * t / 4.3 + ph)
    for j in ("neck", "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist"):  # fmt: skip
        f, v, l = out[j]
        out[j] = (f, v + breath, l)
    if upright and sway > 0:
        side = math.radians(1.3) * sway * math.sin(2 * math.pi * t / 7.1 + ph)
        fore = math.radians(0.9) * sway * math.sin(2 * math.pi * t / 5.3 + 2 * ph)
        out = lean(lean(out, F_AXIS, side), L_AXIS, fore)
    # Head: towards whoever or whatever they're looking at, within what a neck allows.
    want = 0.0
    if actor.look_at is not None and actor.posture != "absent":
        rel = wrap(heading_to(actor.pos, actor.look_at) - actor.heading)
        want = max(-MAX_HEAD_YAW, min(MAX_HEAD_YAW, rel))
    elif upright and sway > 0:
        want = math.radians(25) * math.sin(2 * math.pi * t / 13.0 + 3 * ph) ** 3  # glancing about
    step = HEAD_TURN_RATE * dt
    actor.look_yaw += max(-step, min(step, want - actor.look_yaw))
    if abs(actor.look_yaw) > 1e-3 and actor.posture != "floor":
        out = turn_head(out, actor.look_yaw)
    # Talking: a forearm comes up now and then, as people gesture.
    target = 0.0
    if talking:
        beat = math.sin(2 * math.pi * t / 3.7 + ph) + 0.6 * math.sin(2 * math.pi * t / 1.9 + 2 * ph)
        target = max(0.0, min(1.0, beat - 0.3))
    actor.gesture += (target - actor.gesture) * min(1.0, dt * 3.0)
    if actor.gesture > 0.01:
        side = "right" if ph % 2 < 1 else "left"
        g = actor.gesture
        out = turn_limb(out, f"{side}_shoulder", L_AXIS, -math.radians(18) * g)
        out = turn_limb(out, f"{side}_elbow", L_AXIS, -math.radians(65) * g)
    return out


def turn_head(pose: BodyPose, yaw: float) -> BodyPose:
    """Turn the head on the neck (about the body's own up axis). Not turn_limb("neck"): that
    swings the shoulders and arms round with it."""
    neck = pose["neck"]
    return {j: rotate(p, V_AXIS, yaw, neck) if j in HEAD_JOINTS else p for j, p in pose.items()}


def stance(actor: Actor) -> Stance:
    return Stance(actor.body, actor.heading, actor.pos[0], actor.pos[1], actor.base)


def apply(actor: Actor, st: Stance) -> None:
    actor.body, actor.heading, actor.pos, actor.base = st.body, st.heading, (st.x, st.z), st.base


def world_pose(actor: Actor) -> WorldPose:
    joints = getattr(actor, "joints", None)
    if joints:  # an ExternalActor: the live person's lifted joints
        return dict(joints)
    return body(actor.body, actor.heading, actor.pos[0], actor.pos[1], actor.base, actor.size)


class Blend:
    """From the body as it is now to a target stance over `duration` seconds."""

    def __init__(self, start: Stance, end: Stance, duration: float, ease=None) -> None:
        from hearth.sim.pose import smoothstep

        self.start, self.end, self.duration = start, end, max(duration, 1e-3)
        self.ease = ease or smoothstep

    def at(self, tau: float) -> Stance:
        return blend_stance(self.start, self.end, self.ease(min(tau / self.duration, 1.0)))

    def done(self, tau: float) -> bool:
        return tau >= self.duration


def flat(pose: BodyPose) -> bool:
    """Is the body lying along the floor?"""
    neck, hips = pose["neck"], hips_of(pose)
    return abs(neck[1] - hips[1]) < abs(neck[0] - hips[0])


class Play:
    """A choreographed motion (hearth.sim.motion), from the body as it is now.

    end: where the root finishes (x, z); heading: where the body turns to; base: the
    surface height at the end.
    """

    def __init__(self, start: Stance, motion: str, variant: int, end: tuple[float, float], heading: float,
                 base: float = 0.0, pace: float = 1.0) -> None:  # fmt: skip
        self.keys: list[Key] = timeline(motion, variant, pace=pace)
        if flat(start.body) and (start.body["nose"][0] >= hips_of(start.body)[0]) != (
            self.keys[0].pose["nose"][0] >= hips_of(self.keys[0].pose)[0]
        ):
            start = start.flipped()  # lying the other way round: describe it from the other end
        self.start, self.end, self.base = start, end, base
        self.turn = wrap(heading - start.heading)
        self.duration = self.keys[-1].at

    def _key(self, key: Key) -> Stance:
        s = self.start
        return Stance(key.pose, s.heading + self.turn * key.turn,
                      s.x + (self.end[0] - s.x) * key.move, s.z + (self.end[1] - s.z) * key.move,
                      s.base + (self.base - s.base) * key.move)  # fmt: skip

    def at(self, tau: float) -> Stance:
        prev_t, prev = 0.0, self.start
        for key in self.keys:
            if tau < key.at:
                span = key.at - prev_t
                return blend_stance(prev, self._key(key), key.ease((tau - prev_t) / span) if span > 0 else 1.0)
            prev_t, prev = key.at, self._key(key)
        return prev

    def done(self, tau: float) -> bool:
        return tau >= self.duration

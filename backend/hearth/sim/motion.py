"""Choreographed movements: falls, stumbles and getting up, as timelines of key poses.

A plain segment blends once from wherever the body was into the label's pose. That is
fine for sitting down, but a fall is not one movement: someone who trips pitches
forward, throws their arms out, lands on hands and knees and ends up face down. A
Segment with a `motion` plays a timeline of keys instead, all inside that one segment,
so a fall is still exactly one ground-truth record and one expected alert.

Each key says when it is reached (seconds from the segment start), the pose, how the
body accelerates into it (ease), how far the body has travelled from the segment's start
point towards its end point (move) and how far it has turned towards the segment's
heading (turn). The first key is wherever the body was when the segment began.

`variant` seeds the details: which way the limbs end up, small differences in timing,
and for some motions how it ends (a sit_miss may stay sitting on the floor). The same
variant always gives the same movement.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from hearth.sim.pose import (
    F_AXIS,
    L_AXIS,
    KEY_POSES,
    BodyPose,
    ease_in,
    fit_bones,
    lean,
    lying_pose,
    smoothstep,
    tilt,
)

Ease = Callable[[float], float]

MIN_STEP_S = 0.15  # shortest time between two keys, before pace


def drop(x: float) -> float:
    """Speeding up as it goes, like ease_in, but gentler: limbs land, they don't snap."""
    x = min(max(x, 0.0), 1.0)
    return x * (0.6 + 0.4 * x)


@dataclass(frozen=True)
class Key:
    at: float  # seconds from the start of the segment
    pose: BodyPose
    ease: Ease = smoothstep
    move: float = 1.0  # fraction of the way from the start point to the end point
    turn: float = 1.0  # fraction of the turn towards the segment's heading


def _rng(motion: str, variant: int | None) -> random.Random | None:
    return None if variant is None else random.Random(f"motion/{motion}/{variant}")


def _p(name: str) -> BodyPose:
    return KEY_POSES[name]


def _roll(pose: BodyPose, angle: float) -> BodyPose:
    """Tip the whole body sideways about its feet."""
    return fit_bones(tilt(pose, F_AXIS, angle))


def _scaled(keys: list[Key], k: float) -> list[Key]:
    return [Key(key.at * k, key.pose, key.ease, key.move, key.turn) for key in keys]


def timeline(motion: str, variant: int | None = None, *, pace: float = 1.0, fit: float | None = None) -> list[Key]:
    """The keys of a motion. pace scales its timing (2.0 = twice as slow); fit, if given,
    squeezes the whole thing into that many seconds when it would take longer."""
    rng = _rng(motion, variant)

    def u(lo: float, hi: float) -> float:
        return (lo + hi) / 2 if rng is None else rng.uniform(lo, hi)

    def chance(p: float) -> bool:
        return rng is not None and rng.random() < p

    side = -1.0 if chance(0.5) else 1.0
    keys: list[Key]
    if motion == "collapse":
        # The plain fall hand-written scenarios get: knees go, then down onto the side.
        # pace is the segment's transition; going down takes a little longer than that.
        keys = [
            Key(0.45, _p("buckle"), smoothstep, 0.1, 0.0),
            Key(0.8, _p("kneel"), ease_in, 0.4, 0.3),
            Key(1.45, _p("fallen"), ease_in, 1.0, 1.0),
        ]
    elif motion == "trip":
        prone = lying_pose("prone", rng=rng)
        keys = [Key(u(0.2, 0.3), lean(_p("stumble"), L_AXIS, math.radians(u(0, 12))), smoothstep, 0.25, 0.3),
                Key(u(0.45, 0.55), _p("lunge"), ease_in, 0.5, 0.6)]  # fmt: skip
        if chance(0.6):  # catches themselves on hands and knees first
            keys.append(Key(u(0.75, 0.85), _p("hands_knees"), ease_in, 0.8, 0.9))
            keys.append(Key(u(1.3, 1.7), prone, smoothstep, 1.0, 1.0))
        else:  # straight down
            keys.append(Key(u(0.8, 0.95), prone, ease_in, 1.0, 1.0))
    elif motion == "faint":
        keys = [Key(u(0.35, 0.55), _p("buckle"), smoothstep, 0.1, 0.0)]
        if chance(0.7):  # knees give, slumps sideways off them
            keys.append(Key(u(0.75, 0.9), _p("kneel"), ease_in, 0.4, 0.2))
            keys.append(Key(u(1.25, 1.45), _roll(_p("kneel"), side * math.radians(u(25, 40))), drop, 0.6, 0.6))
            keys.append(Key(u(2.0, 2.4), lying_pose("side", rng=rng, roll=side * math.pi / 2), smoothstep, 1.0, 1.0))
        else:  # sits down hard and goes over backwards
            keys.append(Key(u(0.95, 1.15), _p("floor_sit"), drop, 0.6, 0.8))
            keys.append(Key(u(1.8, 2.2), lying_pose("sprawl", head=-1.0, rng=rng), drop, 1.0, 1.0))
    elif motion == "sit_miss":
        keys = [Key(u(0.4, 0.6), _p("crouch"), smoothstep, 0.3, 0.5),
                Key(u(0.85, 1.05), _p("floor_sit"), drop, 0.8, 1.0)]  # fmt: skip
        ending = rng.random() if rng is not None else 0.0
        if ending < 0.4:  # stays sitting on the floor, slumped a little
            keys.append(Key(u(1.4, 1.8), lean(_p("floor_sit"), L_AXIS, -math.radians(u(5, 20))), smoothstep, 1.0, 1.0))
        elif ending < 0.75:  # tips back
            keys.append(Key(u(1.8, 2.2), lying_pose("sprawl", head=-1.0, rng=rng), drop, 1.0, 1.0))
        else:  # rolls onto a side
            keys.append(Key(u(1.9, 2.4), lying_pose("side", head=-1.0, rng=rng, roll=side * math.pi / 2), drop, 1.0, 1.0))
    elif motion == "stand_up":
        keys = [Key(u(0.5, 0.8), _p("half_stand"), smoothstep, 0.3, 0.0),
                Key(u(0.95, 1.2), _p("lean_back"), smoothstep, 0.5, 0.2),
                Key(u(1.5, 1.75), _p("floor_sit"), drop, 0.8, 0.6)]  # fmt: skip
        if chance(0.6):
            keys.append(Key(u(2.1, 2.5), lying_pose("sprawl", head=-1.0, rng=rng), drop, 1.0, 1.0))
        else:
            keys.append(Key(u(2.2, 2.6), lying_pose("side", head=-1.0, rng=rng, roll=side * math.pi / 2), drop, 1.0, 1.0))
    elif motion == "collision":
        keys = [Key(u(0.2, 0.35), _roll(_p("lean_back"), side * math.radians(u(10, 20))), smoothstep, 0.3, 0.3),
                Key(u(0.55, 0.7), _roll(_p("buckle"), side * math.radians(u(20, 35))), ease_in, 0.6, 0.6),
                Key(u(0.95, 1.15), _roll(_p("floor_sit"), side * math.radians(u(10, 25))), drop, 0.85, 0.85),
                Key(u(1.7, 2.1), lying_pose("side_out", rng=rng, roll=side * math.pi / 2), smoothstep, 1.0, 1.0)]  # fmt: skip
    elif motion == "stumble":
        # Knocked off balance and catches it: a step back, arms out, upright again.
        knock = _roll(_p("lean_back"), side * math.radians(u(5, 15)))
        keys = [Key(u(0.25, 0.4), knock, smoothstep, 0.6, 0.6),
                Key(u(0.6, 0.8), lean(_p("stumble"), F_AXIS, side * math.radians(u(0, 10))), smoothstep, 0.9, 0.9),
                Key(u(1.2, 1.6), _p("standing"), smoothstep, 1.0, 1.0)]  # fmt: skip
    elif motion == "get_up":
        # From the floor: over onto hands and knees, one foot up, push up to standing.
        keys = [Key(u(1.2, 1.8), _p("hands_knees"), smoothstep, 0.3, 1.0),
                Key(u(2.4, 3.2), _p("half_kneel"), smoothstep, 0.6, 1.0),
                Key(u(3.4, 4.2), _p("half_stand"), smoothstep, 0.9, 1.0),
                Key(u(4.0, 5.0), _p("standing"), smoothstep, 1.0, 1.0)]  # fmt: skip
    else:
        raise ValueError(f"unknown motion {motion!r}")

    # Random key times can come out too close or out of order; keep each step a real step.
    for i in range(1, len(keys)):
        if keys[i].at < keys[i - 1].at + MIN_STEP_S:
            k = keys[i]
            keys[i] = Key(keys[i - 1].at + MIN_STEP_S, k.pose, k.ease, k.move, k.turn)
    if rng is not None:
        pace *= rng.uniform(0.85, 1.2)
    keys = _scaled(keys, pace)
    if fit is not None and keys[-1].at > fit:
        keys = _scaled(keys, fit / keys[-1].at)
    return keys


def length(motion: str, variant: int | None = None, *, pace: float = 1.0) -> float:
    """Seconds a motion takes."""
    return timeline(motion, variant, pace=pace)[-1].at

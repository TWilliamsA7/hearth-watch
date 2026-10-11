"""Who each simulated person is, and where they are right now.

A Profile is fixed for the run: archetype, height, walking pace, how sociable, how long
they tend to stay put and what they like doing. An Appearance is only for drawing (the
3D view colours each person differently). The Actor holds the runtime state the engine
updates every tick: position, heading, posture, body pose, the activity being carried
out and the plan of activities after it.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from hearth.sim.nav import Floor
from hearth.sim.pose import KEY_POSES, BodyPose

if TYPE_CHECKING:
    from hearth.sim.live.activities import Activity

Posture = Literal["standing", "sitting", "lying", "floor", "absent"]
Mobility = Literal["normal", "slow"]


@dataclass(frozen=True)
class Archetype:
    name: str
    speed: tuple[float, float]  # cruising pace, m/s
    sociability: tuple[float, float]
    dwell: tuple[float, float]  # multiplies how long they stay with an activity
    mobility: Mobility
    prefs: dict[str, float]  # intent -> weight multiplier


ARCHETYPES: dict[str, Archetype] = {
    a.name: a
    for a in (
        Archetype("steady", (0.85, 1.05), (0.4, 0.7), (0.9, 1.2), "normal",
                  {"sit": 1.0, "rest": 0.8, "window": 1.0, "browse": 1.0, "wander": 1.0, "chat": 1.0, "leave": 1.0}),
        Archetype("frail", (0.5, 0.7), (0.3, 0.6), (1.3, 1.8), "slow",
                  {"sit": 1.6, "rest": 1.6, "window": 0.7, "browse": 0.6, "wander": 0.4, "chat": 0.8, "leave": 0.5}),
        Archetype("restless", (0.9, 1.15), (0.3, 0.6), (0.5, 0.8), "normal",
                  {"sit": 0.6, "rest": 0.4, "window": 1.3, "browse": 1.3, "wander": 1.8, "chat": 0.8, "leave": 1.6}),
        Archetype("social", (0.75, 1.0), (0.75, 1.0), (0.9, 1.3), "normal",
                  {"sit": 1.0, "rest": 0.6, "window": 0.7, "browse": 0.8, "wander": 0.8, "chat": 2.0, "leave": 0.7}),
    )
}  # fmt: skip

SKIN = ("#f1d3bc", "#e0b394", "#c68c6a", "#a5694f", "#7a4b35", "#5a3626")
TOPS = ("#4f7cac", "#c46b5c", "#6c9a6b", "#8a6fb3", "#d9a441", "#5f8f9b", "#b5566e", "#e0e0e0", "#3f5d7d", "#9b7a5a")
BOTTOMS = ("#2f3542", "#4b4f58", "#5b4a3a", "#3c4f6b", "#6d6a63", "#2a2a2a", "#7b6d5a")
HAIR = ("#e8e4dc", "#c9c4bb", "#9a948a", "#5c4a3c", "#2b2420", "#b08a5a")


@dataclass(frozen=True)
class Appearance:
    skin: str
    top: str
    bottom: str
    hair: str
    build: float  # width relative to average


@dataclass(frozen=True)
class Profile:
    archetype: str
    height: float  # metres
    speed: float  # cruising pace, m/s
    sociability: float  # 0..1
    dwell: float
    mobility: Mobility
    prefs: dict[str, float]


def make_profile(rng: random.Random, archetype: str | None = None) -> tuple[Profile, Appearance]:
    arch = ARCHETYPES[archetype or rng.choices(list(ARCHETYPES), weights=(4, 3, 2, 3))[0]]
    height = round(rng.uniform(1.52, 1.86), 3)
    profile = Profile(
        archetype=arch.name,
        height=height,
        speed=round(rng.uniform(*arch.speed), 3),
        sociability=round(rng.uniform(*arch.sociability), 3),
        dwell=round(rng.uniform(*arch.dwell), 3),
        mobility=arch.mobility,
        prefs={k: round(v * rng.uniform(0.75, 1.3), 3) for k, v in arch.prefs.items()},
    )
    looks = Appearance(rng.choice(SKIN), rng.choice(TOPS), rng.choice(BOTTOMS), rng.choice(HAIR),
                       round(rng.uniform(0.9, 1.15), 3))  # fmt: skip
    return profile, looks


@dataclass
class Actor:
    person_id: str
    profile: Profile
    appearance: Appearance
    rng: random.Random
    pos: Floor
    heading: float
    base: float = 0.0  # metres above the floor the pose rests on
    posture: Posture = "standing"
    body: BodyPose = field(default_factory=lambda: dict(KEY_POSES["standing"]))
    velocity: Floor = (0.0, 0.0)
    gait_phase: float = 0.0
    activity: Activity | None = None
    plan: deque[Activity] = field(default_factory=deque)
    intent: str = "idle"  # what the current plan is for (sit, chat, fall...), shown to people watching
    intent_since: float = 0.0
    history: deque[tuple[str, float]] = field(default_factory=lambda: deque(maxlen=12))
    cooldowns: dict[str, float] = field(default_factory=dict)  # intent -> scenario time it's allowed again
    slots: list[str] = field(default_factory=list)  # slot ids held (sitting in, lying on, or headed for)
    conversation: str | None = None
    engaged_by: str | None = None  # someone is coming over to talk
    incident: str | None = None  # kind of the incident in progress
    look_at: Floor | None = None
    look_yaw: float = 0.0  # current head turn, radians, smoothed
    gesture: float = 0.0  # 0..1, how far an arm is raised while talking
    idle_phase: float = 0.0
    destination: Floor | None = None
    path: list[Floor] = field(default_factory=list)
    blocked_for: float = 0.0
    steps: int = 0  # ticks lived, for staggering expensive work

    @property
    def size(self) -> float:
        return self.profile.height

    @property
    def present(self) -> bool:
        return self.posture != "absent"

    @property
    def speed(self) -> float:
        return math.hypot(*self.velocity)

    @property
    def on_floor_space(self) -> bool:
        """Standing or walking on open floor (not on furniture, not away): a body others walk around."""
        return self.posture == "standing" and not self.slots_in_use

    @property
    def slots_in_use(self) -> bool:
        return bool(self.slots) and self.posture in ("sitting", "lying")

    @property
    def label(self) -> str:
        if self.activity is not None:
            return self.activity.label_for(self)
        return {"standing": "standing", "sitting": "sitting", "lying": "lying_intentional", "floor": "fall",
                "absent": "absent"}[self.posture]  # fmt: skip


LIVE_LOOKS = Appearance("#c9a48a", "#e5484d", "#2f3542", "#5c4a3c", 1.0)  # the LIVE avatar's colours


@dataclass
class ExternalActor(Actor):
    """A real person seen by a live camera, placed in the room from their lifted keypoints.

    Never driven by behaviour, never given ground truth or incidents. Kept in World.externals,
    not World.actors: residents see the body (they steer round it, it blocks the simulated
    camera, they come to help when it is down), and nothing else in the engine touches it."""

    source: str = ""
    joints: dict = field(default_factory=dict)  # name -> (x, y, z) metres, from hearth.sim.live.lift
    seen_at: float = 0.0  # wall-clock seconds of the last live frame
    live_label: str = "standing"  # what the real monitor concluded (a TruthLabel-like word)

    @property
    def label(self) -> str:
        return self.live_label


def external_profile(height: float) -> Profile:
    return Profile(archetype="live", height=height, speed=0.0, sociability=0.0, dwell=1.0, mobility="normal",
                   prefs={})  # fmt: skip

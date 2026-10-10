"""API shapes for the live engine (exported to backend/schema/sim.schema.json).

Like the world track, all of this is simulator-only: it's what people watching the
simulation see (who's doing what, where they're going, ground truth), never what the
monitor sees.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hearth.sim.live.incidents import IncidentRecord
from hearth.sim.live.spec import EngineSpec
from hearth.sim.scenario import Scene
from hearth.sim.world import WorldPerson, WorldRoom

LiveState = Literal["idle", "playing", "paused"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class LiveAppearance(_Model):
    skin: str
    top: str
    bottom: str
    hair: str
    build: float = Field(description="Width relative to average.")


class LiveActorInfo(_Model):
    person_id: str
    archetype: str
    height: float
    speed: float = Field(description="Cruising pace, m/s.")
    sociability: float
    mobility: str
    appearance: LiveAppearance


class LivePerson(WorldPerson):
    """A WorldPerson plus what they're up to. Ground truth: for people watching only."""

    label: str = Field(description="Ground-truth label right now (TruthLabel).")
    activity: str = Field(description="What they're doing this moment, e.g. 'sitting down', 'talking'.")
    intent: str = Field(description="What the current plan is for: sit, rest, chat, fall...")
    posture: str
    destination: tuple[float, float] | None = None
    path: list[tuple[float, float]] = Field(default_factory=list, description="Rest of the route, floor (x, z).")
    conversation: str | None = None
    incident: str | None = None
    visible: float = Field(description="Fraction of joints the camera could see last frame (0..1), before noise.")


class LiveConversation(_Model):
    conv_id: str
    members: list[str]
    centre: tuple[float, float]


class LiveFrame(_Model):
    """Everyone right now. Pushed over WS /sim/live/ws as {"type": "frame", "data": ...}."""

    t: float
    state: LiveState
    speed: float
    people: list[LivePerson]
    conversations: list[LiveConversation]
    incidents: list[IncidentRecord] = Field(description="Incidents in progress.")


class LiveWorld(_Model):
    """The run's fixed parts: sent once per connection and on every new room ({"type": "world"})."""

    run_id: str
    scenario_id: str
    seed: int
    spec: EngineSpec
    room: WorldRoom
    scene: Scene = Field(description="The furniture as the camera sees it, for the 2D camera view.")
    actors: list[LiveActorInfo]


class ActorDetail(_Model):
    info: LiveActorInfo
    now: LivePerson
    since: float = Field(description="Scenario seconds the current intent began.")
    plan: list[str] = Field(description="Activities still to come.")
    history: list[tuple[str, float]] = Field(description="Recent intents and when they began.")
    cooldowns: dict[str, float]
    incidents: list[IncidentRecord]


class LiveStatus(_Model):
    state: LiveState
    run_id: str | None = None
    scenario_id: str | None = None
    seed: int | None = None
    t: float = 0.0
    speed: float = 1.0
    actors: int = 0

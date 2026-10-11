"""Telling a simulated person what to do: sit here, lie there, go and kneel over there.

For the Sim Studio, to stage a moment on demand (or reproduce one) instead of waiting for
the Brain to get round to it. A command replaces the person's plan, as an incident does,
and is built from the same activities their own choices are (sit_plan, lie_plan, Walk,
Transition...), so a commanded body moves exactly as an unprompted one. When the plan
runs out the Brain takes over again.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from hearth.sim.live.activities import (
    Activity,
    Do,
    Hold,
    Transition,
    Walk,
    dwell,
    floor_sit_steps,
    get_up,
    lie_plan,
    lie_stances,
    reserve,
    sit_plan,
    tie_shoe_steps,
)
from hearth.sim.live.actor import Actor
from hearth.sim.live.spec import IncidentSpec

if TYPE_CHECKING:
    from hearth.sim.live.engine import World

CommandAction = Literal["go", "sit", "lie", "kneel", "tie_shoe", "pick_up", "floor_sit", "fall"]
SLOT_ACTIONS = ("sit", "lie")


class ActorCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    action: CommandAction
    slot_id: str | None = Field(default=None, description="sit, lie: which slot (LiveWorld.slots).")
    at: tuple[float, float] | None = Field(default=None, description="Other actions: the floor point (x, z) to do it at.")


class CommandError(ValueError):
    """The person can't do that now (409), with a reason to show."""


def command_plan(a: Actor, w: World, cmd: ActorCommand) -> list[Activity]:
    """The plan for `cmd`, after reserving any slot it needs. Raises CommandError if it can't
    be done, KeyError for an unknown slot."""
    if not a.present:
        raise CommandError(f"{a.person_id} is out of the room")
    if a.incident:
        raise CommandError(f"{a.person_id} has a {a.incident} in progress; reset them first")
    if a.posture == "floor":
        raise CommandError(f"{a.person_id} is on the floor; reset them first")
    if cmd.action in SLOT_ACTIONS:
        if cmd.slot_id is None:
            raise CommandError(f"{cmd.action} needs a slot_id")
        slot = w.room.slot(cmd.slot_id)
        return _sit(a, w, slot) if cmd.action == "sit" else _lie(a, w, slot)
    if cmd.at is None:
        raise CommandError(f"{cmd.action} needs a floor point (at)")
    at = (round(cmd.at[0], 3), round(cmd.at[1], 3))
    if not w.room.nav.free(at):
        raise CommandError("that's not open floor: pick somewhere a person can stand")
    there = get_up(a, w) + [Walk(at, purpose="as told", stop_short_ok=False)]
    if cmd.action == "go":
        return there + [Hold(dwell(a, w, 6.0, 15.0), "standing about")]
    if cmd.action == "kneel":
        return there + [
            Transition("kneel", 1.2, name="kneeling down", label="bending", posture="standing"),
            Hold(a.rng.uniform(5.0, 10.0), "kneeling", pose="kneel", label="bending", sway=0.3),
            Transition("standing", 1.2, name="getting up", label="standing", posture="standing"),
        ]
    if cmd.action == "tie_shoe":
        return there + tie_shoe_steps(a)
    if cmd.action == "floor_sit":
        return there + floor_sit_steps(a, w)
    if cmd.action == "pick_up":
        return there + [
            Transition("bending", 0.9, name="bending down", label="bending", posture="standing"),
            Hold(a.rng.uniform(1.5, 3.5), "picking up", pose="bending", label="bending", sway=0.3),
            Transition("standing", 1.0, name="straightening up", label="standing", posture="standing"),
        ]
    # fall: get there, then fall where they stand (the incident does the rest: the record,
    # the ground truth, someone coming to help).
    pid = a.person_id
    return there + [Do(lambda a, w: w.incidents.trigger(IncidentSpec(kind="fall", actor=pid, in_view=False)))]


def _sit(a: Actor, w: World, seat) -> list[Activity]:
    if seat.kind != "sit":
        raise CommandError(f"{seat.slot_id} is for lying on; pick a seat")
    if seat.occupant not in (None, a.person_id):
        raise CommandError(f"{seat.occupant} has {seat.slot_id}")
    if any(o.kind == "lie" and o.group == seat.group and o.occupant not in (None, a.person_id) for o in w.room.slots):
        raise CommandError(f"someone is lying on {seat.group}")
    if a.posture == "sitting" and seat.slot_id in a.slots:
        raise CommandError(f"{a.person_id} is already sitting there")
    if w.room.approach(seat, a.size) is None:
        raise CommandError(f"there's no way up to {seat.slot_id} from open floor")
    first = get_up(a, w)
    reserve(a, w, seat)
    plan = sit_plan(a, w, seat)
    assert plan is not None
    return first + plan + [Hold(dwell(a, w, 25.0, 90.0), "seated")]


def _lie(a: Actor, w: World, lie) -> list[Activity]:
    if lie.kind != "lie":
        lie = next((s for s in w.room.slots if s.kind == "lie" and s.group == lie.group), None) or lie
        if lie.kind != "lie":
            raise CommandError(f"{lie.slot_id} isn't something to lie on")
    if w.room.group_busy(lie.group, besides=a.person_id):
        raise CommandError(f"someone is already using {lie.group}")
    if lie_stances(lie, w.room.pieces[lie.piece], a.size) is None:
        raise CommandError(f"{a.person_id} ({a.size:.2f} m) doesn't fit on it, even curled up")
    entries = [s for s in w.room.slots if s.group == lie.group and s.kind == "sit" and w.room.approach(s, a.size)]
    if not entries:
        raise CommandError(f"there's no way onto {lie.group} from open floor")
    entry = min(entries, key=lambda s: (s.slot_id != lie.entry, math.dist(s.at, a.pos)))
    first = get_up(a, w)
    reserve(a, w, lie, entry)
    plan = lie_plan(a, w, lie, entry)
    assert plan is not None
    return first + plan


def apply_command(w: World, person_id: str, cmd: ActorCommand) -> None:
    """Have `person_id` do `cmd` now, instead of whatever they were doing."""
    a = w.actor(person_id)
    held = list(a.slots)
    plan = command_plan(a, w, cmd)
    taken = [w.room.slot(s) for s in a.slots if s not in held]
    w.replace_plan(a, plan, "command")
    reserve(a, w, *taken)  # replace_plan frees seats the person isn't in yet

"""World track: where every simulated body really is, in 3D, for the dashboard's 3D view.

The camera stream carries 2D keypoints only. The world track carries the same bodies,
noise-free, as metres in a room. In fact the bodies are built here first: a pose
(hearth.sim.pose) is placed in the room (body()), and the camera's keypoints are where
that body lands in the picture (to_image()), so the two views always agree. Like ground truth,
it is simulator-only: it never travels in the Observation stream and the monitor never
sees it. Every resident is in it, including people the camera cannot see and during a
camera outage; the world goes on while the camera is dark.

Coordinates follow three.js: metres, y up. The camera stands at the front of the room
looking into it along -z, so image right is +x and room depth d (0 = front, 1 = back
wall) is z = -d * ROOM_DEPTH_M. The 2D sim fakes perspective with Scene.depth_shrink;
the world inverts it, so a frame x at depth d lands where that camera would have seen it:
    world_x = (x - 0.5) * front_width / (1 - d * depth_shrink)
"""

from __future__ import annotations

import math
from typing import Literal

from hearth_schema.models import KeypointName
from pydantic import BaseModel, ConfigDict, Field

from hearth.sim.pose import BodyPose
from hearth.sim.scenario import Facing, ObjectKind, Scene, SceneObject

PERSON_HEIGHT_M = 1.7  # a resident with height 1.0
ROOM_DEPTH_M = 4.0  # front of the room (depth 0) to the back wall (depth 1)
ROOM_HEIGHT_M = 2.6

# Front-to-back size of furniture, which the 2D scene doesn't record.
OBJECT_DEPTH_M: dict[str, float] = {"bed": 1.0, "chair": 0.5, "couch": 0.9, "mat": 1.0, "occluder": 0.35}
MAT_HEIGHT_M = 0.02
OCCLUDER_GAP_M = 0.5  # an occluder stands this far in front of the depth it hides

HEADING: dict[str, float] = {"camera": 0.0, "right": math.pi / 2, "left": -math.pi / 2, "away": math.pi}
"""Radians about +y. 0 faces the camera (+z), pi/2 faces image right (+x)."""

Vec3 = tuple[float, float, float]
WorldPose = dict[str, Vec3]
"""joint name -> (x, y, z) in metres."""
ImagePose = dict[str, tuple[float, float]]
"""joint name -> (x, y) in normalized frame coordinates (may fall outside the frame)."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class WorldPerson(_Model):
    person_id: str | None = Field(description="Matches Observation.person_id; None in a single-person run.")
    heading: float = Field(description="Radians about +y; 0 faces the camera, +pi/2 faces image right.")
    height: float = Field(description="Standing height in metres. Constant per person, unlike limb lengths in motion.")
    joints: dict[KeypointName, Vec3] = Field(description="Metres; every joint, visible to the camera or not.")


class WorldFrame(_Model):
    t: float = Field(description="Scenario seconds.")
    people: list[WorldPerson]


class WorldObject(_Model):
    kind: ObjectKind
    center: Vec3
    size: Vec3 = Field(description="Width (x), height (y), depth (z) in metres.")
    rotation: float = Field(default=0.0, description="Radians about +y: the heading its front faces (as WorldPerson.heading).")
    variant: str | None = Field(
        default=None, description="The full kind for live rooms (armchair, table, bookshelf...); kind is the nearest ObjectKind."
    )
    # Live rooms: the parts people touch, so the 3D view draws them where the simulator puts bodies.
    seat_height: float | None = Field(default=None, description="Metres: top of the seat. None = not for sitting, or unknown.")
    surface_height: float | None = Field(default=None, description="Metres: top of what someone lies on.")
    arm_width: float | None = Field(default=None, description="Metres: width of each arm (armchairs, couches).")
    arm_height: float | None = Field(default=None, description="Metres: top of the arms.")
    back_depth: float | None = Field(
        default=None, description="Metres the backrest (or a bed's headboard) comes forward from the back of the piece."
    )


class WorldDoor(_Model):
    door_id: str
    wall: Literal["back", "left", "right", "front"]
    center: tuple[float, float] = Field(description="Floor point (x, z) at the middle of the opening.")
    width: float
    heading_in: float = Field(description="Radians: the way into the room.")


class WorldWindow(_Model):
    wall: Literal["back", "left", "right", "front"]
    center: tuple[float, float]
    width: float
    sill: float
    height: float


class WorldRoom(_Model):
    width_front: float = Field(description="Metres the camera sees across at depth 0.")
    width_back: float = Field(description="Metres the camera sees across at the back wall.")
    depth: float
    height: float
    camera_position: Vec3 = Field(description="Approximate; for display only.")
    camera_target: Vec3
    camera_fov: float = Field(description="Vertical field of view, degrees.")
    objects: list[WorldObject]
    bounds: tuple[float, float, float, float] | None = Field(
        default=None, description="Live rooms: the walls as (x0, z0, x1, z1). None = the scripted room (width_back x depth)."
    )
    doors: list[WorldDoor] = Field(default_factory=list)
    windows: list[WorldWindow] = Field(default_factory=list)
    camera_aspect: float | None = Field(default=None, description="Width / height of the camera image.")
    lighting: str | None = Field(default=None, description="Live rooms: day, evening or dim.")


class WorldRoute(_Model):
    """One continuous walk: the straight legs a resident takes from t_start to t_end."""

    person_id: str | None
    t_start: float
    t_end: float
    points: list[tuple[float, float]] = Field(description="Floor points (x, z) in metres, start first.")


class WorldTrack(_Model):
    """A run's bodies in 3D (<run>.world.json, GET /sim/world). Simulator-only."""

    scenario_id: str
    seed: int
    fps: float
    duration: float
    room: WorldRoom
    frames: list[WorldFrame] = Field(description="One per frame slot, in time order.")
    routes: list[WorldRoute] = Field(default_factory=list, description="Every walk, for drawing routes on the floor.")


def metres_per_y(scene: Scene, depth: float) -> float:
    """Metres per unit of frame height for something at this depth."""
    return PERSON_HEIGHT_M / scene.person_height / (1.0 - depth * scene.depth_shrink)


def metres_per_x(scene: Scene, depth: float) -> float:
    return metres_per_y(scene, depth) * scene.frame_aspect


def world_x(scene: Scene, x: float, depth: float) -> float:
    return (x - 0.5) * metres_per_x(scene, depth)


def world_z(depth: float) -> float:
    return -depth * ROOM_DEPTH_M


def frame_depth(z: float) -> float:
    return -z / ROOM_DEPTH_M


def frame_x(scene: Scene, x: float, depth: float) -> float:
    """Inverse of world_x."""
    return x / metres_per_x(scene, depth) + 0.5


def lift(scene: Scene, surface_y: float | None, depth: float) -> float:
    """Metres above the floor of a surface at frame y surface_y (None = the floor)."""
    return 0.0 if surface_y is None else max((scene.floor_at(depth) - surface_y) * metres_per_y(scene, depth), 0.0)


def body(pose: BodyPose, heading: float, x: float, z: float, base: float, size: float) -> WorldPose:
    """A pose standing (or lying) at (x, z) on the floor, `base` metres up, `size` metres tall."""
    fx, fz = math.sin(heading), math.cos(heading)  # forward
    lx, lz = math.cos(heading), -math.sin(heading)  # the person's left
    return {joint: (x + size * (f * fx + l * lx), base + v * size, z + size * (f * fz + l * lz))
            for joint, (f, v, l) in pose.items()}  # fmt: skip


def place(
    pose: BodyPose, facing: Facing | float, root_x: float, base_y: float, depth: float, scene: Scene, height: float
) -> WorldPose:
    """body(), from frame coordinates: root at frame x and depth, on the surface at frame y
    base_y, `height` frame units tall at that depth.
    """
    heading = HEADING[facing] if isinstance(facing, str) else facing
    my = metres_per_y(scene, depth)
    base = (scene.floor_at(depth) - base_y) * my
    return body(pose, heading, world_x(scene, root_x, depth), world_z(depth), base, height * my)


def to_image(pose: WorldPose, scene: Scene) -> ImagePose:
    """Where the camera sees each joint: the inverse of the perspective world_x undoes.

    Each joint is projected at its own depth, so a body lying towards the camera is
    foreshortened, and one facing away shows its back.
    """
    out: ImagePose = {}
    for joint, (x, y, z) in pose.items():
        depth = frame_depth(z)
        out[joint] = (frame_x(scene, x, depth), scene.floor_at(depth) - y / metres_per_y(scene, depth))
    return out


def _round(v: Vec3) -> Vec3:
    return (round(v[0], 3), round(v[1], 3), round(v[2], 3))


def world_person(person_id: str | None, heading: float, pose: WorldPose, height: float) -> WorldPerson:
    joints = {j: _round(p) for j, p in pose.items()}
    return WorldPerson(person_id=person_id, heading=round(heading, 4), height=round(height, 3), joints=joints)  # type: ignore[arg-type]


def _object(scene: Scene, o: SceneObject) -> WorldObject:
    bottom = o.y + o.h
    depth = min(max((scene.floor_y - bottom) / scene.depth_rise, 0.0), 1.0) if scene.depth_rise else 0.0
    mx, my = metres_per_x(scene, depth), metres_per_y(scene, depth)
    extent = OBJECT_DEPTH_M[o.kind]
    lift = max((scene.floor_at(depth) - bottom) * my, 0.0)
    height = MAT_HEIGHT_M if o.kind == "mat" else o.h * my
    z = world_z(depth) + (OCCLUDER_GAP_M if o.kind == "occluder" else 0.0)
    center = (world_x(scene, o.x + o.w / 2, depth), lift + height / 2, z)
    return WorldObject(kind=o.kind, center=_round(center), size=_round((o.w * mx, height, extent)))


def room_geometry(scene: Scene) -> WorldRoom:
    front_h = metres_per_y(scene, 0.0)
    # A pinhole camera that shrinks the back wall by depth_shrink stands this far in front
    # of depth 0. Clamped to something a room could hold; the 3D camera is only a marker.
    dist = ROOM_DEPTH_M * (1.0 - scene.depth_shrink) / scene.depth_shrink if scene.depth_shrink else 3.0
    dist = min(max(dist, 1.0), 3.0)
    eye_y = (scene.floor_y - 0.5) * front_h
    fov = math.degrees(2.0 * math.atan(front_h / 2.0 / dist))
    return WorldRoom(
        width_front=round(metres_per_x(scene, 0.0), 3),
        width_back=round(metres_per_x(scene, 1.0), 3),
        depth=ROOM_DEPTH_M,
        height=ROOM_HEIGHT_M,
        camera_position=_round((0.0, eye_y, dist)),
        camera_target=_round((0.0, eye_y, -ROOM_DEPTH_M / 2)),
        camera_fov=round(fov, 2),
        objects=[_object(scene, o) for o in scene.objects],
    )

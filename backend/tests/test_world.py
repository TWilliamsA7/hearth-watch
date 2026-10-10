"""The 3D world track: the same bodies as the camera stream, placed in a room in metres."""

import math

import pytest
from hearth.sim.generator import ObservationGenerator
from hearth.sim.pose import body_pose, project
from hearth.sim.scenario import Scene, SceneObject, load_all
from hearth.sim.world import PERSON_HEIGHT_M, ROOM_DEPTH_M, WorldTrack, place, room_geometry, world_x

SCENARIOS = load_all()
ALL = pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=lambda s: s.scenario_id)
SCENE = Scene()


def standing(facing="camera", x=0.5, depth=0.0, base_y=None, pose="standing"):
    base = SCENE.floor_at(depth) if base_y is None else base_y
    return place(body_pose(pose), facing, x, base, depth, SCENE, SCENE.height_at(depth))


def test_standing_person_at_the_front_is_life_size():
    joints = standing()
    assert joints["left_ankle"][1] == pytest.approx(0.03 * PERSON_HEIGHT_M)
    assert joints["nose"][1] == pytest.approx(0.93 * PERSON_HEIGHT_M)
    assert joints["neck"][0] == pytest.approx(0.0) and joints["neck"][2] == pytest.approx(0.0)


def test_facing_the_camera_puts_the_left_side_at_image_right():
    joints = standing("camera")
    assert joints["left_shoulder"][0] > joints["right_shoulder"][0]
    assert joints["nose"][2] > joints["neck"][2]  # the face points at the camera (+z)


def test_facing_right_points_the_face_along_plus_x():
    joints = standing("right")
    assert joints["nose"][0] > joints["left_hip"][0]
    assert joints["left_shoulder"][2] < joints["right_shoulder"][2]  # left side away from the camera


@pytest.mark.parametrize("pose", ["lying", "fallen"])
def test_lying_bodies_stay_near_the_floor(pose):
    assert max(y for _, y, _ in standing("right", pose=pose).values()) < 0.45


def test_deeper_people_are_further_away_and_the_same_size():
    near, far = standing(depth=0.0), standing(depth=0.8)
    assert far["neck"][2] == pytest.approx(-0.8 * ROOM_DEPTH_M)
    assert far["nose"][1] == pytest.approx(near["nose"][1])  # smaller on screen, not in the room


def test_world_x_inverts_the_camera_perspective():
    # Two people at the same frame x but different depths: the deeper one is further off-centre.
    assert abs(world_x(SCENE, 0.8, 1.0)) > abs(world_x(SCENE, 0.8, 0.0))
    # A person's world width matches their width in the frame at their own depth.
    depth = 0.6
    img = project(body_pose("standing"), "camera", 0.5, SCENE.floor_at(depth), SCENE, SCENE.height_at(depth))
    world = standing(depth=depth)
    img_w = (img["left_shoulder"][0] - img["right_shoulder"][0]) * SCENE.frame_aspect
    world_w = world["left_shoulder"][0] - world["right_shoulder"][0]
    assert world_w / PERSON_HEIGHT_M == pytest.approx(img_w / SCENE.height_at(depth))


def test_a_raised_surface_lifts_the_body():
    on_floor = standing("right", pose="lying")
    on_bed = standing("right", pose="lying", base_y=SCENE.floor_y - 0.15)
    lift = on_bed["left_hip"][1] - on_floor["left_hip"][1]
    assert lift == pytest.approx(0.15 * PERSON_HEIGHT_M / SCENE.person_height)


def test_furniture_sits_on_the_floor_at_its_depth():
    scene = Scene(objects=[SceneObject(kind="chair", x=0.1, y=0.66, w=0.12, h=0.218)])
    (chair,) = room_geometry(scene).objects
    assert chair.center[2] == pytest.approx(-0.3 * ROOM_DEPTH_M, abs=1e-3)  # bottom edge 0.878 -> depth 0.3
    assert chair.center[1] == pytest.approx(chair.size[1] / 2, abs=1e-3)


@ALL
def test_world_track_is_deterministic_and_valid(scenario):
    a = ObservationGenerator(scenario).world_track()
    b = ObservationGenerator(scenario).world_track()
    assert a == b
    assert WorldTrack.model_validate_json(a.model_dump_json()) == a
    assert len(a.frames) == len(list(ObservationGenerator(scenario).frame_times()))
    for frame in a.frames:
        for person in frame.people:
            assert len(person.joints) == 18
            assert all(math.isfinite(c) for joint in person.joints.values() for c in joint)


def test_absent_residents_are_left_out_but_outages_are_not():
    track = ObservationGenerator(SCENARIOS["day-room-01"]).world_track()
    p2_absent = next(g for g in SCENARIOS["day-room-01"].ground_truth() if g.person_id == "p2" and g.truth_label == "absent")
    mid = next(f for f in track.frames if p2_absent.t_start + 0.5 <= f.t < p2_absent.t_end)
    assert "p2" not in {p.person_id for p in mid.people}

    outage = ObservationGenerator(SCENARIOS["camera-outage-01"])
    gt = next(g for g in SCENARIOS["camera-outage-01"].ground_truth() if g.truth_label == "camera_outage")
    dark = [f for f in outage.world_track().frames if gt.t_start <= f.t < gt.t_end]
    assert dark and all(len(f.people) == 1 for f in dark)  # the camera is dark; the person is still there


def test_heading_turns_through_a_transition():
    gen = ObservationGenerator(SCENARIOS["day-room-02"])
    p1 = gen.tracks[0]  # sits facing the camera, then lies down side-on over 2.5 s
    start = p1.plans[1].t_start
    assert p1.world_at(start - 0.1).heading == pytest.approx(0.0)
    assert 0.0 < p1.world_at(start + 1.0).heading < math.pi / 2
    assert p1.world_at(start + 3.0).heading == pytest.approx(math.pi / 2, abs=1e-3)

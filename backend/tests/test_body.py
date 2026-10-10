"""The body: poses keep their proportions, walks look like walking, falls differ."""

import math

import pytest
from hearth.sim.generator import NoiseConfig, ObservationGenerator, ResidentTrack
from hearth.sim.motion import timeline
from hearth.sim.pose import KEY_POSES, LATERAL, REST, BONES, bone_lengths, body_pose, hips_of
from hearth.sim.scenario import Resident, Scenario, Scene, Segment, load_all
from hearth.sim.world import PERSON_HEIGHT_M, place, to_image

SCENE = Scene()
MOTIONS = ["trip", "faint", "sit_miss", "stand_up", "collision", "stumble", "get_up", "collapse"]


def track(*segments, **kw) -> ResidentTrack:
    return ResidentTrack(Resident(segments=list(segments), **kw), SCENE)


def world_bones(joints) -> dict:
    def at(name):
        if name == "hips":
            return tuple((a + b) / 2 for a, b in zip(joints["left_hip"], joints["right_hip"]))
        return joints[name]

    return {(p, c): math.dist(at(p), at(c)) for p, c in BONES}


def test_the_camera_sees_the_body_where_the_old_flat_model_put_it():
    """Facing the camera, projecting the 3D body matches placing the pose flat in the frame."""
    depth, x = 0.4, 0.3
    height = SCENE.height_at(depth)
    img = to_image(place(body_pose("standing"), "camera", x, SCENE.floor_at(depth), depth, SCENE, height), SCENE)
    for joint, (f, v, _) in KEY_POSES["standing"].items():
        flat = (x + LATERAL[joint] * height / SCENE.frame_aspect, SCENE.floor_at(depth) - v * height)
        assert img[joint] == pytest.approx(flat, abs=0.01)


@pytest.mark.parametrize("name", sorted(KEY_POSES))
def test_every_key_pose_has_standing_proportions(name):
    lengths = bone_lengths(KEY_POSES[name])
    assert all(lengths[b] == pytest.approx(REST[b], rel=1e-6) for b in REST)


@pytest.mark.parametrize("scenario", load_all().values(), ids=lambda s: s.scenario_id)
def test_bones_keep_their_length_through_every_movement(scenario):
    gen = ObservationGenerator(scenario, noise=NoiseConfig.off())
    for frame in gen.world_track().frames[::3]:
        for person in frame.people:
            for bone, length in world_bones(person.joints).items():
                assert length == pytest.approx(REST[bone] * person.height, rel=0.02, abs=0.005), (frame.t, bone)


@pytest.mark.parametrize(("depth_from", "depth_to", "heading"), [(0.1, 0.9, math.pi), (0.9, 0.1, 0.0)])
def test_walking_into_and_out_of_the_room_faces_the_way_of_travel(depth_from, depth_to, heading):
    t = track(Segment(label="walking", duration=4.0, transition=0.0, x=0.5, depth=depth_to), start_depth=depth_from)
    assert abs(math.remainder(t.world_at(2.0).heading - heading, 2 * math.pi)) < 1e-3


def test_a_walk_speeds_up_cruises_and_slows_down():
    t = track(Segment(label="walking", duration=6.0, transition=0.0, x=0.9), start_x=0.1)

    def speed(at):
        (x0, z0), (x1, z1) = t.root_at(at - 0.05), t.root_at(at + 0.05)
        return math.hypot(x1 - x0, z1 - z0) / 0.1

    assert speed(0.1) < 0.5 * speed(3.0)
    assert speed(5.9) < 0.5 * speed(3.0)
    assert t.root_at(6.0) == pytest.approx(t.plans[0].end, abs=1e-6)


def test_the_planted_foot_does_not_skate():
    t = track(Segment(label="walking", duration=8.0, transition=0.0, x=1.0), start_x=0.0)
    cruise = t.plans[0].walk.cruise
    dt = 1 / 60
    ankle = [t.world_pose_at(0, 3.0 + k * dt)["left_ankle"] for k in range(120)]  # two seconds of cruising
    speeds = [math.dist(a, b) / dt for a, b in zip(ankle, ankle[1:])]
    assert min(speeds) < 0.2 * cruise  # at some point in each step the foot is (nearly) still


@pytest.mark.parametrize("motion", MOTIONS)
def test_motions_keep_proportions_and_vary_by_variant(motion):
    keys = [timeline(motion, v) for v in range(4)]
    for ks in keys:
        assert all(k.at > 0 for k in ks) and [k.at for k in ks] == sorted(k.at for k in ks)
        for k in ks:
            assert all(bone_lengths(k.pose)[b] == pytest.approx(REST[b], rel=1e-6) for b in REST)
    if motion != "collapse":  # collapse is the plain fall hand-written scenarios get, the same every time
        ends = {tuple(round(c, 4) for c in ks[-1].pose["left_wrist"]) for ks in keys}
        times = {round(ks[-1].at, 4) for ks in keys}
        assert len(ends) > 1 or len(times) > 1


def test_each_kind_of_fall_ends_differently():
    ends = {m: timeline(m, 0)[-1].pose for m in ("trip", "faint", "sit_miss", "stand_up")}
    trip = ends.pop("trip")
    hips = hips_of(trip)
    assert trip["nose"][0] > hips[0] + 0.2  # landed head first, along the way they were going
    for other in ends.values():
        assert other != trip


def test_a_trip_lands_face_down():
    prone = timeline("trip", 3)[-1].pose
    # Face down: the nose is no higher than the back of the neck.
    assert prone["nose"][1] <= prone["neck"][1] + 0.02


def fall_and_get_up(motion, variant):
    return Scenario(
        scenario_id="fall-and-get-up", title="t", variation={"duration": 0, "transition": 0, "position": 0},
        start_x=0.5, start_facing="right",
        segments=[
            Segment(label="standing", duration=2.0, transition=0.0),
            Segment(label="fall", duration=6.0, transition=0.8, motion=motion, variant=variant),
            Segment(label="standing", duration=7.0, transition=0.5, motion="get_up", variant=variant),
        ],
    )


@pytest.mark.parametrize("motion", ["trip", "faint", "sit_miss", "stand_up", "collision"])
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_falling_and_getting_up_is_continuous(motion, variant):
    gen = ObservationGenerator(fall_and_get_up(motion, variant), noise=NoiseConfig.off())
    prev = None
    for _, t in gen.frame_times():
        (person,) = gen.people_at(t)
        if prev is not None:
            jump = max(math.dist(person.world[j], prev[j]) for j in prev)
            assert jump < 0.5, f"t={t:.2f} jump {jump:.2f} m"  # hands fly out at ~5 m/s, but no teleports
        prev = person.world
    end = gen.people_at(14.9)[0]
    assert not end.lying and end.world["nose"][1] > 0.85 * PERSON_HEIGHT_M


def test_falls_are_on_the_floor_by_the_end():
    for motion in ("trip", "faint", "stand_up", "collision"):
        gen = ObservationGenerator(fall_and_get_up(motion, 5), noise=NoiseConfig.off())
        person = gen.people_at(6.0)[0]
        assert person.lying, motion
        assert max(y for _, y, _ in person.world.values()) < 0.6

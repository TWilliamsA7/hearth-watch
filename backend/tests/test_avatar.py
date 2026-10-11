"""The live avatar: steady, whole and accurate on noisy, flickering single-camera keypoints."""

import math
import random

import pytest
from hearth.sim.live.avatar import BLEND_S, POSTURE_HOLD_S, LiveAvatar
from hearth.sim.live.camera import Camera
from hearth.sim.live.room import generate_room
from hearth.sim.live.spec import CameraSpec, RoomSpec
from hearth.sim.pose import KEY_POSES
from hearth.sim.world import body

ROOM = generate_room(RoomSpec(width=7.0, depth=6.0), 1)
CAMERA = Camera(ROOM, CameraSpec(preset="jetson", fov=43.0))
H = 1.7
DT = 0.1  # the Jetson sends ~10 observations a second


def frame(x, z, pose="standing", heading=0.0, noise=0.0, drop=0.0, rng=None, body_pose=None):
    """Edge-style keypoints (name, u, v) for a body at (x, z): jittered, some joints missing."""
    rng = rng or random.Random(0)
    truth = body(body_pose or KEY_POSES[pose], heading, x, z, 0.0, H)
    out = []
    for name, p in truth.items():
        q = CAMERA.project(p)
        if q is None or not (0 <= q[0] <= 1 and 0 <= q[1] <= 1) or rng.random() < drop:
            continue
        out.append((name, q[0] + rng.gauss(0, noise), q[1] + rng.gauss(0, noise)))
    return out


def run(avatar, frames, posture="standing", t0=0.0):
    t = t0
    for kps in frames:
        avatar.update(kps, posture, CAMERA, ROOM, t)
        t += DT
    return t


def test_stands_where_the_person_stands():
    a = LiveAvatar(H)
    run(a, [frame(0.5, -3.0)] * 20)
    assert math.dist(a.root, (0.5, -3.0)) < 0.1


def test_noise_and_dropouts_barely_move_a_still_person():
    rng = random.Random(1)
    a = LiveAvatar(H)
    roots = []
    t = 0.0
    for _ in range(80):  # 8 s standing still; ~3 px of jitter at 640x480, 15% of joints dropping
        a.update(frame(0.0, -3.0, noise=0.005, drop=0.15, rng=rng), "standing", CAMERA, ROOM, t)
        t += DT
        roots.append(a.root)
    later = roots[20:]
    spread = max(math.dist(r, later[0]) for r in later)
    assert spread < 0.05


def test_the_skeleton_never_stretches():
    """The 'broken neck': bone lengths must not change, however the keypoints flicker."""
    rng = random.Random(2)
    a = LiveAvatar(H)
    necks, arms = [], []
    t = 0.0
    for i in range(60):
        a.update(frame(0.3 * math.sin(i / 10), -3.0, noise=0.008, drop=0.3, rng=rng), "standing", CAMERA, ROOM, t)
        t += DT
        j = a.joints()
        necks.append(math.dist(j["neck"], j["nose"]))
        arms.append(math.dist(j["left_shoulder"], j["left_elbow"]))
    assert max(necks) - min(necks) < 1e-6
    assert max(arms) - min(arms) < 1e-6


def test_follows_someone_walking():
    a = LiveAvatar(H)
    t = run(a, [frame(-0.5, -3.0)] * 10)
    walk = [frame(-0.5 + 0.1 * i, -3.0) for i in range(1, 11)]  # 1 m in 1 s
    t = run(a, walk, t0=t)
    t = run(a, [frame(0.5, -3.0)] * 5, t0=t)  # 0.5 s after they stop
    assert math.dist(a.root, (0.5, -3.0)) < 0.15


def test_one_wild_frame_is_ignored():
    a = LiveAvatar(H)
    t = run(a, [frame(0.0, -3.0)] * 15)
    before = a.root
    t = run(a, [frame(2.5, -1.2)], t0=t)  # a mis-detection 3 m away
    t = run(a, [frame(0.0, -3.0)] * 3, t0=t)
    assert math.dist(a.root, before) < 0.05


def test_a_real_move_is_taken_after_it_persists():
    a = LiveAvatar(H)
    t = run(a, [frame(-1.5, -3.0)] * 15)
    run(a, [frame(1.5, -3.0)] * 10, t0=t)  # they really are over there now
    assert math.dist(a.root, (1.5, -3.0)) < 0.2


def test_posture_changes_wait_then_blend():
    a = LiveAvatar(H)
    t = run(a, [frame(0.0, -3.0)] * 10)
    lying = frame(0.0, -3.0, pose="side_lying")
    t = run(a, [lying] * 2, posture="floor", t0=t)  # 0.2 s: not yet
    assert a.posture == "standing"
    t = run(a, [lying] * 3, posture="floor", t0=t)  # past POSTURE_HOLD_S: blending
    assert a.posture == "floor"
    neck_mid = a.joints()["neck"][1]
    t = run(a, [lying] * int((BLEND_S + POSTURE_HOLD_S) / DT + 2), posture="floor", t0=t)
    neck_end = a.joints()["neck"][1]
    assert neck_end < 0.4 < neck_mid or neck_mid > neck_end  # still coming down mid-blend, on the floor after


def test_a_raised_arm_is_followed_and_forgotten_when_unseen():
    raised = dict(KEY_POSES["standing"])
    sh = raised["left_shoulder"]
    raised["left_elbow"] = (sh[0], sh[1] + 0.18, sh[2] + 0.03)
    raised["left_wrist"] = (sh[0], sh[1] + 0.34, sh[2] + 0.04)
    a = LiveAvatar(H)
    t = run(a, [frame(0.0, -3.0, body_pose=raised)] * 15)
    j = a.joints()
    assert j["left_wrist"][1] > j["left_shoulder"][1] + 0.3  # the arm is up
    assert j["right_wrist"][1] < j["right_shoulder"][1]  # the other one isn't
    no_arm = [k for k in frame(0.0, -3.0) if not k[0].startswith("left_") or k[0] == "left_hip"]
    run(a, [no_arm] * 12, t0=t)  # 1.2 s without seeing that arm
    j = a.joints()
    assert j["left_wrist"][1] < j["left_shoulder"][1]  # back down by its side


def test_heading_follows_a_turn():
    a = LiveAvatar(H)
    run(a, [frame(0.0, -3.0, heading=0.0)] * 15)
    assert abs(math.remainder(a.heading, 2 * math.pi)) < 0.3
    a2 = LiveAvatar(H)
    run(a2, [frame(0.0, -3.0, heading=math.pi)] * 30)
    assert abs(math.remainder(a2.heading - math.pi, 2 * math.pi)) < 0.4


@pytest.mark.parametrize("posture", ["standing", "sitting", "floor"])
def test_every_posture_gives_a_whole_body(posture):
    pose = {"standing": "standing", "sitting": "sitting", "floor": "side_lying"}[posture]
    a = LiveAvatar(H)
    run(a, [frame(0.0, -3.0, pose=pose)] * 20, posture=posture)
    assert set(KEY_POSES["standing"]) <= set(a.joints())

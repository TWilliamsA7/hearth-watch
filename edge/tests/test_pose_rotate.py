"""Rotation mapping tests (poseNet on a rotated frame, keypoints mapped back).

Run on the laptop: .venv/bin/python -m pytest edge/tests
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hearth_edge import features as f  # noqa: E402
from hearth_edge import pose_rotate as pr  # noqa: E402

W, H = 640, 480
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("k", [1, -1])
def test_mapping_matches_numpy_rot90(k):
    # Mark single pixels in an image, rotate it with numpy, and check where they land.
    img = np.zeros((H, W), dtype=np.uint16)
    points = [(0, 0), (W - 1, 0), (0, H - 1), (123, 45), (600, 470)]
    for i, (x, y) in enumerate(points, 1):
        img[y, x] = i
    rot = np.rot90(img, k)
    for i, (x, y) in enumerate(points, 1):
        xr, yr = pr.rotate_point(x, y, k, W, H)
        assert rot[yr, xr] == i
        assert pr.unrotate_point(xr, yr, k, W, H) == (x, y)


def test_unrotate_pose_turns_an_upright_skeleton_into_a_lying_one():
    # In the rotated (portrait, 480x640) frame poseNet sees an upright person...
    upright_in_rotated = {
        "keypoints": {"nose": (240, 100), "neck": (240, 160), "left_hip": (230, 330), "right_hip": (250, 330)},
        "bbox": (200, 80, 280, 560),
    }
    assert f.torso_angle_deg(upright_in_rotated["keypoints"]) == pytest.approx(0.0, abs=0.5)
    # ...which, mapped back to the original landscape frame, is a person lying sideways.
    lying = pr.unrotate_pose(upright_in_rotated, 1, W, H)
    assert f.torso_angle_deg(lying["keypoints"]) == pytest.approx(90.0, abs=0.5)
    left, top, right, bottom = lying["bbox"]
    assert right - left > bottom - top  # wide box, as for someone lying down
    assert f.bbox_aspect(lying["bbox"]) == pytest.approx(480 / 80.0)


def test_unrotate_point_rejects_other_angles():
    with pytest.raises(ValueError):
        pr.unrotate_point(1, 1, 2, W, H)


# --- when to rotate, which way, and when to trust the result ----------------------------

STANDING = {"keypoints": {"nose": (320, 60), "neck": (320, 110), "left_hip": (310, 250),
                          "right_hip": (330, 250), "left_ankle": (310, 440)},
            "bbox": (280, 50, 360, 445)}
# What plain poseNet found in the 2026-10-10 screenshot: head on the right, shoulders, no hips.
HEAD_ONLY_LYING = {"keypoints": {"nose": (1000, 395), "left_eye": (995, 385), "neck": (930, 490),
                                 "left_shoulder": (880, 440), "right_shoulder": (985, 535)},
                   "bbox": (880, 380, 1070, 540)}


def test_standing_needs_no_rotation():
    assert not pr.looks_lying_or_incomplete(STANDING)
    assert pr.RotationPolicy().directions(STANDING, 0.0) == []


def test_partial_or_lying_pose_triggers_rotation_head_side_first():
    assert pr.looks_lying_or_incomplete(HEAD_ONLY_LYING)  # no hips
    assert pr.preferred_direction(HEAD_ONLY_LYING) == 1    # head right of neck -> CCW
    mirrored = {"keypoints": {"nose": (100, 395), "neck": (170, 490)}, "bbox": (90, 380, 200, 500)}
    assert pr.preferred_direction(mirrored) == -1           # head left -> CW
    assert pr.RotationPolicy().directions(HEAD_ONLY_LYING, 0.0) == [1, -1]


def test_empty_room_tries_one_direction_once_a_second_alternating():
    pol = pr.RotationPolicy()
    tries = [(t / 10.0, pol.directions(None, t / 10.0)) for t in range(30)]
    tries = [(t, d) for t, d in tries if d]
    assert tries == [(0.0, [-1]), (1.0, [1]), (2.0, [-1])]


def test_after_an_accepted_rotation_every_frame_is_tried():
    pol = pr.RotationPolicy()
    pol.directions(None, 0.0)
    pol.accepted(-1, 0.0)
    assert pol.directions(None, 0.1) == [-1, 1]   # remembers the direction that worked
    assert pol.directions(None, 0.2) == [-1, 1]
    assert len(pol.directions(None, 1.5)) == 1    # beyond KEEP_TRYING_S: back to the slow
    assert pol.directions(None, 1.6) == []        # empty-room rhythm


def test_rotated_pose_must_clearly_beat_the_plain_one():
    full_lying = {"keypoints": dict(HEAD_ONLY_LYING["keypoints"], left_hip=(600, 470), right_hip=(600, 560),
                                    left_knee=(350, 480), right_knee=(350, 590), left_ankle=(60, 500),
                                    right_ankle=(60, 600)),
                  "bbox": (40, 380, 1070, 610)}
    assert pr.better_rotated(HEAD_ONLY_LYING, full_lying)
    assert pr.better_rotated(None, full_lying)              # nothing found upright
    ghost = {"keypoints": {"nose": (10, 10), "left_eye": (12, 8)}, "bbox": (5, 5, 20, 20)}
    assert not pr.better_rotated(None, ghost)               # empty-room hallucination
    no_torso = {"keypoints": dict((k, (i, i)) for i, k in enumerate(
        ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist"])), "bbox": (0, 0, 10, 10)}
    assert not pr.better_rotated(None, no_torso)            # many points but no torso
    elsewhere = dict(full_lying, bbox=(2000, 2000, 2100, 2100))
    assert not pr.better_rotated(HEAD_ONLY_LYING, elsewhere)  # a different person


def test_scale_pose_doubles_coordinates():
    p = pr.scale_pose({"keypoints": {"neck": (10, 20)}, "bbox": (1, 2, 3, 4)}, 2.0)
    assert p == {"keypoints": {"neck": (20.0, 40.0)}, "bbox": (2.0, 4.0, 6.0, 8.0)}


def test_half_size_rotation_maps_back_to_full_frame():
    # A point at full-frame (400, 300) is at (200, 150) in the half-size frame; rotated there
    # and mapped back with the half-size dimensions, then scaled x2, it must return to (400, 300).
    xr, yr = pr.rotate_point(200, 150, 1, W // 2, H // 2)
    back = pr.scale_pose(pr.unrotate_pose({"keypoints": {"a": (xr, yr)}, "bbox": (xr, yr, xr, yr)},
                                          1, W // 2, H // 2), 2.0)
    assert back["keypoints"]["a"] == (400.0, 300.0)


def test_upright_pass_skipped_on_most_frames_only_while_lying():
    pol = pr.RotationPolicy()
    assert not pol.skip_upright(0.0)               # nobody lying: always run it
    pol.accepted(1, 0.0)
    pattern = [pol.skip_upright(0.1 * i) for i in range(1, 7)]
    assert pattern == [True, True, False, True, True, False]  # upright on 1 frame in 3
    assert not pol.skip_upright(2.0)               # rotated pose no longer accepted: run it


def test_backs_off_after_both_directions_lose_to_the_plain_pose():
    """Seated near the camera (hips hidden): incomplete every frame, rotation never better."""
    pol = pr.RotationPolicy()
    tried = []
    for i in range(14):                                  # 1 s at 14 FPS
        t = i / 14.0
        d = pol.directions(HEAD_ONLY_LYING, t)
        if d:
            tried.append(t)
            pol.rejected(t)
    assert len(tried) == 2                               # t = 0 and ~0.5 s, not 14 times


def test_no_back_off_while_a_rotated_pose_keeps_being_accepted():
    pol = pr.RotationPolicy()
    pol.accepted(1, 0.0)
    pol.rejected(0.1)                                    # one miss while lying down...
    assert pol.directions(HEAD_ONLY_LYING, 0.2) == [1, -1]   # ...still tries the next frame
    pol.rejected(1.1)                                    # a miss after the accept went stale
    assert pol.directions(HEAD_ONLY_LYING, 1.2) == []        # ...does back off

"""Posture feature tests. Run on the laptop: .venv/bin/python -m pytest edge/tests"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hearth_edge import features as f  # noqa: E402

W, H = 640, 480

STANDING = {
    "keypoints": {
        "nose": (320, 60), "left_eye": (315, 55), "right_eye": (325, 55),
        "neck": (320, 110), "left_shoulder": (290, 110), "right_shoulder": (350, 110),
        "left_hip": (305, 250), "right_hip": (335, 250),
        "left_ankle": (305, 440), "right_ankle": (335, 440),
    },
    "bbox": (280, 50, 360, 445),
}

LYING = {
    "keypoints": {
        "nose": (100, 400), "neck": (160, 405),
        "left_hip": (320, 410), "right_hip": (320, 400),
        "left_ankle": (520, 410),
    },
    "bbox": (90, 380, 530, 430),
}


def test_standing_is_upright():
    assert f.torso_angle_deg(STANDING["keypoints"]) == pytest.approx(0.0, abs=0.5)
    assert f.bbox_aspect(STANDING["bbox"]) < 0.5


def test_lying_is_horizontal():
    assert f.torso_angle_deg(LYING["keypoints"]) == pytest.approx(90.0, abs=5.0)
    assert f.bbox_aspect(LYING["bbox"]) > 2.0


def test_angle_uses_pixels_not_normalized_coords():
    # 45 degrees in pixels; normalizing by 640x480 first would report about 37 degrees.
    kps = {"neck": (100, 100), "left_hip": (200, 200)}
    assert f.torso_angle_deg(kps) == pytest.approx(45.0, abs=0.01)


def test_neck_falls_back_to_shoulder_midpoint():
    kps = dict(STANDING["keypoints"])
    del kps["neck"]
    assert f.neck_point(kps) == (320.0, 110.0)
    assert f.torso_angle_deg(kps) == pytest.approx(0.0, abs=0.5)


def test_missing_neck_and_one_shoulder_gives_no_angle():
    kps = {"left_shoulder": (290, 110), "left_hip": (305, 250)}
    assert f.torso_angle_deg(kps) is None


def test_missing_hips_omits_hip_features():
    pose = {"keypoints": {"nose": (320, 60), "neck": (320, 110)}, "bbox": (300, 50, 340, 120)}
    out = f.compute_features(pose, W, H, f.VerticalVelocity(), t=0.0)
    assert "torso_angle_deg" not in out
    assert "hip_y" not in out
    assert "vertical_velocity" not in out
    assert out["head_y"] == pytest.approx(60 / 480.0, abs=1e-3)


def test_face_only_pose_omits_bbox_aspect():
    # Measured on the Jetson: a face-only pose has a wide, short box (aspect 2.4-4.6),
    # which would look like lying down. It must not be reported.
    face = {
        "keypoints": {"nose": (320, 240), "left_eye": (305, 230), "right_eye": (335, 230),
                      "left_ear": (285, 240), "right_ear": (355, 240)},
        "bbox": (285, 228, 355, 246),
    }
    out = f.compute_features(face, W, H, f.VerticalVelocity(), t=0.0)
    assert "bbox_aspect" not in out
    assert out["head_y"] == pytest.approx(0.49, abs=0.01)


def test_torso_visible_needs_upper_and_lower_body():
    assert f.torso_visible({"neck": (0, 0), "left_hip": (0, 1)})
    assert f.torso_visible({"right_shoulder": (0, 0), "right_hip": (0, 1)})
    assert not f.torso_visible({"neck": (0, 0), "left_shoulder": (0, 0)})
    assert not f.torso_visible({"left_hip": (0, 1), "left_knee": (0, 2)})
    assert f.compute_features(LYING, W, H, f.VerticalVelocity(), 0.0)["bbox_aspect"] > 2.0


def test_heights_are_normalized_and_clamped():
    assert f.hip_y(STANDING["keypoints"], H) == pytest.approx(250 / 480.0)
    assert f.head_y({"nose": (10, -5)}, H) == 0.0
    assert f.hip_y({"left_hip": (10, 999)}, H) == 1.0


def test_velocity_positive_when_moving_down():
    v = f.VerticalVelocity(alpha=1.0)
    assert v.update(0.40, 0.0) is None          # first sample has nothing to compare to
    assert v.update(0.50, 0.1) == pytest.approx(1.0)   # 0.1 down in 0.1 s
    assert v.update(0.45, 0.2) == pytest.approx(-0.5)  # moving back up


def test_velocity_smoothing_damps_jitter():
    v = f.VerticalVelocity(alpha=0.5)
    v.update(0.50, 0.0)
    assert abs(v.update(0.52, 0.1)) < abs((0.52 - 0.50) / 0.1)


def test_velocity_resets_after_gap_or_missing_hips():
    v = f.VerticalVelocity(alpha=1.0, reset_gap_s=1.0)
    v.update(0.30, 0.0)
    assert v.update(0.90, 2.0) is None   # reappeared after 2 s: no fake fall
    assert v.update(None, 2.1) is None
    assert v.update(0.20, 2.2) is None   # first sample after losing hips


def test_select_pose_picks_largest():
    small = {"keypoints": {}, "bbox": (0, 0, 10, 10)}
    assert f.select_pose([small, STANDING]) is STANDING
    assert f.select_pose([]) is None

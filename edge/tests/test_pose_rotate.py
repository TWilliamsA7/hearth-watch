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

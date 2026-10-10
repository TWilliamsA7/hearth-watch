import ast
from pathlib import Path

import pytest
from hearth.sim import features
from hearth.sim.features import compute_features


def kp(name, x, y, confidence=0.9):
    return {"name": name, "x": x, "y": y, "confidence": confidence}


UPRIGHT = [kp("nose", 0.5, 0.2), kp("neck", 0.5, 0.3), kp("left_hip", 0.52, 0.6), kp("right_hip", 0.48, 0.6)]


def test_upright_torso_is_zero_degrees():
    f = compute_features(UPRIGHT, frame_aspect=4 / 3)
    assert f["torso_angle_deg"] == pytest.approx(0.0)
    assert f["head_y"] == 0.2
    assert f["hip_y"] == 0.6


def test_horizontal_torso_is_ninety_degrees():
    pts = [kp("neck", 0.3, 0.8), kp("left_hip", 0.6, 0.8), kp("right_hip", 0.6, 0.8)]
    assert compute_features(pts, frame_aspect=4 / 3)["torso_angle_deg"] == pytest.approx(90.0)


def test_angle_uses_pixel_proportions():
    # 0.3 normalized x on a 4:3 frame is 0.4 frame-heights: atan(0.4 / 0.4) = 45 degrees.
    pts = [kp("neck", 0.2, 0.4), kp("left_hip", 0.5, 0.8), kp("right_hip", 0.5, 0.8)]
    assert compute_features(pts, frame_aspect=4 / 3)["torso_angle_deg"] == pytest.approx(45.0)


def test_neck_falls_back_to_shoulder_midpoint():
    pts = [kp("left_shoulder", 0.55, 0.3), kp("right_shoulder", 0.45, 0.3), kp("left_hip", 0.5, 0.6)]
    assert compute_features(pts, frame_aspect=1.0)["torso_angle_deg"] == pytest.approx(0.0)


def test_low_confidence_points_are_ignored():
    pts = UPRIGHT[:2] + [kp("left_hip", 0.9, 0.3, confidence=0.1)]
    f = compute_features(pts, frame_aspect=1.0)
    assert "torso_angle_deg" not in f and "hip_y" not in f


def test_vertical_velocity_positive_when_moving_down():
    f = compute_features(UPRIGHT, frame_aspect=1.0, prev_hip_y=0.5, dt=0.5)
    assert f["vertical_velocity"] == pytest.approx(0.2)
    assert "vertical_velocity" not in compute_features(UPRIGHT, frame_aspect=1.0)


def test_bbox_aspect_in_pixels():
    pts = [kp("nose", 0.4, 0.2), kp("left_ankle", 0.55, 0.8), kp("right_ankle", 0.45, 0.8)]
    assert compute_features(pts, frame_aspect=2.0)["bbox_aspect"] == pytest.approx(0.15 * 2.0 / 0.6)


def test_nothing_measurable_returns_empty():
    assert compute_features([], frame_aspect=1.0) == {}


def test_module_stays_edge_compatible():
    """The edge (JetPack Python 3.6, stdlib only) must be able to use a copy of this file."""
    tree = ast.parse(Path(features.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            assert all(n.split(".")[0] in {"math"} for n in names), names
        assert not isinstance(node, (ast.AnnAssign, ast.NamedExpr, ast.Match))
        if isinstance(node, ast.FunctionDef):
            assert node.returns is None and all(a.annotation is None for a in node.args.args)
        if isinstance(node, ast.JoinedStr):
            pytest.fail("f-strings: keep this file trivially 3.6-safe")

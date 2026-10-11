"""Our posture model's Jetson side (edge/hearth_edge/posture_mlp.py).

Run on the laptop: .venv/bin/python -m pytest edge/tests
The exact match with training (PyTorch) is checked by edge/pc/08_export_weights.py on the PC,
which runs our recordings through this module; these tests check its rules without PyTorch.
"""

import os
import sys

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hearth_edge import posture_mlp as pm  # noqa: E402

STANDING = {"nose": (320, 60), "left_eye": (315, 55), "right_eye": (325, 55),
            "left_shoulder": (300, 110), "right_shoulder": (340, 110), "left_hip": (305, 250),
            "right_hip": (335, 250), "left_knee": (305, 340), "right_knee": (335, 340),
            "left_ankle": (305, 430), "right_ankle": (335, 430), "neck": (320, 110)}


def lying():
    """STANDING turned 90 degrees, head to the right, low in the frame."""
    return dict((n, (100 + (y - 50) * 1.0, 400 + (x - 320) * 0.3)) for n, (x, y) in STANDING.items())


@pytest.fixture
def model_file(tmp_path):
    rng = np.random.RandomState(0)
    path = str(tmp_path / "m.npz")
    np.savez(path, W1=rng.randn(64, 51).astype(np.float32), b1=np.zeros(64, np.float32),
             W2=rng.randn(64, 64).astype(np.float32) * 0.1, b2=np.zeros(64, np.float32),
             W3=rng.randn(3, 64).astype(np.float32) * 0.1, b3=np.zeros(3, np.float32),
             classes=np.array(["upright", "sitting", "floor"]), min_conf=np.float32(0.5),
             min_found=np.int64(4), keypoint_names=np.array(pm.KEYPOINT_NAMES),
             source_model=np.array("test.pt"))
    return path


def test_features_shape_and_neck_ignored(model_file):
    m = pm.PostureMLP(model_file)
    x = m.features(STANDING, 640, 480)
    assert x.shape == (51,) and x.dtype == np.float32
    assert x.reshape(17, 3)[:, 2].sum() == 11            # 11 COCO keypoints found; neck ignored
    assert x.reshape(17, 3)[3, 2] == 0 and x.reshape(17, 3)[3, 0] == 0   # left_ear missing -> 0


def test_features_ignore_where_and_how_big(model_file):
    m = pm.PostureMLP(model_file)
    x = m.features(STANDING, 640, 480)
    moved = dict((n, (px + 120, py - 20)) for n, (px, py) in STANDING.items())
    half = dict((n, (200 + px / 2.0, 100 + py / 2.0)) for n, (px, py) in STANDING.items())
    assert np.allclose(m.features(moved, 640, 480), x, atol=1e-5)
    assert np.allclose(m.features(half, 640, 480), x, atol=1e-5)


def test_features_undo_the_frame_shape(model_file):
    """The same body in pixels gives the same inputs whatever the frame's width."""
    m = pm.PostureMLP(model_file)
    assert np.allclose(m.features(STANDING, 640, 480), m.features(STANDING, 960, 480), atol=1e-5)


def test_lying_and_standing_differ_in_orientation(model_file):
    m = pm.PostureMLP(model_file)
    up, down = m.features(STANDING, 640, 480).reshape(17, 3), m.features(lying(), 640, 480).reshape(17, 3)
    assert np.ptp(up[up[:, 2] > 0, 1]) > np.ptp(up[up[:, 2] > 0, 0])      # taller than wide
    assert np.ptp(down[down[:, 2] > 0, 0]) > np.ptp(down[down[:, 2] > 0, 1])  # wider than tall


def test_predict_too_few_keypoints_and_probabilities(model_file):
    m = pm.PostureMLP(model_file)
    assert m.predict({"nose": (1, 1), "left_eye": (2, 2), "right_eye": (3, 3)}, 640, 480) is None
    r = m.predict(STANDING, 640, 480)
    assert r["label"] in ("upright", "sitting", "floor", "unknown")
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-5
    assert r["confidence"] == pytest.approx(max(r["probabilities"].values()))


def test_unknown_under_the_cut_off(model_file):
    m = pm.PostureMLP(model_file)
    m.min_conf = 0.999
    assert m.predict(STANDING, 640, 480)["label"] == "unknown"
    m.min_conf = 0.0
    assert m.predict(STANDING, 640, 480)["label"] != "unknown"


def test_missing_model_file_says_how_to_make_it(tmp_path):
    with pytest.raises(IOError, match="08_export_weights"):
        pm.PostureMLP(str(tmp_path / "nope.npz"))


@pytest.mark.skipif(not os.path.exists(pm.default_path()), reason="export the real model first (08)")
def test_real_model_tells_standing_from_lying():
    m = pm.PostureMLP()
    assert m.predict(STANDING, 640, 480)["label"] == "upright"
    assert m.predict(lying(), 640, 480)["label"] == "floor"

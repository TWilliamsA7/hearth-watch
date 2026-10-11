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


# --- sending our model to the backend --------------------------------------------------------

def _pose(kps):
    xs, ys = [p[0] for p in kps.values()], [p[1] for p in kps.values()]
    return {"keypoints": kps, "bbox": (min(xs), min(ys), max(xs), max(ys))}


def test_backend_prediction_passes_the_contract_in_every_case(model_file):
    hs = pytest.importorskip("hearth_schema")
    m = pm.PostureMLP(model_file)
    up = {"label": "upright", "confidence": 0.9, "probabilities": {}}
    down = {"label": "floor", "confidence": 0.8, "probabilities": {}}
    unsure = {"label": "unknown", "confidence": 0.4, "probabilities": {}}
    cases = {
        "nobody": [],
        "tracked standing": [(_pose(STANDING), True, up)],
        "too few keypoints": [(_pose(STANDING), True, None)],
        "unsure": [(_pose(STANDING), True, unsure)],
        "someone else down": [(_pose(STANDING), True, up), (_pose(lying()), False, down)],
    }
    got = {}
    for name, people in cases.items():
        p = m.backend_prediction(people, 640, 480)
        hs.PosturePrediction.model_validate(p)
        got[name] = (p["label"], p["detection_count"], p["model_id"])
    assert got["nobody"] == ("unknown", 0, pm.MODEL_ID)
    assert got["tracked standing"][0] == "standing" and got["too few keypoints"][0] == "unknown"
    assert got["unsure"][0] == "unknown" and got["someone else down"][0] == "fallen"


def test_team_backend_config_trusts_our_model():
    import json
    team = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "hearth", "monitor", "config.json")
    if not os.path.exists(team):
        pytest.skip("backend config not in this checkout")
    config = json.load(open(team))
    assert config["fall_model_id"] == pm.MODEL_ID and config["model_confidence_min"] == 0.5


def test_backend_alerts_and_recovers_on_our_model(model_file):
    """Person 2's real FallDetector with the team's default config: 4 s of our "fallen" raises
    possible_fall; then 2 s of "standing" records recovered."""
    detector = pytest.importorskip("hearth.monitor.detector")
    hs = pytest.importorskip("hearth_schema")
    from hearth.monitor.config import MonitorConfig
    det = detector.FallDetector(MonitorConfig.load())
    from hearth_edge import contract
    m = pm.PostureMLP(model_file)
    ids, events, t0 = contract.ObservationIds("jetson-01", start=1791700000), [], 1791700000.0
    for i in range(60):                                    # 10 Hz: 4 s down, then 2 s up
        t = t0 + i * 0.1
        result = ({"label": "floor", "confidence": 0.9, "probabilities": {}} if i < 40 else
                  {"label": "upright", "confidence": 0.9, "probabilities": {}})
        pose = _pose(lying() if i < 40 else STANDING)
        pred = m.backend_prediction([(pose, True, result)], 640, 480)
        obs = contract.build_observation(ids.next(), t, "jetson-01", "room-01", person_detected=True,
                                         posture_prediction=pred)
        events += [e.event_type for e in det.process(hs.Observation.model_validate(obs))]
    assert events.count("possible_fall") == 1 and events.count("recovered") == 1


def test_person_prediction_is_what_backend_prediction_sends_for_one_person(model_file):
    m = pm.PostureMLP(model_file)
    up = {"label": "upright", "confidence": 0.9, "probabilities": {}}
    pose = _pose(STANDING)
    assert m.person_prediction(pose, up, 640, 480) == m.backend_prediction([(pose, True, up)], 640, 480)


def test_backend_watches_each_person_separately(model_file):
    """Two people from one camera, each with their own person_id and our model's call: only the
    one on the floor raises possible_fall, and that event carries their person_id."""
    detector = pytest.importorskip("hearth.monitor.detector")
    hs = pytest.importorskip("hearth_schema")
    from hearth.monitor.config import MonitorConfig
    from hearth_edge import contract
    det = detector.FallDetector(MonitorConfig.load())
    m = pm.PostureMLP(model_file)
    ids, events, t0 = contract.ObservationIds("jetson-01", start=1791700000), [], 1791700000.0
    up = {"label": "upright", "confidence": 0.9, "probabilities": {}}
    down = {"label": "floor", "confidence": 0.9, "probabilities": {}}
    for i in range(40):                                    # 10 Hz for 4 s
        t = t0 + i * 0.1
        for person_id, pose, result in (("p1", _pose(STANDING), up), ("p2", _pose(lying()), down)):
            obs = contract.build_observation(ids.next(), t, "jetson-01", "room-01", person_detected=True,
                                             posture_prediction=m.person_prediction(pose, result, 640, 480),
                                             person_id=person_id)
            events += [(e.event_type, e.person_id) for e in det.process(hs.Observation.model_validate(obs))]
    assert [e for e in events if e[0] == "possible_fall"] == [("possible_fall", "p2")]

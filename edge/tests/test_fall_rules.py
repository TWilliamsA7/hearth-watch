"""The Jetson port of Person 2's posture_prediction rules must agree with their real function.

Run on the laptop: .venv/bin/python -m pytest edge/tests
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hearth_edge import fall_rules as port  # noqa: E402

real = pytest.importorskip("hearth.perception.fall_model")
hs = pytest.importorskip("hearth_schema")

W, H = 640, 480


def real_prediction(detections):
    settings = real.ModelSettings.load()
    p = real.prediction_from_detections(detections, W, H, settings)
    return p.model_dump(mode="json")


def test_constants_match_person_2s_model_json():
    settings = real.ModelSettings.load()
    assert port.MODEL_ID == settings.repo_id
    assert port.MODEL_REVISION == settings.revision
    assert port.DETECTION_CONFIDENCE_MIN == settings.detection_confidence_min


@pytest.mark.parametrize("detections", [
    [],                                                                          # nobody
    [{"label": "fallen", "confidence": 0.91, "xyxy": [100, 300, 500, 420]}],     # one fallen
    [{"label": "Standing", "confidence": 0.8, "xyxy": [200, 20, 300, 460]}],     # case/name mapping
    [{"label": "standing", "confidence": 0.8, "xyxy": [0, 0, 50, 50]},
     {"label": "sitting", "confidence": 0.7, "xyxy": [300, 100, 400, 300]}],     # two people
    [{"label": "fallen", "confidence": 0.2, "xyxy": [100, 300, 500, 420]}],      # below 0.25
    [{"label": "dancing", "confidence": 0.9, "xyxy": [10, 10, 100, 200]}],       # unsupported label
    [{"label": "fallen", "confidence": 0.9, "xyxy": [-50, 400, 700, 600]}],      # clamped to frame
    [{"label": "fallen", "confidence": 0.9, "xyxy": [700, 10, 800, 50]}],        # entirely outside
])
def test_port_matches_person_2s_function(detections):
    assert port.prediction_from_detections(detections, W, H) == real_prediction(detections)


def test_port_matches_on_random_detections():
    rng = random.Random(7)
    labels = ["fallen", "sitting", "standing", "Fall", "falling", "other"]
    for _ in range(300):
        dets = []
        for _ in range(rng.randint(0, 3)):
            x1, y1 = rng.uniform(-100, 700), rng.uniform(-100, 550)
            dets.append({"label": rng.choice(labels), "confidence": rng.random(),
                         "xyxy": [x1, y1, x1 + rng.uniform(-20, 400), y1 + rng.uniform(-20, 300)]})
        assert port.prediction_from_detections(dets, W, H) == real_prediction(dets)


def test_port_output_passes_the_contract():
    for dets in ([], [{"label": "fallen", "confidence": 0.9, "xyxy": [100, 300, 500, 420]}]):
        hs.PosturePrediction.model_validate(port.prediction_from_detections(dets, W, H))


def test_invalid_detections_raise_like_the_original():
    with pytest.raises(ValueError):
        port.prediction_from_detections([{"label": "fallen", "confidence": 1.5, "xyxy": [0, 0, 1, 1]}], W, H)
    with pytest.raises(ValueError):
        port.prediction_from_detections([], 0, H)

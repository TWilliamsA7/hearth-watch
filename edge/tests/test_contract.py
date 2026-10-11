"""Edge-built messages must pass the real shared contract (Pydantic, extra='forbid').

Run on the laptop: .venv/bin/python -m pytest edge/tests
"""

import os
import re
import sys
import typing

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hearth_edge import contract as c  # noqa: E402
from hearth_edge import features as f  # noqa: E402

hs = pytest.importorskip("hearth_schema")
from pydantic import ValidationError  # noqa: E402

W, H = 640, 480
T0 = 1791700000.123  # fixed time so results are repeatable
POSE = {
    "keypoints": {
        "nose": (320, 60), "neck": (320, 110),
        "left_shoulder": (290, 110), "right_shoulder": (350, 110),
        "left_hip": (305, 250), "right_hip": (335, 250),
    },
    "bbox": (280, 50, 360, 445),
}


def make_obs(**overrides):
    ids = c.ObservationIds("jetson-01", start=1791700000)
    args = dict(
        obs_id=ids.next(), ts=T0, source="jetson-01", room_id="room-01", person_detected=True,
        keypoints=c.normalize_keypoints(POSE["keypoints"], W, H, confidence=None),
        features=f.compute_features(POSE, W, H, f.VerticalVelocity(), T0),
        face_present=True, face_confidence=0.87,
    )
    args.update(overrides)
    return c.build_observation(**args)


def test_keypoint_names_match_contract():
    assert set(c.KEYPOINT_NAMES) == set(typing.get_args(hs.KeypointName))


def test_full_observation_passes_contract():
    obs = hs.Observation.model_validate(make_obs())
    assert obs.source_type == "edge"
    assert obs.features.torso_angle_deg == pytest.approx(0.0, abs=0.5)
    assert [k.name for k in obs.keypoints][0] == "nose"


def test_no_person_observation_passes_contract():
    obs = hs.Observation.model_validate(make_obs(person_detected=False, face_present=None,
                                                 face_confidence=None))
    assert obs.keypoints is None and obs.features is None


def test_face_null_and_false_are_kept_distinct():
    assert make_obs(face_present=None, face_confidence=None)["face_present"] is None
    assert make_obs(face_present=False, face_confidence=None)["face_present"] is False
    hs.Observation.model_validate(make_obs(face_present=False, face_confidence=None))


def test_batch_passes_contract():
    hs.ObservationBatch.model_validate({"observations": [make_obs(), make_obs()]})


def test_heartbeat_passes_contract():
    hb = c.build_heartbeat(T0, "jetson-01", "room-01", fps=14.1, inference_ms=57.4,
                           mem_used_mb=1403.0)
    assert hs.Heartbeat.model_validate(hb).fps == pytest.approx(14.1)


def test_keypoints_outside_frame_are_clamped():
    kps = c.normalize_keypoints({"nose": (-12, 500), "neck": (700, 10)}, W, H, confidence=None)
    assert kps[0] == {"name": "nose", "x": 0.0, "y": 1.0}
    assert kps[1]["x"] == 1.0
    hs.Observation.model_validate(make_obs(keypoints=kps))


def test_timestamp_format():
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$", c.utc_iso(T0))


def test_ids_are_unique_and_ordered():
    ids = c.ObservationIds("jetson-01", start=1791700000)
    a, b = ids.next(), ids.next()
    assert a == "jetson-01-1791700000-000001" and b == "jetson-01-1791700000-000002"


def test_keypoints_without_confidence_pass_contract():
    # poseNet measures no per-keypoint confidence, so none is sent (null = unknown in the
    # contract) instead of a made-up 1.0 the backend can't tell apart from a real score.
    from hearth_edge import config
    assert config.KEYPOINT_CONFIDENCE is None
    obs = make_obs()
    assert all("confidence" not in k for k in obs["keypoints"])
    model = hs.Observation.model_validate(obs)
    assert all(k.confidence is None for k in model.keypoints)


def test_run_id_is_optional_and_only_sent_when_given():
    live = make_obs()
    assert "run_id" not in live
    replayed = make_obs(run_id="replay-01-20261010-043512-1")
    assert hs.Observation.model_validate(replayed).run_id == "replay-01-20261010-043512-1"
    hb = c.build_heartbeat(T0, "replay-01", "room-01", fps=10.0, source_type="replay",
                           run_id="replay-01-20261010-043512-1")
    assert hs.Heartbeat.model_validate(hb).run_id == "replay-01-20261010-043512-1"
    assert "run_id" not in c.build_heartbeat(T0, "jetson-01", "room-01")


def test_person_id_is_optional_and_only_sent_when_given():
    assert "person_id" not in make_obs()
    assert hs.Observation.model_validate(make_obs(person_id="p2")).person_id == "p2"


def _schema(name):
    import json
    root = os.path.join(os.path.dirname(__file__), "..", "..", "shared", "schema")
    with open(os.path.join(root, name + ".schema.json")) as fh:
        return json.load(fh)


def _fits(message, schema, defs):
    """Required keys present and no unknown keys: what a strict JSON Schema validator checks."""
    assert set(schema.get("required", [])) <= set(message), sorted(set(schema["required"]) - set(message))
    assert set(message) <= set(schema["properties"]), sorted(set(message) - set(schema["properties"]))


def test_edge_messages_fit_the_exported_json_schema():
    # The schema files, not just Pydantic, must accept what the Jetson sends, so a
    # non-Python consumer validating against shared/schema agrees with the backend.
    obs_schema = _schema("observation")
    defs = obs_schema["$defs"]
    for obs in (make_obs(), make_obs(person_detected=False, keypoints=None, features=None,
                                     face_present=None, face_confidence=None)):
        _fits(obs, obs_schema, defs)
        for kp in obs["keypoints"] or []:
            _fits(kp, defs["Keypoint"], defs)
        if obs["features"]:
            _fits(obs["features"], defs["PostureFeatures"], defs)
    _fits(c.build_heartbeat(T0, "jetson-01", "room-01", fps=10.0), _schema("heartbeat"), {})


def test_floor_guide_matches_the_backend_threshold():
    import json
    from hearth_edge import config
    path = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "hearth", "monitor", "config.json")
    with open(path) as fh:
        assert config.FLOOR_GUIDE_Y == json.load(fh)["floor_hip_y_min"]

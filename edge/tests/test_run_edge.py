"""Tests for run_edge.py's pure helpers (the camera loop itself runs only on the Jetson).

Run on the laptop: .venv/bin/python -m pytest edge/tests
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import run_edge  # noqa: E402
from hearth_edge import config, contract, features  # noqa: E402

hs = pytest.importorskip("hearth_schema")

FRAME = {"t": 1791700000.5, "width": 640, "height": 480}
POSE = {
    "keypoints": {"nose": (320, 60), "neck": (320, 110), "left_hip": (305, 250),
                  "right_hip": (335, 250)},
    "bbox": (280, 50, 360, 445),
}


def test_rate_limiter_gives_10_per_second_from_14_fps():
    limiter = run_edge.RateLimiter(10.0)
    frames = [i / 14.0 for i in range(14 * 10)]  # 10 s of a 14 FPS camera
    sent = sum(1 for t in frames if limiter.ready(t))
    assert 98 <= sent <= 101  # not ~70, which "every frame >= 0.1 s apart" would give


def test_rate_limiter_does_not_burst_after_a_stall():
    limiter = run_edge.RateLimiter(10.0)
    for i in range(10):
        limiter.ready(i * 0.1)
    # camera stalls for 3 s, then frames resume at 14 FPS
    resumed = [3.0 + i / 14.0 for i in range(14)]
    assert sum(1 for t in resumed if limiter.ready(t)) <= 11


def test_observation_with_person_passes_contract():
    ids = contract.ObservationIds("jetson-01", start=1791700000)
    feats = features.compute_features(POSE, 640, 480, features.VerticalVelocity(), FRAME["t"])
    obs = run_edge.observation_for_frame(ids, FRAME, POSE, feats, "jetson-01", "room-01")
    model = hs.Observation.model_validate(obs)
    assert model.person_detected and model.face_present is None
    assert model.features.torso_angle_deg == pytest.approx(0.0, abs=0.5)
    assert len(model.keypoints) == 4


def test_observation_without_person_passes_contract():
    ids = contract.ObservationIds("jetson-01", start=1791700000)
    obs = run_edge.observation_for_frame(ids, FRAME, None, None, "jetson-01", "room-01")
    model = hs.Observation.model_validate(obs)
    assert not model.person_detected and model.keypoints is None and model.features is None


def test_defaults_come_from_config():
    args = run_edge.parse_args([])
    assert args.backend == config.BACKEND_URL
    assert args.source == config.SOURCE_ID and args.room == config.ROOM_ID
    assert args.obs_hz == config.OBS_HZ and not args.dry_run
    assert run_edge.parse_args(["--dry-run", "--obs-hz", "5"]).obs_hz == 5.0
    assert args.posture_mlp and not run_edge.parse_args(["--no-posture-mlp"]).posture_mlp
    assert run_edge.parse_args(["--posture-mlp"]).posture_mlp   # old spelling still accepted


def test_log_survives_a_closed_stderr(monkeypatch):
    # Ctrl+C over ssh can close our stderr pipe first; logging must not crash the shutdown.
    class Broken(object):
        def write(self, _):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(run_edge.sys, "stderr", Broken())
    run_edge.log("stopping")  # must not raise


# --- optional fall model (posture_prediction) -------------------------------------------

def fallen_prediction():
    from hearth_edge import fall_rules
    return fall_rules.prediction_from_detections(
        [{"label": "fallen", "confidence": 0.9, "xyxy": [100, 300, 500, 420]}], 640, 480)


def test_observation_carries_posture_prediction_and_passes_contract():
    ids = contract.ObservationIds("jetson-01", start=1791700000)
    feats = features.compute_features(POSE, 640, 480, features.VerticalVelocity(), FRAME["t"])
    obs = run_edge.observation_for_frame(ids, FRAME, POSE, feats, "jetson-01", "room-01",
                                         posture_prediction=fallen_prediction())
    model = hs.Observation.model_validate(obs)
    assert model.posture_prediction.label == "fallen" and model.keypoints


def test_model_detection_counts_as_person_when_posenet_lost_them():
    ids = contract.ObservationIds("jetson-01", start=1791700000)
    obs = run_edge.observation_for_frame(ids, FRAME, None, None, "jetson-01", "room-01",
                                         posture_prediction=fallen_prediction())
    model = hs.Observation.model_validate(obs)
    assert model.person_detected and model.keypoints is None
    from hearth_edge import fall_rules
    nobody = fall_rules.prediction_from_detections([], 640, 480)
    obs = run_edge.observation_for_frame(ids, FRAME, None, None, "jetson-01", "room-01",
                                         posture_prediction=nobody)
    assert obs["person_detected"] is False


def test_backend_alerts_on_model_fallen_even_when_pose_drops_out():
    """Person 2's real FallDetector with the team's default config: our posture model's "fallen"
    on every observation (poseNet losing the person on some frames) completes the 3 s hold."""
    mon = pytest.importorskip("hearth.monitor.detector")
    from hearth.monitor.config import MonitorConfig
    from hearth_edge import posture_mlp
    det = mon.FallDetector(MonitorConfig.load())
    ids = contract.ObservationIds("jetson-01", start=1791700000)
    pred = {"model_id": posture_mlp.MODEL_ID, "model_revision": "test", "input_processing": "original",
            "label": "fallen", "confidence": 0.9, "detection_count": 1, "uncertainty_reason": None,
            "bbox": {"x1": 0.1, "y1": 0.6, "x2": 0.8, "y2": 0.9}}
    kinds = []
    for i in range(45):                                   # 4.5 s at 10 Hz
        frame = dict(FRAME, t=FRAME["t"] + i * 0.1)
        pose = POSE if i % 3 else None                    # poseNet misses every third frame
        feats = (features.compute_features(pose, 640, 480, features.VerticalVelocity(), frame["t"])
                 if pose else None)
        obs = run_edge.observation_for_frame(ids, frame, pose, feats, "jetson-01", "room-01",
                                             posture_prediction=pred)
        kinds += [e.event_type for e in det.process(hs.Observation.model_validate(obs))]
    assert kinds.count("possible_fall") == 1

def test_people_in_view_drops_the_tracked_persons_plain_copy_and_sorts():
    def person(cx, tag):
        return {"keypoints": {"nose": (cx, 100), "tag": tag}, "bbox": (cx - 20, 50, cx + 20, 400)}
    tracked = person(305, "tracked, rotated pass")
    plain = [person(300, "tracked, plain pass"), person(100, "left"), person(550, "right")]
    got = run_edge.people_in_view(tracked, plain, max_dist=160)
    assert [p["keypoints"]["tag"] for p, _ in got] == ["left", "tracked, rotated pass", "right"]
    assert [t for _, t in got] == [False, True, False]
    assert [p["keypoints"]["tag"] for p, _ in run_edge.people_in_view(None, plain, 160)] == \
        ["left", "tracked, plain pass", "right"]
    # nobody near the tracked pose in the plain pass (e.g. upright pass skipped): nothing dropped
    assert len(run_edge.people_in_view(tracked, [person(100, "left")], 160)) == 2

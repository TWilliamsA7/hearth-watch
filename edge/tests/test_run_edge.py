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


def test_log_survives_a_closed_stderr(monkeypatch):
    # Ctrl+C over ssh can close our stderr pipe first; logging must not crash the shutdown.
    class Broken(object):
        def write(self, _):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(run_edge.sys, "stderr", Broken())
    run_edge.log("stopping")  # must not raise

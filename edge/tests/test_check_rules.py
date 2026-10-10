"""check_rules.py tests: the real backend detector's verdict plus per-condition explanations.

Run on the laptop: .venv/bin/python -m pytest edge/tests
"""

import importlib.util
import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from hearth_edge import contract as c  # noqa: E402

pytest.importorskip("hearth.monitor.detector")
from hearth.monitor.config import MonitorConfig  # noqa: E402

spec = importlib.util.spec_from_file_location("check_rules", os.path.join(HERE, "..", "tools", "check_rules.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

T0 = 1791700000.0
STAND = {"torso_angle_deg": 5.0, "head_y": 0.2, "hip_y": 0.5, "bbox_aspect": 0.35}
FLOOR = {"torso_angle_deg": 85.0, "head_y": 0.8, "hip_y": 0.82, "bbox_aspect": 2.4}
FACE_ONLY = {"head_y": 0.6}
KPS = c.normalize_keypoints({"nose": (320, 100), "neck": (320, 150), "left_hip": (310, 300)}, 640, 480, None)


def clip(tmp_path, segments, hz=10.0):
    """segments: [(seconds, features or None for no person)] -> path of a JSONL recording."""
    ids = c.ObservationIds("jetson-01", start=int(T0))
    t, lines = T0, []
    for seconds, feats in segments:
        for _ in range(int(seconds * hz)):
            lines.append(json.dumps(c.build_observation(
                ids.next(), t, "jetson-01", "room-01", feats is not None,
                keypoints=KPS if feats else None, features=feats)))
            t += 1.0 / hz
    path = tmp_path / "clip.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def run(path):
    out = io.StringIO()
    result = cr.report(path, MonitorConfig.load(), out=out)
    return result, out.getvalue()


def test_sustained_floor_gives_a_real_backend_alert_and_recovery(tmp_path):
    result, text = run(clip(tmp_path, [(1, STAND), (4, FLOOR), (1, STAND)]))
    kinds = [k for _, k, _ in result["events"]]
    assert kinds == ["possible_fall", "recovered"]
    alert_t = result["events"][0][0]
    assert 3.9 <= alert_t <= 4.2  # floor starts at 1.0 s, backend hold is 3 s
    assert result["floor"] == 40 and result["standing"] == 20
    assert "VERDICT: possible_fall" in text and "VERDICT: recovered" in text
    assert "SS" in text and "FFFFFFFF" in text  # timeline


def test_too_short_on_the_floor_gives_no_alert(tmp_path):
    result, text = run(clip(tmp_path, [(1, STAND), (2, FLOOR), (1, STAND)]))
    assert result["alerts"] == 0 and "VERDICT: no possible-fall alert" in text


def test_face_only_frames_explain_what_is_missing(tmp_path):
    result, text = run(clip(tmp_path, [(2, FACE_ONLY)]))
    assert result["alerts"] == 0 and result["standing"] == 0 and result["floor"] == 0
    assert result["fails"]["no torso angle (needs neck/shoulders + a hip)"] == 20
    assert result["fails"]["no hips"] == 20
    assert result["fails"]["head above the 0.65 line"] == 20
    assert "100%  no hips" in text


def test_lying_too_high_in_the_frame_is_explained(tmp_path):
    high = dict(FLOOR, head_y=0.5, hip_y=0.55)  # lying, but mid-frame: camera not aimed at floor
    result, text = run(clip(tmp_path, [(4, high)]))
    assert result["alerts"] == 0
    assert result["fails"] == {"head above the 0.65 line": 40, "hips above the 0.65 line": 40}


def test_gap_longer_than_backend_limit_resets_the_hold(tmp_path):
    result, _ = run(clip(tmp_path, [(2, FLOOR), (1.5, None), (2, FLOOR)]))
    # no-person frames break the floor streak, so neither 2 s stretch reaches the 3 s hold
    assert result["alerts"] == 0

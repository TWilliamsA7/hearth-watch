"""Simulated and live posture features must agree, or the simulator measures a different system.

Runs the real edge module (edge/hearth_edge/features.py, as edge/run_edge.py uses it per frame)
and hearth.sim.features on the same pose sequences, and requires the same numbers.
"""

import importlib.util
import random
from pathlib import Path

import pytest
from hearth.sim.features import FeatureTracker

EDGE = Path(__file__).resolve().parents[2] / "edge" / "hearth_edge" / "features.py"
W, H = 640, 480


def _edge_module():
    spec = importlib.util.spec_from_file_location("edge_features", EDGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


edge = _edge_module()

STANDING = {"nose": (320, 60), "left_eye": (314, 54), "right_eye": (326, 54), "neck": (320, 110),
            "left_shoulder": (290, 110), "right_shoulder": (350, 110),
            "left_hip": (305, 250), "right_hip": (335, 250),
            "left_knee": (305, 340), "right_knee": (335, 340),
            "left_ankle": (305, 430), "right_ankle": (335, 430)}
LYING = {"nose": (120, 400), "neck": (170, 405), "left_shoulder": (170, 395), "right_shoulder": (170, 415),
         "left_hip": (320, 410), "right_hip": (320, 425), "left_knee": (420, 412), "left_ankle": (520, 415)}


def _between(a, b, f):
    return {n: (a[n][0] + (b[n][0] - a[n][0]) * f, a[n][1] + (b[n][1] - a[n][1]) * f) for n in a if n in b}


def _bbox(kps):
    xs, ys = [p[0] for p in kps.values()], [p[1] for p in kps.values()]
    return (min(xs), min(ys), max(xs), max(ys))  # poseNet's box wraps the keypoints it found


def _sequence(seed):
    """A fall with jitter, dropped joints and a skipped frame: (t, pixel keypoints or None)."""
    rng = random.Random(seed)
    frames = []
    for i in range(40):
        t = i * 0.1
        kps = _between(STANDING, LYING, min(1.0, max(0.0, (t - 1.0) / 1.2)))
        kps = {n: (x + rng.gauss(0, 2), y + rng.gauss(0, 2)) for n, (x, y) in kps.items()
               if rng.random() > 0.15}
        frames.append((t, None if i in (25, 26) else kps))
    return frames


def _edge_run(frames):
    velocity, hold, out = edge.VerticalVelocity(), edge.KeypointHold(), []
    for t, kps in frames:
        if kps is None:
            velocity.reset()
            out.append(None)
            continue
        merged, _ = hold.update(kps, t)
        out.append(edge.compute_features({"keypoints": merged, "bbox": _bbox(kps)}, W, H, velocity, t))
    return out


def _sim_run(frames):
    tracker, out = FeatureTracker(W / H), []
    for t, kps in frames:
        if kps is None:
            tracker.lost()
            out.append(None)
            continue
        keypoints = [{"name": n, "x": x / W, "y": y / H, "confidence": None} for n, (x, y) in kps.items()]
        out.append(tracker.update(keypoints, t))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_sim_and_edge_compute_the_same_features(seed):
    frames = _sequence(seed)
    for (t, _), e, s in zip(frames, _edge_run(frames), _sim_run(frames)):
        if e is None:
            assert s is None
            continue
        assert set(e) == set(s), t
        for name in e:
            # Edge rounds pixel results, sim rounds normalized ones: a few 1e-4 apart at most.
            assert s[name] == pytest.approx(e[name], abs=2e-3), (t, name)

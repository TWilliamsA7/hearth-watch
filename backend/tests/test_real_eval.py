"""Recorded clips -> real monitor -> score against hand-labelled truth.

The clips here are written exactly as the Jetson records them (edge/hearth_edge/contract.py
and features.py, one bare Observation per line), from synthetic pixel poses: they test the
pipeline, not detection accuracy on camera data.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from hearth.eval.__main__ import main
from hearth.eval.real import evaluate_recording, sweep, timeline, timeline_text
from hearth.eval.recording import label_template, load_recording, truth_path, write_truth
from hearth.sim.runner import run_headless, write_run
from hearth.sim.scenario import load_all
from hearth_schema.models import GroundTruth

EDGE = Path(__file__).resolve().parents[2] / "edge"
W, H = 640, 480
T0 = 1791700000.0

STANDING = {"nose": (320, 60), "neck": (320, 110), "left_shoulder": (290, 110), "right_shoulder": (350, 110),
            "left_hip": (305, 250), "right_hip": (335, 250), "left_knee": (305, 340), "left_ankle": (305, 430)}
LYING = {"nose": (120, 400), "neck": (170, 405), "left_shoulder": (170, 395), "right_shoulder": (170, 415),
         "left_hip": (320, 410), "right_hip": (320, 425), "left_knee": (420, 412), "left_ankle": (520, 415)}


def _edge(name):
    sys.path.insert(0, str(EDGE))
    try:
        spec = importlib.util.spec_from_file_location(f"hearth_edge.{name}", EDGE / "hearth_edge" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EDGE))


contract, features = _edge("contract"), _edge("features")


def _pose(a, b, f):
    kps = {n: (a[n][0] + (b[n][0] - a[n][0]) * f, a[n][1] + (b[n][1] - a[n][1]) * f) for n in a}
    xs, ys = [p[0] for p in kps.values()], [p[1] for p in kps.values()]
    return {"keypoints": kps, "bbox": (min(xs), min(ys), max(xs), max(ys))}


def write_edge_clip(path: Path, *, down_at: float | None, seconds: float = 12.0) -> Path:
    """Stand, then (if down_at) go down over 1 s and stay down. 10 Hz, as run_edge.py records."""
    ids, velocity, lines = contract.ObservationIds("jetson-01", start=T0), features.VerticalVelocity(), []
    for i in range(int(seconds * 10)):
        t = T0 + i / 10
        f = 0.0 if down_at is None else min(1.0, max(0.0, (i / 10 - down_at) / 1.0))
        pose = _pose(STANDING, LYING, f)
        obs = contract.build_observation(
            ids.next(), t, "jetson-01", "room-01", True,
            keypoints=contract.normalize_keypoints(pose["keypoints"], W, H, confidence=None),
            features=features.compute_features(pose, W, H, velocity, t),
            face_present=(True if i % 5 == 0 else None), face_confidence=(0.8 if i % 5 == 0 else None))
        lines.append(json.dumps(obs))
    path.write_text("\n".join(lines) + "\n")
    return path


def truth(name, segments):
    return [GroundTruth(scenario_id=name, seed=0, segment_id=f"s{i}", t_start=a, t_end=b, truth_label=label,
                        expects_alert=label == "fall") for i, (a, b, label) in enumerate(segments)]


@pytest.fixture
def clips(tmp_path):
    fall = write_edge_clip(tmp_path / "lie_down-20261010-063944.jsonl", down_at=4.0)
    write_truth(truth_path(fall), truth(fall.stem, [(0, 4, "standing"), (4, 12, "fall")]))
    stand = write_edge_clip(tmp_path / "stand-20261010-064500.jsonl", down_at=None)
    write_truth(truth_path(stand), truth(stand.stem, [(0, 12, "standing")]))
    unlabelled = write_edge_clip(tmp_path / "lie_down-20261010-070000.jsonl", down_at=2.0, seconds=8)
    return {"fall": fall, "stand": stand, "unlabelled": unlabelled, "dir": tmp_path}


def test_loads_a_bare_edge_recording(clips):
    rec = load_recording(clips["fall"])
    assert rec.source == "jetson-01" and rec.name == "lie_down-20261010-063944"
    assert len(rec.observations) == 120 and rec.duration == pytest.approx(11.9)
    assert all(k.confidence is None for o in rec.observations for k in o.keypoints)


def test_loads_a_simulator_stream_with_its_truth(tmp_path):
    paths = write_run(run_headless(load_all()["possible-fall-01"]), tmp_path)
    rec = load_recording(paths["stream"])
    assert rec.truth() is not None and rec.source == "sim"
    assert any(type(i).__name__ == "Heartbeat" for i in rec.items)


def test_staged_fall_is_detected_and_standing_is_quiet(clips):
    fall = evaluate_recording(load_recording(clips["fall"]))
    assert fall.report.counts.true_positives == 1 and fall.report.counts.false_positives == 0
    assert 3.0 <= fall.report.detection_delays[0] <= 6.0  # 1 s going down + 3 s floor hold
    stand = evaluate_recording(load_recording(clips["stand"]))
    assert stand.report.counts.false_positives == 0 and not stand.alerts


def test_unlabelled_clip_is_replayed_but_not_scored(clips):
    result = evaluate_recording(load_recording(clips["unlabelled"]))
    assert result.report is None and len(result.alerts) == 1


def test_label_template_guesses_from_the_clip_name(clips):
    gt = label_template(load_recording(clips["unlabelled"]))
    assert [(g.truth_label, g.expects_alert, g.t_start) for g in gt] == [("fall", True, 0.0)]


def test_timeline_shows_the_detector_reaching_the_floor(clips):
    rows = timeline(load_recording(clips["fall"]))
    states = [s for b in rows for s in b.states]
    assert "upright" in states and "possible_fall" in states
    text = timeline_text(load_recording(clips["fall"]), rows)
    assert "possible_fall" in text and "FLOOR" in text


def test_sweep_shows_a_threshold_trading_detection(clips):
    recs = [load_recording(clips[k]) for k in ("fall", "stand")]
    rows = {r.value: r.summary.counts for r in sweep(recs, "floor_hold_seconds", [3.0, 30.0])}
    assert rows[3.0].true_positives == 1 and rows[30.0].missed == 1


def test_cli_real_writes_a_summary(clips, tmp_path, monkeypatch):
    from hearth.eval.real_monitor import CONFIG_ENV, OVERRIDES_ENV

    monkeypatch.setenv(OVERRIDES_ENV, "{}")
    monkeypatch.setenv(CONFIG_ENV, "")
    out = tmp_path / "out"
    assert main(["real", str(clips["dir"]), "--out", str(out), "--sweep", "floor_hold_seconds=2,3"]) == 0
    md = (out / "summary.md").read_text()
    assert "lie_down-20261010-063944" in md and "Not labelled yet" in md and "Sweep: floor_hold_seconds" in md
    assert "Features by true activity" in md
    assert (out / "clips" / "lie_down-20261010-063944.report.json").exists()


def test_cli_label_template_does_not_overwrite(clips, capsys):
    assert main(["label-template", str(clips["unlabelled"])]) == 0
    assert truth_path(clips["unlabelled"]).exists()
    assert main(["label-template", str(clips["fall"])]) == 0
    assert "skip" in capsys.readouterr().out


def test_cli_scores_a_live_session_export_by_start_time(clips, tmp_path, capsys):
    rec = load_recording(clips["fall"])
    result = evaluate_recording(rec)
    export = tmp_path / "export.jsonl"
    export.write_text("".join(e.model_dump_json() + "\n" for e in result.events))
    assert main(["score", "--start", rec.start.isoformat(), "--truth", str(truth_path(clips["fall"])),
                 "--events", str(export), "--source", "jetson-01"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["true_positives"] == 1


def test_fall_model_recording_scores_like_any_clip(tmp_path):
    """hearth.perception.cli --record on a video: model posture predictions, no keypoints."""
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from hearth.perception.cli import Recorder
    from hearth.perception.fall_model import ModelSettings, prediction_from_detections
    from hearth_schema import Observation

    settings = ModelSettings.load()
    start = datetime(2026, 10, 10, 15, 0, tzinfo=timezone.utc)
    context = dict(source="fall-model-01", source_type="replay", room_id="room-01", run_id="video-1")
    path = tmp_path / "video-fall.jsonl"
    recorder = Recorder(path)
    for i in range(60):  # 12 s at 5 fps: standing 4 s, then fallen
        label = "Standing" if i < 20 else "Fallen"
        box = [200, 80, 300, 460] if i < 20 else [100, 330, 500, 460]
        prediction = prediction_from_detections([dict(label=label, confidence=0.9, xyxy=box)], 640, 480, settings)
        recorder.write(Observation(obs_id=str(uuid4()), timestamp=start + timedelta(seconds=i / 5),
                                   person_detected=True, posture_prediction=prediction, face_present=None,
                                   **context), context, {"fps": 5.0, "inference_ms": 80.0})
    recorder.close()
    write_truth(truth_path(path), truth("video-fall", [(0, 4, "standing"), (4, 12, "fall")]))

    rec = load_recording(path)
    assert rec.source == "fall-model-01" and any(type(i).__name__ == "Heartbeat" for i in rec.items)
    result = evaluate_recording(rec)
    assert result.report.counts.true_positives == 1 and result.report.counts.false_positives == 0

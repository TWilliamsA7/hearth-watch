"""Face detection tests (logic only; the facenet model itself runs on the Jetson).

Run on the laptop: .venv/bin/python -m pytest edge/tests
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import run_edge  # noqa: E402
from hearth_edge import contract as c  # noqa: E402
from hearth_edge import face as face_mod  # noqa: E402

hs = pytest.importorskip("hearth_schema")

FRAME = {"t": 1791700000.5, "width": 640, "height": 480, "img": object()}


class FakeDetection(object):
    def __init__(self, confidence):
        self.Confidence = confidence


class FakeNet(object):
    def __init__(self, detections, ms=48.6):
        self.detections = detections
        self.ms = ms
        self.overlays = []

    def Detect(self, img, overlay="none"):
        self.overlays.append(overlay)
        return self.detections

    def GetNetworkTime(self):
        return self.ms


def detector(detections, draw=False):
    d = face_mod.FaceDetector.__new__(face_mod.FaceDetector)  # skip loading facenet
    d.net = FakeNet(detections)
    d.overlay = "box,conf" if draw else "none"
    return d


def test_check_reports_highest_confidence_face():
    assert detector([FakeDetection(0.62), FakeDetection(0.91)]).check(FRAME) == (True, 0.91, 48.6)


def test_check_with_no_face_is_false_not_null():
    present, conf, _ = detector([]).check(FRAME)
    assert present is False and conf is None


def test_boxes_drawn_only_when_streaming():
    quiet, shown = detector([]), detector([], draw=True)
    quiet.check(FRAME), shown.check(FRAME)
    assert quiet.net.overlays == ["none"] and shown.net.overlays == ["box,conf"]


def test_schedule_runs_at_most_once_per_interval():
    s = face_mod.FaceSchedule(0.5)
    times = [i * 0.1 for i in range(21)]  # 2 s of 10/s observations
    due = [t for t in times if s.due(t)]
    assert due == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])


@pytest.mark.parametrize("present,conf", [(True, 0.91), (False, None), (None, None)])
def test_face_fields_in_observations_pass_contract(present, conf):
    ids = c.ObservationIds("jetson-01", start=1791700000)
    for pose in (None, {"keypoints": {"neck": (320, 110)}, "bbox": (300, 100, 340, 120)}):
        obs = run_edge.observation_for_frame(ids, FRAME, pose, {"head_y": 0.2} if pose else None,
                                             "jetson-01", "room-01", present, conf)
        model = hs.Observation.model_validate(obs)
        assert model.face_present is present
        assert model.face_confidence == (pytest.approx(conf) if conf is not None else None)


def test_face_flags_parse():
    args = run_edge.parse_args(["--no-face"])
    assert args.no_face
    args = run_edge.parse_args(["--face-interval", "1", "--face-threshold", "0.6"])
    assert args.face_interval == 1.0 and args.face_threshold == 0.6 and not args.no_face

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
    def __init__(self, confidence, box=(100, 80, 180, 170)):
        self.Confidence = confidence
        self.Left, self.Top, self.Right, self.Bottom = box


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


def detector(detections):
    d = face_mod.FaceDetector.__new__(face_mod.FaceDetector)  # skip loading facenet
    d.net = FakeNet(detections)
    d.model = "facenet"
    d.hold_s = 0.7
    d.last = None
    d._drawer = None
    return d


def test_check_reports_highest_confidence_face():
    assert detector([FakeDetection(0.62), FakeDetection(0.91)]).check(FRAME) == (True, 0.91, 48.6)


def test_check_with_no_face_is_false_not_null():
    present, conf, _ = detector([]).check(FRAME)
    assert present is False and conf is None


def test_model_never_draws_itself():
    d = detector([FakeDetection(0.9)])
    d.check(FRAME)
    assert d.net.overlays == ["none"]  # boxes are drawn by draw(), held between checks


def test_boxes_are_held_until_the_next_check():
    d = detector([FakeDetection(0.9, (100, 80, 180, 170))])
    d.check(FRAME)
    t = FRAME["t"]
    assert d.boxes_to_draw(t + 0.3) == [(100, 80, 180, 170, 0.9)]  # between checks: still shown
    assert d.boxes_to_draw(t + 0.8) == []                            # stale: hidden
    d.net.detections = []
    d.check(dict(FRAME, t=t + 0.5))
    assert d.boxes_to_draw(t + 0.6) == []                            # next check found none


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


def test_parse_res10_keeps_faces_above_threshold_in_pixels():
    det = [[[
        [0, 1, 0.98, 0.25, 0.10, 0.50, 0.60],   # strong face
        [0, 1, 0.40, 0.70, 0.70, 0.80, 0.90],   # below threshold
        [0, 1, 0.90, -0.10, 0.90, 0.20, 1.20],  # partly outside the frame: clamped
        [0, 1, 0.95, 0.50, 0.50, 0.50, 0.60],   # zero width: dropped
    ]]]
    boxes = face_mod.parse_res10(det, 640, 480, 0.5)
    assert boxes[0] == pytest.approx((160.0, 48.0, 320.0, 288.0, 0.98))
    assert boxes[1] == pytest.approx((0.0, 432.0, 128.0, 480.0, 0.90))
    assert len(boxes) == 2


def test_default_face_model_is_res10_with_files_outside_edge():
    from hearth_edge import config
    assert config.FACE_MODEL == "res10"
    d = face_mod.default_res10_dir()
    assert d.endswith(os.path.join("models", "res10"))
    assert os.sep + "edge" + os.sep not in d  # syncing edge/ must not delete the model files


# --- head close-up ----------------------------------------------------------------------

def test_head_box_is_a_square_around_the_head_a_bit_below_the_eyes():
    kps = {"left_eye": (300, 200), "right_eye": (340, 200), "nose": (320, 220),
           "left_hip": (310, 400)}  # non-head points are ignored
    left, top, right, bottom = face_mod.head_box(kps, 640, 480)
    assert right - left == bottom - top == 120          # 3 x the 40 px spread
    assert (left + right) / 2 == pytest.approx(320, abs=1)
    assert (top + bottom) / 2 > 206                      # centre below the eye/nose average


def test_head_box_distant_face_gets_minimum_size_and_needs_two_points():
    far = face_mod.head_box({"left_eye": (100, 100), "right_eye": (106, 100)}, 640, 480)
    assert far[2] - far[0] == 64                         # min size, so res10 sees some context
    assert face_mod.head_box({"nose": (100, 100)}, 640, 480) is None
    assert face_mod.head_box({"left_hip": (1, 1), "right_hip": (2, 2)}, 640, 480) is None


def test_head_box_is_clamped_to_the_frame():
    box = face_mod.head_box({"left_eye": (2, 3), "right_eye": (40, 3)}, 640, 480)
    assert box[0] == 0 and box[1] == 0 and box[2] <= 640 and box[3] <= 480


def test_boxes_found_in_a_close_up_move_back_to_frame_coordinates():
    assert face_mod.offset_boxes([(10, 20, 30, 40, 0.9)], 100, 50) == [(110, 70, 130, 90, 0.9)]


def test_summarize():
    assert face_mod.summarize([]) == (False, None)
    assert face_mod.summarize([(0, 0, 1, 1, 0.6), (0, 0, 1, 1, 0.8)]) == (True, 0.8)


# --- background checker -----------------------------------------------------------------

def wait_for(cond, timeout=2.0):
    import time as _t
    end = _t.time() + timeout
    while _t.time() < end:
        if cond():
            return True
        _t.sleep(0.005)
    return cond()


def test_background_checker_runs_one_check_at_a_time_and_returns_each_result_once():
    import time as _t
    seen = []

    def slow_detect(bgr, roi):
        seen.append((bgr, roi))
        _t.sleep(0.05)
        return [(1, 2, 3, 4, 0.95)], 50.0

    checker = face_mod.AsyncFaceChecker(slow_detect, to_bgr=lambda img: "copy-of-%s" % img)
    try:
        assert checker.submit({"t": 1.0, "img": "frame1"}, roi=(0, 0, 10, 10))
        assert not checker.submit({"t": 1.1, "img": "frame2"})   # busy: refused, not queued
        assert checker.poll() is None                             # not finished yet
        assert wait_for(checker.idle)
        assert checker.poll() == (1.0, [(1, 2, 3, 4, 0.95)], 50.0)
        assert checker.poll() is None                             # handed out only once
        assert seen == [("copy-of-frame1", (0, 0, 10, 10))]      # the frame was copied
    finally:
        checker.stop()


def test_background_checker_reports_a_failed_check_and_keeps_working():
    calls = []

    def flaky(bgr, roi):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("model error")
        return [], 10.0

    checker = face_mod.AsyncFaceChecker(flaky, to_bgr=lambda img: img)
    try:
        checker.submit({"t": 1.0, "img": "a"})
        assert wait_for(checker.idle)
        assert checker.poll() == (1.0, None, 0.0)                 # None = check failed
        checker.submit({"t": 2.0, "img": "b"})
        assert wait_for(checker.idle)
        assert checker.poll() == (2.0, [], 10.0)
    finally:
        checker.stop()

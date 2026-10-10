"""Decoding the fall model's raw output (edge/hearth_edge/fall_decode.py).

Run on the laptop: .venv/bin/python -m pytest edge/tests
The comparison with Ultralytics runs only where ultralytics + onnxruntime + the exported model
are installed (skipped otherwise).
"""

import os
import sys

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hearth_edge import fall_decode as fd  # noqa: E402

NAMES = {0: "fallen", 1: "sitting", 2: "standing"}


def raw(rows, n=50):
    """Build a [1, 7, n] output; rows: [(cx, cy, w, h, fallen, sitting, standing)]."""
    out = np.zeros((1, 7, n), dtype=np.float32)
    for i, r in enumerate(rows):
        out[0, :, i] = r
    return out


def test_decode_picks_best_class_and_undoes_letterbox():
    # 640x480 frame letterboxed into 416: scale 0.65, pad top (416-312)/2 = 52, pad left 0
    out = raw([(208, 208, 104, 52, 0.9, 0.05, 0.02)])
    dets = fd.decode(out, NAMES, 0.65, 0, 52, 640, 480)
    assert len(dets) == 1 and dets[0]["label"] == "fallen"
    assert dets[0]["confidence"] == pytest.approx(0.9)
    assert dets[0]["xyxy"] == pytest.approx([240.0, 200.0, 400.0, 280.0])


def test_decode_drops_low_scores_and_duplicates_but_keeps_other_classes():
    out = raw([(200, 200, 100, 100, 0.8, 0, 0),      # fallen
               (202, 201, 100, 100, 0.7, 0, 0),      # same box, same class: duplicate
               (201, 200, 100, 100, 0, 0, 0.6),      # same box, different class: kept
               (50, 50, 20, 20, 0.1, 0.1, 0.1)])     # below 0.25
    dets = fd.decode(out, NAMES, 1.0, 0, 0, 416, 416)
    assert [(d["label"], round(d["confidence"], 2)) for d in dets] == [("fallen", 0.8), ("standing", 0.6)]


def test_decode_nothing_and_clipping():
    assert fd.decode(raw([]), NAMES, 1.0, 0, 0, 416, 416) == []
    dets = fd.decode(raw([(5, 5, 40, 40, 0.9, 0, 0)]), NAMES, 1.0, 0, 0, 416, 416)
    assert dets[0]["xyxy"][:2] == [0.0, 0.0]


def test_letterbox_shape_padding_and_scale():
    pytest.importorskip("cv2")
    rgb = np.full((480, 640, 3), 200, dtype=np.uint8)
    x, scale, left, top = fd.letterbox(rgb, 416)
    assert x.shape == (1, 3, 416, 416) and x.dtype == np.float32
    assert scale == pytest.approx(0.65) and left == 0 and top == 52
    assert x[0, 0, 0, 0] == pytest.approx(114 / 255.0)       # grey padding
    assert x[0, 0, 208, 208] == pytest.approx(200 / 255.0)   # image


def test_matches_ultralytics_predict_on_the_real_model():
    ort = pytest.importorskip("onnxruntime")
    ultra = pytest.importorskip("ultralytics")
    cv2 = pytest.importorskip("cv2")
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "..", "models",
                             "human-fall-detection", "best_416.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip("export the model first (edge/pc/03_export_onnx.py)")
    from ultralytics.utils import ASSETS
    bgr = cv2.imread(str(ASSETS / "bus.jpg"))
    x, scale, left, top = fd.letterbox(bgr[:, :, ::-1].copy(), 416)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ours = fd.decode(sess.run(None, {"images": x})[0], NAMES, scale, left, top,
                     bgr.shape[1], bgr.shape[0])
    theirs = ultra.YOLO(onnx_path, task="detect").predict(str(ASSETS / "bus.jpg"), imgsz=416,
                                                          conf=0.25, verbose=False)[0]
    their = sorted(zip(theirs.boxes.conf.tolist(), theirs.boxes.cls.tolist(), theirs.boxes.xyxy.tolist()),
                   reverse=True)
    assert len(ours) == len(their)
    for d, (conf, cls, box) in zip(ours, their):
        assert d["label"] == NAMES[int(cls)]
        assert d["confidence"] == pytest.approx(conf, abs=0.03)
        assert d["xyxy"] == pytest.approx(box, abs=6.0)   # pixels; letterbox rounding differs slightly

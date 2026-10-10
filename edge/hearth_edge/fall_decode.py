"""Pre- and post-processing for the YOLO11 fall model's raw output. numpy only (Python 3.6 OK).

The ONNX/TensorRT model takes a letterboxed square image and returns one tensor
[1, 4 + classes, N]: for each of N candidate positions, a box (centre x, centre y, width,
height, in pixels of the square input) and one score per class. This module:
  letterbox()  - camera frame (RGB uint8, H x W x 3) -> model input (1 x 3 x S x S float32, 0..1),
                 aspect kept, grey (114) padding; same as Ultralytics and edge/pc/03_export_onnx.py
  decode()     - raw output -> detections in camera-frame pixels: best class per position,
                 score >= min_score, boxes un-letterboxed and clipped, then class-wise
                 non-maximum suppression (drops overlapping duplicates), like Ultralytics' NMS.
edge/tests/test_fall_decode.py checks it against Ultralytics' own predict() on the real model.
"""

import numpy as np


def letterbox(rgb, size):
    """Returns (input tensor, scale, pad_left, pad_top)."""
    import cv2
    h, w = rgb.shape[:2]
    scale = min(size / float(h), size / float(w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), scale, left, top


def nms(boxes, scores, iou_threshold):
    """Indices to keep, highest score first. boxes: N x 4 (x1, y1, x2, y2)."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        ix1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode(output, names, scale, pad_left, pad_top, frame_w, frame_h, min_score=0.25,
           iou_threshold=0.7, max_det=300):
    """output: [1, 4 + C, N]. names: {class index: name}. Returns
    [{"label": name, "confidence": score, "xyxy": [x1, y1, x2, y2]}] in camera-frame pixels."""
    pred = output[0].T                                   # N x (4 + C)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(axis=1)
    conf = scores_all[np.arange(len(cls)), cls]
    mask = conf >= min_score
    if not mask.any():
        return []
    pred, cls, conf = pred[mask], cls[mask], conf[mask]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    # Class-wise NMS: shift each class's boxes far apart so different classes never suppress.
    offset = cls[:, None].astype(np.float32) * 7680.0
    keep = nms(boxes + offset, conf, iou_threshold)[:max_det]
    out = []
    for i in keep:
        x1 = (boxes[i, 0] - pad_left) / scale
        y1 = (boxes[i, 1] - pad_top) / scale
        x2 = (boxes[i, 2] - pad_left) / scale
        y2 = (boxes[i, 3] - pad_top) / scale
        out.append({"label": names[int(cls[i])], "confidence": float(conf[i]),
                    "xyxy": [float(min(max(x1, 0), frame_w)), float(min(max(y1, 0), frame_h)),
                             float(min(max(x2, 0), frame_w)), float(min(max(y2, 0), frame_h))]})
    return out

"""Our own posture model on the Jetson: poseNet keypoints -> upright / sitting / floor. numpy only.

Trained on the PC in PyTorch (edge/pc/06_train.py, results in edge/pc/RESULTS.md: on people it
never saw, right 97.7%, floor caught 96.1%) and exported by edge/pc/08_export_weights.py into
posture_mlp.npz next to this file: three small layers (7,683 numbers) plus the settings.

Inputs are built exactly as in training (06_train.py features()): each keypoint in 0-1 of the
frame, x stretched back to true proportions (times width/height), centred on the mean of the
found keypoints, divided by the larger side of their box; missing keypoints are 0 with a
"found" flag of 0. poseNet's extra "neck" isn't one of the 17 COCO points and is ignored.
08_export_weights.py runs our recordings through this module and checks every decision matches
the PyTorch pipeline before it writes the file.

Not sent to the backend: its posture_prediction only trusts the Hugging Face model's id. Shown
on the preview panel (run_edge.py --posture-mlp).
"""

import os

import numpy as np

KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def default_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "posture_mlp.npz")


class PostureMLP(object):
    def __init__(self, path=None):
        self.path = path or default_path()
        if not os.path.exists(self.path):
            raise IOError("posture model not found: %s (run edge/pc/08_export_weights.py on the "
                          "PC and commit the file)" % self.path)
        d = np.load(self.path)
        if [str(n) for n in d["keypoint_names"]] != KEYPOINT_NAMES:
            raise ValueError("posture model uses a different keypoint order")
        self.layers = [(d["W%d" % i].astype(np.float32), d["b%d" % i].astype(np.float32))
                       for i in (1, 2, 3)]
        self.classes = [str(c) for c in d["classes"]]
        self.min_conf = float(d["min_conf"])
        self.min_found = int(d["min_found"])
        self.source = str(d["source_model"])

    def features(self, keypoints, width, height):
        """{name: (x_px, y_px)} -> the 51 inputs (float32), or None if too few keypoints."""
        xy = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32)
        found = np.zeros(len(KEYPOINT_NAMES), dtype=bool)
        for i, name in enumerate(KEYPOINT_NAMES):
            if name in keypoints:
                x, y = keypoints[name]
                xy[i] = (min(max(x / float(width), 0.0), 1.0), min(max(y / float(height), 0.0), 1.0))
                found[i] = True
        if found.sum() < self.min_found:
            return None
        p = xy * np.array([width / float(height), 1.0], dtype=np.float32)
        pts = p[found]
        centre = pts.mean(axis=0)
        size = max(float((pts.max(axis=0) - pts.min(axis=0)).max()), 1e-3)
        rel = (p - centre) / size
        rel[~found] = 0.0
        return np.concatenate([rel, found[:, None].astype(np.float32)], axis=1).reshape(-1)

    def scores(self, x):
        """51 inputs -> raw scores per class (Linear, ReLU, Linear, ReLU, Linear)."""
        h = x
        for i, (w, b) in enumerate(self.layers):
            h = np.dot(h, w.T) + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h

    def predict(self, keypoints, width, height):
        """-> {"label", "confidence", "probabilities"}; label "unknown" under the cut-off;
        None when there are too few keypoints to judge."""
        x = self.features(keypoints, width, height)
        if x is None:
            return None
        s = self.scores(x)
        e = np.exp(s - s.max())
        p = e / e.sum()
        best = int(p.argmax())
        confidence = float(p[best])
        return {"label": self.classes[best] if confidence >= self.min_conf else "unknown",
                "confidence": confidence,
                "probabilities": dict((c, float(v)) for c, v in zip(self.classes, p))}

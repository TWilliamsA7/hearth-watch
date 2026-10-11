"""Face templates on the Jetson, for opted-in names on the laptop. Off by default (--face-id).

Compute only runs when it can count: the res10 face check (face.py) already looks inside one
person's head box; only when it finds a face there, and poseNet has that person's eyes and nose,
does SFace run on the face, aligned with those three keypoints (no landmark model). SFace is
opencv/face_recognition_sface @ 3d70824, the same model the laptop enrolls with, built into a
TensorRT FP16 engine by edge/tools/get_face_id_model.sh (files in <repo>/models/face-id/).

What leaves the Jetson: that person's anonymous person_id and the 128-value template, POSTed to
the laptop's /face-demo/edge-embeddings with a shared token. Never the frame or a face crop.
Matching, consent and the name stay on the laptop; the edge never learns who it saw.

Standard library plus numpy and cv2 (Python 3.6). TensorRT and CUDA as in fall_model.py.
"""

import ctypes
import json
import os
import sys
import threading
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from hearth_edge import contract

# SFace's aligned 112x112 face: where OpenCV's alignCrop puts the person's right eye (image
# left when they face the camera), left eye and nose tip. poseNet's eye names are the person's own.
REFERENCE = (("right_eye", (38.2946, 51.6963)), ("left_eye", (73.5318, 51.5014)),
             ("nose", (56.0252, 71.7366)))
SIZE = 112


def default_engine_path():
    """<repo>/models/face-id/sface_fp16.engine (this file is <repo>/edge/hearth_edge/...)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "models", "face-id", "sface_fp16.engine")


def similarity_transform(src, dst):
    """2x3 matrix (rotation, uniform scale, shift) taking points src onto dst, least squares
    (Umeyama). Mirroring is never allowed: a face can turn, not flip."""
    import numpy as np
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(d.T.dot(s) / len(src))
    flip = np.diag([1.0, 1.0 if np.linalg.det(u) * np.linalg.det(vt) > 0 else -1.0])
    rotation = u.dot(flip).dot(vt)
    scale = np.trace(np.diag(sv).dot(flip)) / ((s ** 2).sum() / len(src))
    return np.hstack([scale * rotation, (mu_d - scale * rotation.dot(mu_s)).reshape(2, 1)])


def eye_distance(kps):
    (x1, y1), (x2, y2) = kps["left_eye"][:2], kps["right_eye"][:2]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def face_in_head(boxes, head):
    """The most confident res10 box whose centre is inside this person's head box, or None.
    A face elsewhere (someone else's) never stands in for this person's."""
    left, top, right, bottom = head
    inside = [b for b in boxes or []
              if left <= (b[0] + b[2]) / 2.0 <= right and top <= (b[1] + b[3]) / 2.0 <= bottom]
    return max(inside, key=lambda b: b[4]) if inside else None


def alignable(kps, min_eye_px):
    """True when poseNet has this person's eyes and nose, with the eyes far enough apart to align."""
    return all(name in kps for name, _ in REFERENCE) and eye_distance(kps) >= min_eye_px


def align(bgr, kps):
    """The person's face, aligned to SFace's 112x112 input with their eyes and nose (BGR)."""
    import cv2
    m = similarity_transform([kps[name][:2] for name, _ in REFERENCE], [p for _, p in REFERENCE])
    return cv2.warpAffine(bgr, m, (SIZE, SIZE))


class SFaceEngine(object):
    """SFace on the GPU with TensorRT 8.2: aligned BGR face -> L2-normalized 128-value template.

    Plain cudaMalloc + cudaMemcpy through ctypes, not managed memory (see fall_model.py: the
    Nano's GPU kills the process if the CPU touches a managed buffer while poseNet runs).
    """

    def __init__(self, engine_path=None):
        import numpy as np
        import tensorrt as trt
        self._np = np
        self.engine_path = engine_path or default_engine_path()
        if not os.path.exists(self.engine_path):
            raise IOError("face template engine not found: %s (run edge/tools/get_face_id_model.sh "
                          "on the Jetson)" % self.engine_path)
        self._logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as fh, trt.Runtime(self._logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(fh.read())
        self._context = self._engine.create_execution_context()
        self._cudart = ctypes.CDLL("libcudart.so")
        self._ptrs, self._nbytes = [], []
        for i in range(self._engine.num_bindings):
            shape = tuple(self._engine.get_binding_shape(i))
            nbytes = int(np.prod(shape)) * 4                 # float32
            ptr = ctypes.c_void_p()
            self._check(self._cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
            self._ptrs.append(ptr.value)
            self._nbytes.append(nbytes)
            if self._engine.binding_is_input(i):
                self._in = i
            else:
                self._out, self._out_host = i, np.empty(shape, dtype=np.float32)

    @staticmethod
    def _check(code, what):
        if code != 0:
            raise RuntimeError("%s failed (CUDA error %d)" % (what, code))

    def _copy(self, dst, src, nbytes, kind):
        self._check(self._cudart.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                            ctypes.c_size_t(nbytes), kind), "cudaMemcpy")

    def embed(self, face_bgr):
        np = self._np
        # The model subtracts 127.5 and scales itself: it wants raw 0-255 RGB, NCHW.
        x = np.ascontiguousarray(face_bgr[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        self._copy(self._ptrs[self._in], x.ctypes.data, self._nbytes[self._in], 1)   # to GPU
        if not self._context.execute_v2(self._ptrs):
            raise RuntimeError("TensorRT execution failed")
        self._copy(self._out_host.ctypes.data, self._ptrs[self._out], self._nbytes[self._out], 2)
        return normalized(self._out_host.ravel())

    def close(self):
        for p in self._ptrs:
            self._cudart.cudaFree(ctypes.c_void_p(p))
        self._ptrs = []


def normalized(vector):
    """L2-normalized list of floats, or None if the vector is empty, zero or not finite."""
    values = [float(v) for v in vector]
    norm = sum(v * v for v in values) ** 0.5
    if not values or not norm > 0 or norm != norm or norm == float("inf"):
        return None
    return [v / norm for v in values]


class FaceTurns(object):
    """Which person the next face check looks at: whoever was checked longest ago (never
    checked first), so everyone in view gets their turn."""

    def __init__(self):
        self.last = {}   # person_id -> time of their last check

    def pick(self, candidates, t):
        """candidates: [(person_id, head_box)] -> one of them, or None."""
        if not candidates:
            return None
        live = set(pid for pid, _ in candidates)
        self.last = dict((pid, at) for pid, at in self.last.items() if pid in live)
        choice = min(candidates, key=lambda c: (self.last.get(c[0], float("-inf")), c[0]))
        self.last[choice[0]] = t
        return choice


class FaceIdCheck(object):
    """The background face check (face.AsyncFaceChecker's detect function) with templates.

    check(bgr, job) runs res10 inside job["head"] exactly as the presence check does and returns
    (boxes, ms) the same way. Only when a face sits in that head box and the person's eyes and
    nose are known does it also align, run SFace and hand the template to `send`.
    """

    def __init__(self, detect_bgr, engine, send, source, room_id, min_eye_px, log=None):
        self.detect_bgr, self.engine, self.send = detect_bgr, engine, send
        self.source, self.room_id, self.min_eye_px = source, room_id, min_eye_px
        self.log = log or (lambda msg: None)
        self.templates = 0           # SFace runs since the last status line
        self.template_ms = 0.0

    def __call__(self, bgr, job):
        boxes, ms = self.detect_bgr(bgr, job["head"])
        face = face_in_head(boxes, job["head"]) if job["head"] is not None else None
        if face is not None and alignable(job["kps"], self.min_eye_px):
            start = time.time()
            embedding = self.engine.embed(align(bgr, job["kps"]))
            spent = (time.time() - start) * 1000.0
            ms += spent
            if embedding is not None:
                self.templates += 1
                self.template_ms += spent
                height, width = bgr.shape[:2]
                self.send(self.template(job, face, embedding, width, height))
        return boxes, ms

    def template(self, job, face, embedding, width, height):
        def unit(v, size):
            return round(min(1.0, max(0.0, v / float(size))), 4)
        return {
            "detector": "res10+sface",
            "source": self.source,
            "room_id": self.room_id,
            "person_id": job["person_id"],
            "obs_id": "%s-%s-face-%d" % (self.source, job["person_id"], int(job["t"] * 1000)),
            "timestamp": contract.utc_iso(job["t"]),
            "face_box": {"x1": unit(face[0], width), "y1": unit(face[1], height),
                         "x2": unit(face[2], width), "y2": unit(face[3], height)},
            "embedding": [round(v, 6) for v in embedding],
        }


def _default_log(msg):
    try:
        sys.stderr.write("[face-id %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        sys.stderr.flush()
    except (IOError, OSError, ValueError):
        pass


class TemplateSender(object):
    """Posts face templates on its own thread; the camera loop and face check never wait.

    Keeps only the newest template per person (an older one is worth nothing once a newer face
    is seen) and at most `max_people` of them. A failed post is dropped, not retried: the next
    check sends a fresh one. A 403/422 is logged once, since it means a setup problem.
    """

    def __init__(self, backend_url, token, path="/face-demo/edge-embeddings", timeout_s=2.0,
                 max_people=8, log=_default_log, opener=None):
        self.url = backend_url.rstrip("/") + path
        self.token, self.timeout_s, self.max_people, self.log = token, timeout_s, max_people, log
        self._open = opener or urlrequest.urlopen
        self._pending = {}         # person_id -> newest template, in arrival order
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._complained = set()
        self.counts = {"sent": 0, "dropped": 0, "failed": 0}
        self.statuses = {}         # person_id -> the laptop's last answer ("confirming", ...)

    def offer(self, template):
        with self._lock:
            pid = template["person_id"]
            if pid in self._pending:
                self.counts["dropped"] += 1
                del self._pending[pid]
            elif len(self._pending) >= self.max_people:
                self._pending.pop(next(iter(self._pending)))
                self.counts["dropped"] += 1
            self._pending[pid] = template
        self._wake.set()

    def start(self):
        self._thread = threading.Thread(target=self._run, name="face-id-sender")
        self._thread.daemon = True
        self._thread.start()

    def stop(self, timeout_s=2.0):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def _take(self):
        with self._lock:
            if not self._pending:
                return None
            pid = next(iter(self._pending))
            return self._pending.pop(pid)

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait(1.0)
            self._wake.clear()
            while not self._stop.is_set():
                template = self._take()
                if template is None:
                    break
                self.post(template)

    def post(self, template):
        body = json.dumps(template).encode("utf-8")
        req = urlrequest.Request(self.url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer %s" % self.token})
        try:
            with self._open(req, timeout=self.timeout_s) as resp:
                answer = json.loads(resp.read().decode("utf-8") or "{}")
            self.counts["sent"] += 1
            self.statuses[template["person_id"]] = answer.get("status")
        except urlerror.HTTPError as e:
            self.counts["failed"] += 1
            if e.code in (403, 422) and e.code not in self._complained:
                self._complained.add(e.code)
                self.log("laptop refused a face template (HTTP %d%s); check HEARTH_FACE_DEMO, "
                         "HEARTH_EDGE_FACE_TOKEN and the clock" % (e.code, _detail(e)))
        except (urlerror.URLError, OSError, ValueError):
            self.counts["failed"] += 1


def _detail(e):
    try:
        return ": " + json.loads(e.read().decode("utf-8")).get("detail", "")
    except Exception:
        return ""


def read_token(path=None, environ=os.environ):
    """The shared token from a file (first line) or HEARTH_EDGE_FACE_TOKEN; None if neither."""
    if path:
        with open(path) as fh:
            token = fh.readline().strip()
    else:
        token = environ.get("HEARTH_EDGE_FACE_TOKEN", "").strip()
    return token or None

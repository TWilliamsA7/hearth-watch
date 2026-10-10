"""Face DETECTION on the Jetson. Never identification.

Answers one question per check: is any face visible, and how confident is the model?
No face crops, embeddings or identities are stored or sent. The result fills the contract's
face_present / face_confidence fields.

Two backends (config.FACE_MODEL):
  * "res10"   (default) OpenCV DNN SSD face model on the CPU. Files from
               edge/tools/get_face_model.sh in <repo>/models/res10/.
  * "facenet" jetson-inference detectNet on the GPU.
Measured with a face in view (edge/tools/face_probe.py): res10 found it in 100% of frames,
facenet in 0%. So res10 is the default. See edge/BENCHMARKS.md.

Two things make res10 work well here:
  * Head close-up: when poseNet found the head (nose/eyes/ears), only a square around it is
    given to the model, so a distant face fills the model's 300x300 input instead of shrinking
    to a few pixels. Without head keypoints the whole frame is checked.
  * Background thread (AsyncFaceChecker): res10 takes ~180 ms on the CPU. Running it beside the
    camera loop, instead of inside it, avoids a stall every check. A result is attached to the
    next observation sent after it finishes, i.e. up to ~0.2 s after the frame it came from.

For the optional video preview, boxes from the last result stay on screen until the next one
(draw()), so they don't flicker between checks.
"""

import os
import threading
import time

RED = (255, 40, 40, 255)
YELLOW = (255, 210, 0, 255)
RES10_PROTO = "deploy.prototxt"
RES10_WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"
HEAD_POINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def default_res10_dir():
    """<repo>/models/res10: this file is <repo>/edge/hearth_edge/face.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "models", "res10")


def head_box(kps, width, height, scale=3.0, min_size=64.0):
    """Square (left, top, right, bottom) around poseNet's head keypoints, or None.

    Needs at least 2 of nose/eyes/ears. Sized to 3x the keypoint spread (at least 64 px) and
    centred a little below eye level, where the middle of the face is.
    """
    pts = [kps[n] for n in HEAD_POINTS if n in kps]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    size = max(scale * max(max(xs) - min(xs), max(ys) - min(ys)), min_size)
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys) + 0.15 * size
    left, top = int(max(0, cx - size / 2)), int(max(0, cy - size / 2))
    right, bottom = int(min(width, cx + size / 2)), int(min(height, cy + size / 2))
    if right - left < 24 or bottom - top < 24:
        return None
    return (left, top, right, bottom)


def parse_res10(det, width, height, threshold):
    """res10 output (shape 1x1xNx7) -> [(left, top, right, bottom, confidence)] above threshold.

    Each row is [image_id, label, confidence, x1, y1, x2, y2] with coordinates in 0..1 of the
    image that was given to the model (width x height).
    """
    boxes = []
    for row in det[0][0]:
        conf = float(row[2])
        if conf < threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in row[3:7]]
        left, top = max(0.0, x1 * width), max(0.0, y1 * height)
        right, bottom = min(float(width), x2 * width), min(float(height), y2 * height)
        if right > left and bottom > top:
            boxes.append((left, top, right, bottom, conf))
    return boxes


def offset_boxes(boxes, dx, dy):
    """Move boxes found in a close-up back into whole-frame coordinates."""
    return [(l + dx, t + dy, r + dx, b + dy, c) for l, t, r, b, c in boxes]


class FaceDetector(object):
    def __init__(self, model="res10", threshold=0.5, draw=False, hold_s=0.7, res10_dir=None):
        self.model = model
        self.threshold = threshold
        self.hold_s = hold_s
        self.last = None  # (t, [(left, top, right, bottom, confidence), ...]) from the last result
        self._drawer = None
        # Jetson-only imports live here so the pure helpers can be tested off the Jetson.
        try:
            import jetson_utils as ju
        except ImportError:  # pragma: no cover - older installs
            import jetson.utils as ju
        self._ju = ju
        if model == "res10":
            import cv2
            d = res10_dir or default_res10_dir()
            proto, weights = os.path.join(d, RES10_PROTO), os.path.join(d, RES10_WEIGHTS)
            if not (os.path.exists(proto) and os.path.exists(weights)):
                raise IOError("res10 face model not found in %s (run edge/tools/get_face_model.sh "
                              "on the Jetson)" % d)
            self._cv2 = cv2
            self.net = cv2.dnn.readNetFromCaffe(proto, weights)
        elif model == "facenet":
            try:
                from jetson_inference import detectNet
            except ImportError:  # pragma: no cover
                from jetson.inference import detectNet
            self.net = detectNet(model, threshold=threshold)
        else:
            raise ValueError("unknown face model %r (use res10 or facenet)" % model)
        if draw:
            self._drawer = (ju.cudaDrawRect, ju.cudaFont(size=20))

    # --- res10 building blocks (also used by the background thread and face_probe.py) -----

    def frame_to_bgr(self, img):
        """Copy a camera frame (CUDA image) into a BGR numpy array OpenCV can use."""
        self._ju.cudaDeviceSynchronize()  # the GPU must finish writing the frame first
        rgb = self._ju.cudaToNumpy(img)[..., :3].astype("uint8")  # astype = a real copy
        return self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)

    def detect_bgr(self, bgr, roi=None, threshold=None):
        """res10 on a BGR image, optionally only inside roi. Returns (boxes, ms), boxes in
        whole-frame coordinates."""
        start = time.time()
        cv2 = self._cv2
        left = top = 0
        image = bgr
        if roi is not None:
            left, top, right, bottom = roi
            image = bgr[top:bottom, left:right]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        boxes = parse_res10(self.net.forward(), image.shape[1], image.shape[0],
                            self.threshold if threshold is None else threshold)
        return offset_boxes(boxes, left, top), (time.time() - start) * 1000.0

    # --- one synchronous check (facenet, bench.py) -----------------------------------------

    def check(self, frame, roi=None):
        """Check one frame now. Returns (face_present, confidence, ms).

        roi (a head_box) is used by res10 only. confidence is the highest face score, or None
        when no face was found.
        """
        if self.model == "res10":
            start = time.time()
            boxes, _ = self.detect_bgr(self.frame_to_bgr(frame["img"]), roi)
            ms = (time.time() - start) * 1000.0
        else:
            detections = self.net.Detect(frame["img"], overlay="none")
            ms = self.net.GetNetworkTime()
            boxes = [(d.Left, d.Top, d.Right, d.Bottom, d.Confidence) for d in detections]
        self.last = (frame["t"], boxes)
        return summarize(boxes) + (ms,)

    def start_background(self):
        """A background checker for res10 (CPU), so checks run beside the camera loop."""
        if self.model != "res10":
            raise ValueError("background checks are for res10 only")
        return AsyncFaceChecker(self.detect_bgr, self.frame_to_bgr)

    # --- preview drawing --------------------------------------------------------------------

    def boxes_to_draw(self, t):
        """Boxes from the last result if it is recent enough to still show, else []."""
        if self.last is None or t - self.last[0] > self.hold_s:
            return []
        return self.last[1]

    def draw(self, frame):
        """Draw the held face boxes on the preview frame (only when streaming)."""
        for left, top, right, bottom, conf in self.boxes_to_draw(frame["t"]):
            self.draw_box(frame, (left, top, right, bottom), RED, "face %.2f" % conf)

    def draw_box(self, frame, box, color, label=None):
        """Outline one box (with an optional label) on the preview frame, if drawing is on."""
        if self._drawer is None:
            return
        draw_rect, font = self._drawer
        left, top, right, bottom = box[:4]
        try:
            draw_rect(frame["img"], (left, top, right, bottom), (0, 0, 0, 0),
                      line_color=color, line_width=3)
            if label:
                font.OverlayText(frame["img"], text=label, x=int(left), y=max(0, int(top) - 22),
                                 color=color, background=(0, 0, 0, 160))
        except Exception:
            pass  # drawing must never stop capture


def summarize(boxes):
    """boxes -> (face_present, highest confidence or None)."""
    if not boxes:
        return False, None
    return True, max(b[4] for b in boxes)


class AsyncFaceChecker(object):
    """Runs one face check at a time on a background thread.

    submit() copies the frame immediately (the camera reuses its buffers) and returns at once;
    poll() hands back each finished result exactly once as (frame_t, boxes, ms). boxes is None
    if the check itself failed.
    """

    def __init__(self, detect_fn, to_bgr):
        self._detect = detect_fn
        self._to_bgr = to_bgr
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job = None
        self._result = None
        self._busy = False
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="face-check")
        self._thread.daemon = True
        self._thread.start()

    def idle(self):
        with self._lock:
            return not self._busy

    def submit(self, frame, roi=None):
        """Start a check of this frame unless one is already running. Returns True if started."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        try:
            bgr = self._to_bgr(frame["img"])
        except Exception:
            with self._lock:
                self._busy = False
            raise
        with self._lock:
            self._job = (frame["t"], bgr, roi)
        self._wake.set()
        return True

    def poll(self):
        with self._lock:
            result, self._result = self._result, None
        return result

    def _run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stopping:
                return
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            t, bgr, roi = job
            try:
                boxes, ms = self._detect(bgr, roi)
            except Exception:
                boxes, ms = None, 0.0
            with self._lock:
                self._result = (t, boxes, ms)
                self._busy = False

    def stop(self, timeout_s=2.0):
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout_s)


class FaceSchedule(object):
    """Decides when to start a face check: at most one per `interval_s` seconds."""

    def __init__(self, interval_s):
        self.interval_s = interval_s
        self.next_t = None

    def due(self, t):
        if self.next_t is None or t >= self.next_t:
            self.next_t = t + self.interval_s
            return True
        return False

"""Face DETECTION on the Jetson (jetson-inference detectNet 'facenet'). Never identification.

Answers one question per checked frame: is any face visible, and how confident is the model?
No face crops, embeddings or identities are produced, stored or sent. The result fills the
contract's face_present / face_confidence fields.

Measured on the Jetson Nano 2GB alongside poseNet: +88 MB memory, ~49 ms per check. Checking
every frame drops 12.7 -> 8.4 FPS, so run_edge.py checks only every FACE_INTERVAL_S seconds and
sends face_present = null ("did not run") on the other frames.
"""

class FaceDetector(object):
    def __init__(self, model="facenet", threshold=0.5, draw=False):
        # Imported here (not at module top) so FaceSchedule can be tested off the Jetson.
        try:
            from jetson_inference import detectNet
        except ImportError:  # pragma: no cover - older installs
            from jetson.inference import detectNet
        self.net = detectNet(model, threshold=threshold)
        # Draw face boxes only when the video preview is streaming.
        self.overlay = "box,conf" if draw else "none"

    def check(self, frame):
        """Run on one frame from PoseCamera.read(). Returns (face_present, confidence, ms).

        confidence is the highest face score, or None when no face was found.
        """
        detections = self.net.Detect(frame["img"], overlay=self.overlay)
        ms = self.net.GetNetworkTime()
        if not detections:
            return False, None, ms
        return True, max(d.Confidence for d in detections), ms


class FaceSchedule(object):
    """Decides which frames get a face check: at most one per `interval_s` seconds."""

    def __init__(self, interval_s):
        self.interval_s = interval_s
        self.next_t = None

    def due(self, t):
        if self.next_t is None or t >= self.next_t:
            self.next_t = t + self.interval_s
            return True
        return False

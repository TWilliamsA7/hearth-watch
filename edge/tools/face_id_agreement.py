"""Does aligning a face with three keypoints (as --face-id does) still match a laptop enrollment?

The laptop enrolls with YuNet's five landmarks (OpenCV alignCrop); the Jetson aligns with
poseNet's eyes and nose. This runs SFace both ways on the same photos and prints the cosine
similarity between the two templates, with the eye/nose points jittered by a few pixels the way
poseNet's are. Compare with the laptop's match threshold (FaceDemo.THRESHOLD) and margin.
With two or more photos of different people it also prints their cross-person similarity,
which must stay well below the threshold.

Laptop only (Python 3.11+, the backend's face-demo extra and downloaded models):
    .venv/Scripts/python.exe edge/tools/face_id_agreement.py photo1.jpg [photo2.jpg ...]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hearth_edge import face_id  # noqa: E402

JITTER_PX = (0, 2, 4)


def templates(path, detector, recognizer, rng, trials):
    import cv2
    image = cv2.imread(path)
    if image is None:
        raise SystemExit("cannot read %s" % path)
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise SystemExit("no face found in %s" % path)
    face = max(faces, key=lambda f: f[2] * f[3])
    laptop = face_id.normalized(recognizer.feature(recognizer.alignCrop(image, face)).ravel())
    # YuNet's landmarks: person's right eye, left eye, nose tip, mouth corners.
    points = {"right_eye": face[4:6], "left_eye": face[6:8], "nose": face[8:10]}
    eye_px = float(np.hypot(*(points["left_eye"] - points["right_eye"])))
    edge = {}
    for jitter in JITTER_PX:
        runs = []
        for _ in range(trials if jitter else 1):
            kps = dict((k, tuple(v + rng.uniform(-jitter, jitter, 2))) for k, v in points.items())
            runs.append(face_id.normalized(recognizer.feature(face_id.align(image, kps)).ravel()))
        edge[jitter] = runs
    return laptop, edge, eye_px


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "..", "..", "models", "face-demo"))
    ap.add_argument("--trials", type=int, default=10, help="jittered alignments per level")
    args = ap.parse_args(argv)
    detector = cv2.FaceDetectorYN.create(os.path.join(args.models, "face_detection_yunet_2023mar.onnx"),
                                         "", (640, 480), .9, .3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(os.path.join(args.models, "face_recognition_sface_2021dec.onnx"), "")
    rng = np.random.RandomState(0)
    results = {}
    print("same photo, laptop alignment vs keypoint alignment (cosine similarity):")
    for path in args.photos:
        laptop, edge, eye_px = templates(path, detector, recognizer, rng, args.trials)
        results[path] = laptop
        cells = []
        for jitter in JITTER_PX:
            sims = [float(np.dot(laptop, e)) for e in edge[jitter]]
            cells.append("+-%dpx min %.3f mean %.3f" % (jitter, min(sims), sum(sims) / len(sims)))
        print("  %s (eyes %.0f px apart): %s" % (os.path.basename(path), eye_px, " | ".join(cells)))
    if len(results) > 1:
        print("different photos, laptop alignment (expect low if they are different people):")
        paths = list(results)
        for i, a in enumerate(paths):
            for b in paths[i + 1:]:
                print("  %s vs %s: %.3f" % (os.path.basename(a), os.path.basename(b),
                                            float(np.dot(results[a], results[b]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

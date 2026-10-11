"""Local YuNet/SFace extraction for opt-in fictional-resident demonstrations."""
import argparse
import base64
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

MODELS = {
    "face_detection_yunet/face_detection_yunet_2023mar.onnx":
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface/face_recognition_sface_2021dec.onnx":
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "models" / "face-demo"
MODEL_REVISIONS = {
    "face_detection_yunet": "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
    "face_recognition_sface": "3d7082438a6e4551e840c9b2bb60b71e8da4b524",
}


def verify(path, expected):
    if not path.is_file():
        raise ValueError("Face models are missing. Run python -m hearth.perception.face_demo --download")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("Face model checksum mismatch")


def download(directory=DEFAULT_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for relative, expected in MODELS.items():
        path = directory / Path(relative).name
        if not path.exists():
            temporary = path.with_suffix("." + uuid4().hex + ".part")
            try:
                repo, filename = relative.split("/")
                revision = MODEL_REVISIONS[repo]
                url = f"https://huggingface.co/opencv/{repo}/resolve/{revision}/{filename}"
                with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                verify(temporary, expected)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        verify(path, expected)
        print("Verified", path.name)


class FaceExtractor:
    def __init__(self, directory=DEFAULT_DIR):
        import cv2
        self.cv = cv2
        paths = []
        for relative, expected in MODELS.items():
            path = Path(directory) / Path(relative).name
            verify(path, expected)
            paths.append(str(path))
        self.detector = cv2.FaceDetectorYN.create(paths[0], "", (640, 480), .9, .3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(paths[1], "")

    def extract(self, encoded, *, body_box=None):
        import numpy as np
        from PIL import Image, UnidentifiedImageError
        try:
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > 800_000:
                raise ValueError("Camera frame is too large")
            with Image.open(BytesIO(raw)) as source:
                if source.format not in {"JPEG", "PNG"} or not (160 <= source.width <= 1280 and 120 <= source.height <= 960):
                    raise ValueError("Use a JPEG/PNG camera frame between 160x120 and 1280x960")
                rgb = np.array(source.convert("RGB"))
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid camera image") from exc
        frame = self.cv.cvtColor(rgb, self.cv.COLOR_RGB2BGR)
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            from hearth.monitor.face_demo import FaceNotVisible
            raise FaceNotVisible("No face visible")
        if len(faces) != 1:
            raise ValueError("Show exactly one face to the camera")
        face = faces[0]
        if body_box is not None:
            x, y = (face[0] + face[2] / 2) / frame.shape[1], (face[1] + face[3] / 2) / frame.shape[0]
            if not (body_box.x1 <= x <= body_box.x2 and body_box.y1 <= y <= body_box.y2):
                raise ValueError("Face does not belong to the detected body")
        if min(face[2:4]) < 70:
            raise ValueError("Move closer: the face is too small")
        aligned = self.recognizer.alignCrop(frame, face)
        if self.cv.Laplacian(self.cv.cvtColor(aligned, self.cv.COLOR_BGR2GRAY), self.cv.CV_64F).var() < 15:
            raise ValueError("Hold still and improve the lighting")
        feature = self.recognizer.feature(aligned).flatten().astype(float)
        norm = np.linalg.norm(feature)
        if not np.isfinite(feature).all() or norm <= 0:
            raise ValueError("Could not extract a face template")
        return (feature / norm).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    if args.check:
        import numpy as np
        extractor = FaceExtractor()
        _, faces = extractor.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        feature = extractor.recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))
        print("Models loaded; blank-frame faces:", 0 if faces is None else len(faces), "embedding shape:", feature.shape)

"""Camera capture + poseNet on the Jetson. The only module (with face.py) that imports jetson_*.

Each read() returns one frame's poses in the plain format features.py expects:
    {"keypoints": {name: (x_px, y_px)}, "bbox": (left, top, right, bottom)}

Optional `stream` sends the annotated video to a viewer on the laptop (e.g.
rtp://192.168.55.100:1234, watched with edge/tools/view_camera.sh). It is off by default,
only goes to that address over the local link, and is never recorded or sent to the backend.

Standalone check on the Jetson (prints posture numbers about twice a second):
    python3 -m hearth_edge.pose [--seconds 20] [--stream rtp://192.168.55.100:1234]
"""

import time

try:  # jetson-inference renamed its Python modules; support both spellings.
    from jetson_inference import poseNet
    from jetson_utils import videoSource, videoOutput
except ImportError:  # pragma: no cover - older installs
    from jetson.inference import poseNet
    from jetson.utils import videoSource, videoOutput


class PoseCamera(object):
    def __init__(self, camera="/dev/video0", width=640, height=480, model="resnet18-body",
                 threshold=0.15, stream=None):
        self.net = poseNet(model, threshold=threshold)
        self.camera = videoSource(camera, argv=[
            "--input-codec=mjpeg",
            "--input-width=%d" % width,
            "--input-height=%d" % height,
        ])
        self.output = videoOutput(stream, argv=["--headless"]) if stream else None

    def read(self):
        """Capture and run poseNet on one frame. Returns None if no frame arrived (timeout)."""
        try:
            img = self.camera.Capture()
        except Exception:  # some jetson_utils versions raise on capture timeout
            return None
        if img is None:
            return None
        t = time.time()
        overlay = "links,keypoints" if self.output is not None else "none"
        raw_poses = self.net.Process(img, overlay=overlay)
        poses = []
        for p in raw_poses:
            kps = {}
            for k in p.Keypoints:
                kps[self.net.GetKeypointName(k.ID)] = (k.x, k.y)
            poses.append({"keypoints": kps, "bbox": (p.Left, p.Top, p.Right, p.Bottom)})
        return {
            "t": t,
            "img": img,  # kept on the Jetson; only face.py and the optional stream use it
            "width": img.width,
            "height": img.height,
            "poses": poses,
            "inference_ms": self.net.GetNetworkTime(),
        }

    def render(self, frame):
        """Send the annotated frame to the optional viewer stream."""
        if self.output is not None:
            self.output.Render(frame["img"])

    def is_open(self):
        return self.camera.IsStreaming()

    def close(self):
        try:
            self.camera.Close()
        except Exception:
            pass
        if self.output is not None:
            try:
                self.output.Close()
            except Exception:
                pass


def _main():
    import argparse
    from hearth_edge import features

    ap = argparse.ArgumentParser(description="Print live posture features from the camera.")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--stream", default=None, help="e.g. rtp://192.168.55.100:1234")
    args = ap.parse_args()

    cam = PoseCamera(camera=args.camera, stream=args.stream)
    velocity = features.VerticalVelocity()
    start = last_print = time.time()
    frames = seen = 0
    try:
        while time.time() - start < args.seconds:
            frame = cam.read()
            if frame is None:
                continue
            frames += 1
            cam.render(frame)
            pose = features.select_pose(frame["poses"])
            if pose is None:
                velocity.reset()
                feats = None
            else:
                seen += 1
                feats = features.compute_features(pose, frame["width"], frame["height"],
                                                  velocity, frame["t"])
            if frame["t"] - last_print >= 0.5:
                last_print = frame["t"]
                if pose is None:
                    print("no person")
                else:
                    print("%2d keypoints  %s" % (len(pose["keypoints"]), feats))
    finally:
        cam.close()
    elapsed = time.time() - start
    print("frames %d in %.1fs = %.1f FPS, person in %d" % (frames, elapsed, frames / elapsed, seen))


if __name__ == "__main__":
    _main()

"""Camera capture + poseNet on the Jetson. The only module (with face.py) that imports jetson_*.

Each read() returns one frame's poses in the plain format features.py expects:
    {"keypoints": {name: (x_px, y_px)}, "bbox": (left, top, right, bottom)}

Optional `stream` sends the annotated video to a viewer on the laptop (e.g.
rtp://192.168.55.100:1234, watched with edge/tools/view_camera.sh). It is off by default,
only goes to that address over the local link, and is never recorded or sent to the backend.
The skeleton is drawn only in render(), never in read(): the face check copies the frame in
between, and must see the clean camera image, not skeleton lines drawn over the face.

Standalone check on the Jetson (prints posture numbers about twice a second):
    python3 -m hearth_edge.pose [--seconds 20] [--stream rtp://192.168.55.100:1234]
"""

import time

try:  # jetson-inference renamed its Python modules; support both spellings.
    from jetson_inference import poseNet
    from jetson_utils import (videoSource, videoOutput, cudaAllocMapped, cudaToNumpy,
                              cudaDeviceSynchronize, cudaDrawCircle, cudaDrawLine, cudaResize)
except ImportError:  # pragma: no cover - older installs
    from jetson.inference import poseNet
    from jetson.utils import (videoSource, videoOutput, cudaAllocMapped, cudaToNumpy,
                              cudaDeviceSynchronize, cudaDrawCircle, cudaDrawLine, cudaResize)

from hearth_edge.pose_rotate import largest, scale_pose, unrotate_pose

AMBER = (255, 190, 0, 255)
# COCO-18 bones, for drawing a pose found on a rotated frame (poseNet's own overlay can't).
BONES = (("nose", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"))


class PoseCamera(object):
    def __init__(self, camera="/dev/video0", width=640, height=480, model="resnet18-body",
                 threshold=0.15, stream=None, fps=None, plain=False):
        self.net = poseNet(model, threshold=threshold)
        self.plain = plain  # stream the camera image without drawing on it (the dashboard draws)
        argv = ["--input-codec=mjpeg", "--input-width=%d" % width, "--input-height=%d" % height]
        if fps:
            argv.append("--input-rate=%d" % fps)
        self.camera = videoSource(camera, argv=argv)
        self.output = videoOutput(stream, argv=["--headless"]) if stream else None
        self._small = self._rot_buf = self._rot_view = None  # reused buffers for rotated passes

    def read(self, run_pose=True):
        """Capture and run poseNet on one frame. Returns None if no frame arrived (timeout).

        run_pose=False captures only (no upright poseNet pass): used while someone stays lying
        down and the rotated pass is doing the work."""
        try:
            img = self.camera.Capture()
        except Exception:  # some jetson_utils versions raise on capture timeout
            return None
        if img is None:
            return None
        t = time.time()
        if not run_pose:
            return {"t": t, "img": img, "width": img.width, "height": img.height,
                    "poses": [], "raw_poses": [], "inference_ms": 0.0, "pose_skipped": True}
        raw_poses = self.net.Process(img, overlay="none")  # drawing happens later, in render()
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
            "raw_poses": raw_poses,  # poseNet's own objects, for drawing the skeleton in render()
            "inference_ms": self.net.GetNetworkTime(),
        }

    def _plain(self, raw_poses):
        out = []
        for p in raw_poses:
            kps = dict((self.net.GetKeypointName(k.ID), (k.x, k.y)) for k in p.Keypoints)
            out.append({"keypoints": kps, "bbox": (p.Left, p.Top, p.Right, p.Bottom)})
        return out

    def rotated_pose(self, frame, k):
        """poseNet on the frame rotated 90 degrees (k=+1 counter-clockwise, -1 clockwise).

        Returns (largest pose mapped back to the original frame or None, milliseconds). A person
        lying on the floor looks upright in the rotated frame, which poseNet handles far better.
        """
        poses, ms = self.rotated_poses(frame, k)
        return largest(poses), ms

    def rotated_poses(self, frame, k):
        """Like rotated_pose, but every pose found, all mapped back: (poses, milliseconds)."""
        import numpy as np
        start = time.time()
        img = frame["img"]
        # poseNet shrinks its input to 224x224 anyway, so shrink to half size on the GPU first
        # and rotate that: a quarter of the pixels for the CPU to move (~27 ms -> a few ms).
        w2, h2 = img.width // 2, img.height // 2
        if self._small is None:
            self._small = cudaAllocMapped(width=w2, height=h2, format="rgb8")
            self._rot_buf = cudaAllocMapped(width=h2, height=w2, format="rgb8")
            self._rot_view = cudaToNumpy(self._rot_buf)
        cudaResize(img, self._small)
        cudaDeviceSynchronize()  # the GPU must have finished before the CPU reads the result
        np.copyto(self._rot_view, np.rot90(cudaToNumpy(self._small), k))
        poses = self._plain(self.net.Process(self._rot_buf, overlay="none"))
        back = [scale_pose(unrotate_pose(p, k, w2, h2), img.width / float(w2)) for p in poses]
        return back, (time.time() - start) * 1000.0

    def render(self, frame):
        """Draw the skeleton and send the frame to the optional viewer stream.

        Call this after anything that needs the clean image (the face check copies it first).
        """
        if self.output is None:
            return
        img = frame["img"]
        if self.plain:
            self.output.Render(img)
            return
        override = frame.get("pose_override")
        if override is not None:  # a pose found on a rotated frame: draw it ourselves, in amber
            try:
                kps = override["keypoints"]
                for a, b in BONES:
                    if a in kps and b in kps:
                        cudaDrawLine(img, tuple(map(int, kps[a])), tuple(map(int, kps[b])), AMBER, 3)
                for x, y in kps.values():
                    cudaDrawCircle(img, (int(x), int(y)), 5, AMBER)
            except Exception:
                pass
            self.output.Render(img)
            return
        try:
            self.net.Overlay(img, frame["raw_poses"], overlay="links,keypoints")
        except TypeError:  # older bindings take width/height positionally
            try:
                self.net.Overlay(img, frame["raw_poses"], img.width, img.height, "links,keypoints")
            except Exception:
                pass  # drawing must never stop capture
        except Exception:
            pass
        self.output.Render(img)

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

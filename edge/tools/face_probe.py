#!/usr/bin/env python3
"""Measure how often res10 finds a face: whole frame vs head close-up, at several cutoffs.

Easiest: run edge/tools/face_test.sh on the laptop (camera window live the whole time).
Direct use on the Jetson (camera must be free):
    cd ~/hearth-watch/edge
    python3 tools/face_probe.py --label 3m 2>/dev/null                 # measure once, 15 s
    python3 tools/face_probe.py --control-stdin --stream rtp://192.168.55.100:1234
        # session: live view; send "probe LABEL SECONDS [COUNTDOWN]" or "quit" on stdin

Uses the exact detector code run_edge.py uses (hearth_edge/face.py). Only frames where poseNet
sees a head are counted. Video preview: skeleton, the head close-up area (yellow) and faces
found in it while measuring (red, score >= 0.3). Nothing is saved; no image leaves the Jetson.
(The earlier facenet / haar / res10 comparison is recorded in edge/BENCHMARKS.md.)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hearth_edge import config, features  # noqa: E402
from hearth_edge.control import read_control  # noqa: E402

CUTOFFS = (0.3, 0.4, 0.5)


class Measurement(object):
    def __init__(self, label):
        self.label = label
        self.frames = self.with_head = 0
        self.whole, self.close = [], []
        self.ms_whole, self.ms_close, self.head_px, self.face_px, self.brightness = [], [], [], [], []

    def table(self):
        def pct(scores, cut):
            return 100.0 * sum(s >= cut for s in scores) / len(scores)

        def med(xs):
            return sorted(xs)[len(xs) // 2] if xs else 0

        label = " [%s]" % self.label if self.label else ""
        lines = ["face probe%s: %d frames, %d with a head in view" % (label, self.frames, self.with_head)]
        if not self.with_head:
            return "\n".join(lines + ["no head keypoints seen: face the camera and run again"]) + "\n"
        lines.append("| res10 input | ms per check | %s |" % " | ".join("found at >= %.1f" % c for c in CUTOFFS))
        lines.append("|---|---|%s" % ("---|" * len(CUTOFFS)))
        for name, scores, ms in (("whole frame", self.whole, self.ms_whole),
                                 ("head close-up", self.close, self.ms_close)):
            lines.append("| %s | %.0f | %s |" % (name, sum(ms) / len(ms),
                                                 " | ".join("%.0f%%" % pct(scores, c) for c in CUTOFFS)))
        lines.append("head box (px): median %d | face box when found (px): median %d | "
                     "head-area brightness (0-255): %.0f"
                     % (med(self.head_px), med(self.face_px), sum(self.brightness) / len(self.brightness)))
        return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="res10 face probe: whole frame vs head close-up.")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--label", default="", help="e.g. close, 3m, mouth-covered")
    ap.add_argument("--stream", default=None, help="e.g. rtp://192.168.55.100:1234 (video preview)")
    ap.add_argument("--control-stdin", action="store_true",
                    help="session mode: live view, measure on 'probe LABEL SECONDS [COUNTDOWN]'")
    args = ap.parse_args(argv)

    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # jetson-inference logs go to stderr

    from hearth_edge.face import YELLOW, FaceDetector, head_box
    from hearth_edge.pose import PoseCamera

    def say(text):
        out.write(text + "\n")
        out.flush()

    cam = PoseCamera(camera=config.CAMERA, width=config.WIDTH, height=config.HEIGHT,
                     fps=config.CAMERA_FPS, stream=args.stream)
    # Low cutoff: record every score and judge later. draw only matters with --stream.
    det = FaceDetector("res10", threshold=0.05, draw=bool(args.stream))

    def head_roi(frame):
        pose = features.select_pose(frame["poses"])
        return head_box(pose["keypoints"], frame["width"], frame["height"]) if pose else None

    def measure_frame(frame, meas):
        meas.frames += 1
        roi = head_roi(frame)
        if roi is None:
            return
        meas.with_head += 1
        bgr = det.frame_to_bgr(frame["img"])  # clean copy, before anything is drawn
        boxes, ms = det.detect_bgr(bgr)
        meas.whole.append(max([b[4] for b in boxes] or [0.0]))
        meas.ms_whole.append(ms)
        boxes_c, ms = det.detect_bgr(bgr, roi)
        meas.close.append(max([b[4] for b in boxes_c] or [0.0]))
        meas.ms_close.append(ms)
        meas.head_px.append(roi[2] - roi[0])
        best = max(boxes + boxes_c, key=lambda b: b[4]) if boxes + boxes_c else None
        if best is not None and best[4] >= 0.3:
            meas.face_px.append(int(max(best[2] - best[0], best[3] - best[1])))
        l, t, r, b = roi
        meas.brightness.append(float(bgr[t:b, l:r].mean()))
        if args.stream:
            det.draw_box(frame, roi, YELLOW, "head close-up")
            det.last = (frame["t"], [x for x in boxes_c if x[4] >= 0.3])
            det.draw(frame)

    try:
        if not args.control_stdin:
            meas = Measurement(args.label)
            sys.stderr.write("face the camera now: probing for %.0f s\n" % args.seconds)
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                frame = cam.read()
                if frame is not None:
                    measure_frame(frame, meas)
                    cam.render(frame)
            out.write(meas.table())
            return 0 if meas.with_head else 1

        say("control: ready (commands: probe LABEL SECONDS [COUNTDOWN] | quit)")
        active = None
        while True:
            line = read_control(sys.stdin)
            if line == "":
                break
            if line:
                parts = line.split()
                if parts == ["quit"]:
                    break
                if len(parts) in (3, 4) and parts[0] == "probe" and active is None:
                    countdown = float(parts[3]) if len(parts) == 4 else 3.0
                    now = time.time()
                    active = {"meas": Measurement(parts[1]), "start": now + countdown,
                              "end": now + countdown + float(parts[2]), "announced": None}
            frame = cam.read()
            if frame is None:
                continue
            now = time.time()
            if active is None or now < active["start"]:
                if active is not None:
                    left = int(active["start"] - now) + 1
                    if left != active["announced"]:
                        active["announced"] = left
                        say("probe '%s' starts in %d" % (active["meas"].label, left))
                roi = head_roi(frame)  # live view: where the head close-up would be
                if roi is not None and args.stream:
                    det.draw_box(frame, roi, YELLOW, "head close-up")
            elif now < active["end"]:
                if active["announced"] != 0:
                    active["announced"] = 0
                    say("measuring '%s'..." % active["meas"].label)
                measure_frame(frame, active["meas"])
            else:
                out.write(active["meas"].table())
                say("probe done: %s" % active["meas"].label)
                active = None
            cam.render(frame)
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Measure how often res10 finds a face: whole frame vs head close-up, at several cutoffs.

Uses the exact detector code run_edge.py uses (hearth_edge/face.py). Run it on the Jetson while
facing the camera, once per situation you care about, with a label so the output says which:
    cd ~/hearth-watch/edge
    python3 tools/face_probe.py --label close        2>/dev/null
    python3 tools/face_probe.py --label 3m           2>/dev/null
    python3 tools/face_probe.py --label mouth-covered 2>/dev/null
To watch while it runs, open the video window on the laptop first
(edge/tools/view_camera.sh --player-only) and add --stream rtp://192.168.55.100:1234: you see
the skeleton, the head close-up area (yellow) and faces found in it (red, score >= 0.3).
Only frames where poseNet sees a head are counted. Nothing is saved or recorded.
(Earlier comparison of facenet / haar / res10 is recorded in edge/BENCHMARKS.md.)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hearth_edge import config, features  # noqa: E402

CUTOFFS = (0.3, 0.4, 0.5)


def main(argv=None):
    ap = argparse.ArgumentParser(description="res10 face probe: whole frame vs head close-up.")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--label", default="", help="e.g. close, 3m, mouth-covered")
    ap.add_argument("--stream", default=None, help="e.g. rtp://192.168.55.100:1234 (video preview)")
    args = ap.parse_args(argv)

    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # jetson-inference logs go to stderr

    from hearth_edge.face import YELLOW, FaceDetector, head_box
    from hearth_edge.pose import PoseCamera

    cam = PoseCamera(camera=config.CAMERA, width=config.WIDTH, height=config.HEIGHT,
                     fps=config.CAMERA_FPS, stream=args.stream)
    # Low cutoff: record every score and judge later. draw only matters with --stream.
    det = FaceDetector("res10", threshold=0.05, draw=bool(args.stream))
    whole, close = [], []
    ms_whole, ms_close, head_px, face_px, brightness = [], [], [], [], []
    frames = with_head = 0
    sys.stderr.write("face the camera now: probing for %.0f s\n" % args.seconds)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        frame = cam.read()
        if frame is None:
            continue
        frames += 1
        pose = features.select_pose(frame["poses"])
        roi = head_box(pose["keypoints"], frame["width"], frame["height"]) if pose else None
        if roi is None:
            cam.render(frame)
            continue
        with_head += 1
        bgr = det.frame_to_bgr(frame["img"])
        boxes, ms = det.detect_bgr(bgr)
        whole.append(max([b[4] for b in boxes] or [0.0]))
        ms_whole.append(ms)
        boxes_c, ms = det.detect_bgr(bgr, roi)
        close.append(max([b[4] for b in boxes_c] or [0.0]))
        ms_close.append(ms)
        head_px.append(roi[2] - roi[0])
        best = max(boxes + boxes_c, key=lambda b: b[4]) if boxes + boxes_c else None
        if best is not None and best[4] >= 0.3:
            face_px.append(int(max(best[2] - best[0], best[3] - best[1])))
        l, t, r, b = roi
        brightness.append(float(bgr[t:b, l:r].mean()))
        if args.stream:  # draw after the clean copy (bgr) was taken
            det.draw_box(frame, roi, YELLOW, "head close-up")
            det.last = (frame["t"], [x for x in boxes_c if x[4] >= 0.3])
            det.draw(frame)
            cam.render(frame)
    cam.close()

    def pct(scores, cut):
        return 100.0 * sum(s >= cut for s in scores) / len(scores)

    def med(xs):
        return sorted(xs)[len(xs) // 2] if xs else 0

    label = " [%s]" % args.label if args.label else ""
    out.write("face probe%s: %d frames, %d with a head in view\n" % (label, frames, with_head))
    if not with_head:
        out.write("no head keypoints seen: face the camera and run again\n")
        return 1
    out.write("| res10 input | ms per check | %s |\n" % " | ".join("found at >= %.1f" % c for c in CUTOFFS))
    out.write("|---|---|%s\n" % ("---|" * len(CUTOFFS)))
    for name, scores, ms in (("whole frame", whole, ms_whole), ("head close-up", close, ms_close)):
        out.write("| %s | %.0f | %s |\n" % (name, sum(ms) / len(ms),
                                           " | ".join("%.0f%%" % pct(scores, c) for c in CUTOFFS)))
    out.write("head box (px): median %d | face box when found (px): median %d | "
              "head-area brightness (0-255): %.0f\n"
              % (med(head_px), med(face_px), sum(brightness) / len(brightness)))
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

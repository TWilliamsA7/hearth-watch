#!/usr/bin/env python3
"""Benchmark the edge pipeline on the Jetson: FPS, model time and memory, with and without faces.

Runs three modes back to back on the live camera (same process, models loaded once):
    pose only | pose + face check every --face-interval s | pose + face check every frame
and prints a Markdown table of MEASURED numbers only, ready for edge/BENCHMARKS.md.

Run on the Jetson (camera must be free; stand in view to also count detections):
    cd ~/hearth-watch/edge && python3 tools/bench.py [--seconds 20]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hearth_edge import config, features  # noqa: E402
from hearth_edge.client import read_mem_used_mb  # noqa: E402


def p95(values):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def mean(values):
    return sum(values) / len(values) if values else None


def run_mode(cam, face, seconds, face_interval):
    """face_interval: None = no face checks, 0 = every frame, >0 = at most once per interval."""
    from hearth_edge.face import FaceSchedule
    schedule = FaceSchedule(face_interval) if face_interval else None
    frames, pose_ms, face_ms, frame_ms = 0, [], [], []
    persons = face_checks = faces = 0
    peak_mem = 0.0
    t0 = time.time()
    while time.time() - t0 < seconds:
        start = time.time()
        frame = cam.read()
        if frame is None:
            continue
        frames += 1
        pose_ms.append(frame["inference_ms"])
        if features.select_pose(frame["poses"]) is not None:
            persons += 1
        if face_interval is not None and (face_interval == 0 or schedule.due(frame["t"])):
            present, _conf, ms = face.check(frame)
            face_ms.append(ms)
            face_checks += 1
            faces += 1 if present else 0
        frame_ms.append((time.time() - start) * 1000.0)
        peak_mem = max(peak_mem, read_mem_used_mb() or 0.0)
    elapsed = time.time() - t0
    return {"fps": frames / elapsed, "frames": frames, "pose_ms": mean(pose_ms),
            "pose_p95": p95(pose_ms), "face_ms": mean(face_ms), "frame_p95": p95(frame_ms),
            "peak_mem": peak_mem, "persons": persons, "face_checks": face_checks, "faces": faces}


def fmt(v, f="%.1f"):
    return "-" if v is None else f % v


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark pose and face detection on the Jetson.")
    ap.add_argument("--seconds", type=float, default=20.0, help="per mode")
    ap.add_argument("--face-interval", type=float, default=config.FACE_INTERVAL_S)
    args = ap.parse_args(argv)

    # jetson-inference logs to stdout; keep stdout for the table.
    sys.stdout.flush()
    table_out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)

    from hearth_edge.face import FaceDetector
    from hearth_edge.pose import PoseCamera

    mem_before = read_mem_used_mb()
    cam = PoseCamera(camera=config.CAMERA, width=config.WIDTH, height=config.HEIGHT,
                     model=config.POSE_MODEL, threshold=config.POSE_THRESHOLD)
    mem_pose = read_mem_used_mb()
    face = FaceDetector(config.FACE_MODEL, config.FACE_THRESHOLD)
    mem_face = read_mem_used_mb()
    for _ in range(30):  # warm-up: the first frames are slow while TensorRT settles
        frame = cam.read()
        if frame is not None:
            face.check(frame)

    modes = [("pose only", None),
             ("pose + face every %.1f s" % args.face_interval, args.face_interval),
             ("pose + face every frame", 0)]
    rows = []
    try:
        for name, interval in modes:
            sys.stderr.write("benchmarking: %s (%.0f s)\n" % (name, args.seconds))
            rows.append((name, run_mode(cam, face, args.seconds, interval)))
    finally:
        cam.close()

    out = table_out
    out.write("Measured %s on Jetson Nano 2GB (JetPack 4.6.1), %s %dx%d MJPEG, %s + %s, "
              "%.0f s per mode after a 30-frame warm-up.\n\n"
              % (time.strftime("%Y-%m-%d %H:%M"), config.CAMERA, config.WIDTH, config.HEIGHT,
                 config.POSE_MODEL, config.FACE_MODEL, args.seconds))
    out.write("Memory in use (system, MB): before models %s, after poseNet %s, after the face model %s.\n\n"
              % (fmt(mem_before, "%.0f"), fmt(mem_pose, "%.0f"), fmt(mem_face, "%.0f")))
    out.write("| Mode | FPS | Pose ms (mean / p95) | Face ms (mean) | Frame ms p95 | Peak memory MB "
              "| Frames with person | Face checks / faces found |\n")
    out.write("|---|---|---|---|---|---|---|---|\n")
    for name, r in rows:
        out.write("| %s | %s | %s / %s | %s | %s | %s | %d of %d | %s |\n" % (
            name, fmt(r["fps"]), fmt(r["pose_ms"]), fmt(r["pose_p95"]), fmt(r["face_ms"]),
            fmt(r["frame_p95"]), fmt(r["peak_mem"], "%.0f"), r["persons"], r["frames"],
            "%d / %d" % (r["face_checks"], r["faces"]) if r["face_checks"] else "-"))
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

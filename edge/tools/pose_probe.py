#!/usr/bin/env python3
"""Does rotating the frame help poseNet see people lying down? Measure it on live frames.

Easiest: run edge/tools/pose_test.sh on the laptop (camera window live the whole time).
Direct use on the Jetson (camera must be free):
    cd ~/hearth-watch/edge
    python3 tools/pose_probe.py --label across 2>/dev/null          # measure once, 15 s
    python3 tools/pose_probe.py --control-stdin --stream rtp://192.168.55.100:1234
        # session: live view; send "probe LABEL SECONDS [COUNTDOWN]" or "quit" on stdin

While measuring, poseNet runs three times per frame: on the frame as-is, rotated 90 degrees
counter-clockwise and rotated 90 degrees clockwise (keypoints mapped back with
hearth_edge/pose_rotate.py). "best of all" keeps whichever found the most keypoints. It also tries poseNet with lower
keypoint thresholds (0.08 and 0.04 instead of 0.15) on the unrotated frame. The table
shows how often the head, neck/shoulders, hips, and all three together were found, the keypoint
count and the torso angle. Video preview: green dots = plain poseNet, amber = best rotated
result. Nothing is saved; no image leaves the Jetson.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hearth_edge import config, features  # noqa: E402
from hearth_edge.control import read_control  # noqa: E402
from hearth_edge.pose_rotate import unrotate_pose  # noqa: E402

HEAD = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
UPPER = ("neck", "left_shoulder", "right_shoulder")
HIPS = ("left_hip", "right_hip")
METHODS = ("as-is", "threshold 0.08", "threshold 0.04", "rotated CCW", "rotated CW", "best of all")
DEFAULT_THRESHOLD = 0.15
GREEN = (60, 230, 60, 255)
AMBER = (255, 190, 0, 255)


def to_plain(net, raw_poses):
    return [{"keypoints": dict((net.GetKeypointName(k.ID), (k.x, k.y)) for k in p.Keypoints),
             "bbox": (p.Left, p.Top, p.Right, p.Bottom)} for p in raw_poses]


def describe(pose):
    if pose is None:
        return None
    names = set(pose["keypoints"])
    has = lambda group: bool(names & set(group))  # noqa: E731
    return {"n": len(names), "head": has(HEAD), "upper": has(UPPER), "hips": has(HIPS),
            "torso": features.torso_angle_deg(pose["keypoints"])}


class Measurement(object):
    def __init__(self, label):
        self.label = label
        self.rows = dict((m, []) for m in METHODS)
        self.best_counts = dict((m, 0) for m in METHODS[:-1])
        self.extra_ms = []
        self.frames = 0

    def add(self, candidates, best_name, ms):
        self.frames += 1
        self.extra_ms.append(ms)
        if best_name:
            self.best_counts[best_name] += 1
        for m in METHODS:
            self.rows[m].append(describe(candidates[m]))

    def table(self):
        label = " [%s]" % self.label if self.label else ""
        lines = ["pose probe%s: %d frames, the extra variants add %.0f ms per frame"
                 % (label, self.frames, sum(self.extra_ms) / len(self.extra_ms) if self.extra_ms else 0)]
        if not self.frames:
            return "\n".join(lines + ["no frames"]) + "\n"
        lines.append("| method | person found | keypoints (median) | head | neck/shoulders | hips "
                     "| head+shoulders+hips | torso deg (median) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in METHODS:
            rows = self.rows[m]
            seen = [r for r in rows if r is not None]

            def pct(key):
                return "%.0f%%" % (100.0 * sum(1 for r in seen if r[key]) / len(rows))
            allthree = 100.0 * sum(1 for r in seen if r["head"] and r["upper"] and r["hips"]) / len(rows)
            torsos = [r["torso"] for r in seen if r["torso"] is not None]
            lines.append("| %s | %.0f%% | %s | %s | %s | %s | %.0f%% | %s |" % (
                m, 100.0 * len(seen) / len(rows),
                statistics.median([r["n"] for r in seen]) if seen else "-",
                pct("head"), pct("upper"), pct("hips"), allthree,
                "%.0f" % statistics.median(torsos) if torsos else "-"))
        lines.append("best of all picked: %s" % ", ".join("%s %d" % kv for kv in self.best_counts.items()))
        return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="poseNet on rotated frames: does it help lying poses?")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--stream", default=None, help="e.g. rtp://192.168.55.100:1234")
    ap.add_argument("--control-stdin", action="store_true",
                    help="session mode: live view, measure on 'probe LABEL SECONDS [COUNTDOWN]'")
    args = ap.parse_args(argv)

    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # jetson-inference logs go to stderr

    import numpy as np
    import jetson_utils as ju
    from hearth_edge.pose import PoseCamera

    def say(text):
        out.write(text + "\n")
        out.flush()

    cam = PoseCamera(camera=config.CAMERA, width=config.WIDTH, height=config.HEIGHT,
                     fps=config.CAMERA_FPS, stream=args.stream)
    net = cam.net

    def draw(frame, pose, colour, radius):
        if args.stream and pose is not None:
            for x, y in pose["keypoints"].values():
                ju.cudaDrawCircle(frame["img"], (int(x), int(y)), radius, colour)

    def measure_frame(frame, meas):
        w, h = frame["width"], frame["height"]
        candidates = {"as-is": features.select_pose(frame["poses"])}
        start = time.time()
        for name, thr in (("threshold 0.08", 0.08), ("threshold 0.04", 0.04)):
            net.SetThreshold(thr)
            candidates[name] = features.select_pose(to_plain(net, net.Process(frame["img"], overlay="none")))
        net.SetThreshold(DEFAULT_THRESHOLD)
        ju.cudaDeviceSynchronize()
        rgb = ju.cudaToNumpy(frame["img"])[..., :3]
        for name, k in (("rotated CCW", 1), ("rotated CW", -1)):
            rot = ju.cudaFromNumpy(np.ascontiguousarray(np.rot90(rgb, k)).astype(np.uint8))
            pose = features.select_pose(to_plain(net, net.Process(rot, overlay="none")))
            candidates[name] = unrotate_pose(pose, k, w, h) if pose else None
        ms = (time.time() - start) * 1000.0
        found = [(m, p) for m, p in candidates.items() if p is not None]
        best_name, best = (max(found, key=lambda mp: len(mp[1]["keypoints"])) if found else (None, None))
        candidates["best of all"] = best
        meas.add(candidates, best_name, ms)
        draw(frame, candidates["as-is"], GREEN, 5)
        draw(frame, best, AMBER, 8)

    try:
        if not args.control_stdin:
            meas = Measurement(args.label)
            sys.stderr.write("lie on the mat now: probing for %.0f s\n" % args.seconds)
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                frame = cam.read()
                if frame is not None:
                    measure_frame(frame, meas)
                    cam.render(frame)
            out.write(meas.table())
            return 0

        say("control: ready (commands: probe LABEL SECONDS [COUNTDOWN] | quit)")
        active = None  # {"meas", "start", "end", "announced"}
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
                draw(frame, features.select_pose(frame["poses"]), GREEN, 5)  # live view: plain poseNet
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

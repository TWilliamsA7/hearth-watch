#!/usr/bin/env python3
"""Explain what the team backend would do with a recording, and why.

Runs on the LAPTOP with the repo's .venv (needs shared/ and backend/ installed):
    .venv/bin/python edge/tools/check_rules.py recordings/lie_down-*.jsonl
    .venv/bin/python edge/tools/check_rules.py recordings/*.jsonl --frames   # per-frame detail

For each clip it reports:
  1. VERDICT - Person 2's real FallDetector (backend/hearth/monitor/detector.py) run on the clip
     with the real thresholds (backend/hearth/monitor/config.json): possible-fall alerts and
     recoveries it would emit, and when.
  2. WHY - for every frame with a person, each condition of the backend's two posture rules
     (mirrored from detector.py) and how often it failed:
        standing: torso <= upright_angle_max_deg, head above hips, hips above floor_hip_y_min
        floor:    floor_angle_min..max_deg torso, head >= floor_head_y_min, hips >= floor_hip_y_min,
                  box aspect >= floor_bbox_aspect_min (or unmeasured)
     plus which body parts poseNet found, the range of each number, and a timeline.
The verdict is authoritative; the per-condition counts are an explanation that mirrors the rules.
"""

import argparse
import json
import os
import statistics
import sys

HEAD = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
UPPER = ("neck", "left_shoulder", "right_shoulder")
HIPS = ("left_hip", "right_hip")
LEGS = ("left_knee", "right_knee", "left_ankle", "right_ankle")


def load_observations(path):
    from hearth_schema import Observation
    raw, models, bad = [], [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                models.append(Observation.model_validate(d))
                raw.append(d)
            except Exception:
                bad += 1
    return raw, models, bad


def frame_conditions(f, c):
    """Each rule condition for one frame's features (dict). Returns (standing_ok, floor_ok, fails)."""
    t, h, p, a = (f.get("torso_angle_deg"), f.get("head_y"), f.get("hip_y"), f.get("bbox_aspect"))
    fails = []
    if t is None:
        fails.append("no torso angle (needs neck/shoulders + a hip)")
    elif not (c.floor_angle_min_deg <= t <= c.floor_angle_max_deg):
        fails.append("torso not %g-%g deg" % (c.floor_angle_min_deg, c.floor_angle_max_deg))
    if h is None:
        fails.append("no head (nose/eyes/ears)")
    elif h < c.floor_head_y_min:
        fails.append("head above the %.2f line" % c.floor_head_y_min)
    if p is None:
        fails.append("no hips")
    elif p < c.floor_hip_y_min:
        fails.append("hips above the %.2f line" % c.floor_hip_y_min)
    if a is not None and a < c.floor_bbox_aspect_min:
        fails.append("box narrower than %.1f" % c.floor_bbox_aspect_min)
    standing = (t is not None and t <= c.upright_angle_max_deg and h is not None and p is not None
                and h < p < c.floor_hip_y_min)
    return standing, not fails, fails


def run_real_detector(models, config):
    from hearth.monitor.detector import FallDetector
    det = FallDetector(config)
    t0 = models[0].timestamp
    out = []
    for m in models:
        for e in det.process(m):
            if e.event_type in ("possible_fall", "recovered"):
                out.append(((e.timestamp - t0).total_seconds(), e.event_type, e.metadata))
    return out


def timeline(raw, codes, bucket_s=0.5):
    """One character per bucket: S standing, F floor, ? person but neither, - no person."""
    from hearth_edge import contract
    if not raw:
        return ""
    t0 = contract.parse_utc_iso(raw[0]["timestamp"])
    buckets = {}
    for o, code in zip(raw, codes):
        i = int((contract.parse_utc_iso(o["timestamp"]) - t0) / bucket_s)
        buckets.setdefault(i, []).append(code)
    order = "FS?-"  # if any frame in the bucket is floor, show F, etc.
    return "".join(min(buckets.get(i, ["-"]), key=order.index) for i in range(max(buckets) + 1))


def report(path, config, show_frames=False, out=sys.stdout):
    raw, models, bad = load_observations(path)
    w = out.write
    w("\n=== %s\n" % os.path.basename(path))
    if not models:
        w("no valid observations\n")
        return None
    span = (models[-1].timestamp - models[0].timestamp).total_seconds()
    person = [o for o in raw if o["person_detected"]]
    w("%d observations over %.1f s (%.1f/s), person in %d%s\n"
      % (len(raw), span, (len(raw) - 1) / span if span else 0, len(person),
         ", %d invalid lines skipped" % bad if bad else ""))

    events = run_real_detector(models, config)
    alerts = [e for e in events if e[1] == "possible_fall"]
    if alerts:
        for t, kind, meta in events:
            w("VERDICT: %-13s at %5.1f s %s\n" % (kind, t, "(floor for %.1f s)" % meta.get("duration_seconds", 0)
                                                    if kind == "possible_fall" else ""))
    else:
        w("VERDICT: no possible-fall alert (backend needs %.0f s of floor-level frames, gaps <= %.0f s)\n"
          % (config.floor_hold_seconds, config.max_observation_gap_seconds))

    codes, fail_counts, n_standing, n_floor = [], {}, 0, 0
    for o in raw:
        if not o["person_detected"]:
            codes.append("-")
            continue
        standing, floor, fails = frame_conditions(o.get("features") or {}, config)
        n_standing += standing
        n_floor += floor
        codes.append("F" if floor else "S" if standing else "?")
        for reason in fails:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1
    if person:
        w("frames matching: standing %d | floor %d | neither %d (of %d with a person)\n"
          % (n_standing, n_floor, len(person) - n_standing - n_floor, len(person)))
        w("floor rule, why frames failed:\n")
        for reason, n in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
            w("  %3d%%  %s\n" % (round(100.0 * n / len(person)), reason))
        names = [set(k["name"] for k in (o["keypoints"] or [])) for o in person]
        w("body parts found (%% of person frames): head %d | neck/shoulders %d | hips %d | knees/ankles %d\n"
          % tuple(round(100.0 * sum(bool(s & set(group)) for s in names) / len(names))
                  for group in (HEAD, UPPER, HIPS, LEGS)))
        for key, label in (("torso_angle_deg", "torso deg"), ("head_y", "head height"),
                           ("hip_y", "hip height"), ("bbox_aspect", "box aspect")):
            vals = [o["features"][key] for o in person if (o.get("features") or {}).get(key) is not None]
            if vals:
                w("  %-11s min %6.2f  median %6.2f  max %6.2f  (%d frames)\n"
                  % (label, min(vals), statistics.median(vals), max(vals), len(vals)))
            else:
                w("  %-11s never measured\n" % label)
    w("timeline (0.5 s per char; S standing, F floor, ? neither, - no person):\n  %s\n"
      % timeline(raw, codes))
    if show_frames:
        from hearth_edge import contract
        t0 = contract.parse_utc_iso(raw[0]["timestamp"])
        for o, code in zip(raw, codes):
            f = o.get("features") or {}
            w("  %5.1f s %s  torso %-6s head %-5s hip %-5s box %-5s  parts %s\n" % (
                contract.parse_utc_iso(o["timestamp"]) - t0, code,
                *("-" if f.get(k) is None else "%.2f" % f[k]
                  for k in ("torso_angle_deg", "head_y", "hip_y", "bbox_aspect")),
                ",".join(k["name"] for k in (o["keypoints"] or []))))
    return {"alerts": len(alerts), "events": events, "standing": n_standing, "floor": n_floor,
            "person": len(person), "fails": fail_counts}


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, os.path.join(repo, "edge"))
    ap = argparse.ArgumentParser(description="Explain the backend's fall verdict for recordings.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--config", default=os.path.join(repo, "backend", "hearth", "monitor", "config.json"),
                    help="backend monitor config (default: the repo's)")
    ap.add_argument("--frames", action="store_true", help="also print every frame")
    args = ap.parse_args(argv)
    try:
        from hearth.monitor.config import MonitorConfig
    except ImportError:
        sys.stderr.write("needs the team backend: .venv/bin/pip install -e 'backend[dev]'\n")
        return 1
    config = MonitorConfig.load(args.config)
    print("thresholds from %s: floor hold %.0f s, torso %g-%g deg, head/hip below %.2f/%.2f, box >= %.1f; "
          "standing: torso <= %g deg, hips above %.2f"
          % (os.path.relpath(args.config, repo), config.floor_hold_seconds, config.floor_angle_min_deg,
             config.floor_angle_max_deg, config.floor_head_y_min, config.floor_hip_y_min,
             config.floor_bbox_aspect_min, config.upright_angle_max_deg, config.floor_hip_y_min))
    for path in args.files:
        report(path, config, args.frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())

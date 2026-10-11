#!/usr/bin/env python3
"""Replays recorded observations into a backend, clearly labelled as replay.

Every message is rewritten before sending:
    source      -> "replay-01" (never the live camera's name, so their online status can't mix)
    source_type -> "replay"    (README invariant: every record says where it came from)
    run_id      -> a new id per playback pass (also on heartbeats), so repeated runs of the same
                   clip stay separate in the backend's history and evaluation
    obs_id, timestamp -> fresh ids and the current time, so the backend's timing works normally
Use --speed 1 when comparing alert timing: faster playback shortens time spent on the floor.
Everything else (keypoints, features, face_present) is sent exactly as recorded. The original
spacing between observations is kept (scaled by --speed). Heartbeats are sent while playing and
stop afterwards, so the backend shows the replay source going offline when playback ends.

Standard library only (works on the laptop or the Jetson). From the repo root on the laptop:
    .venv/bin/python edge/tools/replay.py recordings/lie_down-20261010-031500.jsonl
    .venv/bin/python edge/tools/replay.py recordings/*.jsonl --speed 2
    .venv/bin/python edge/tools/replay.py recordings/sit-*.jsonl --loop     # Ctrl+C to stop
    # Demo fallback: standing, lying down (alert), standing again (recovery) as ONE run:
    .venv/bin/python edge/tools/replay.py --join recordings/stand-*.jsonl \
        recordings/lie_down-20261010-063944.jsonl recordings/stand-*.jsonl
    # Two people at once (p1 stands; p2 stands, lies down, gets up), with our posture model's call:
    .venv/bin/python edge/tools/replay.py --together --posture-mlp \
        recordings/demo/stand.jsonl+recordings/demo/stand.jsonl+recordings/demo/stand.jsonl \
        recordings/demo/stand.jsonl+recordings/demo/lie_down.jsonl+recordings/demo/stand.jsonl
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hearth_edge import contract  # noqa: E402
from hearth_edge.client import EdgeClient  # noqa: E402

REPLAY_SOURCE = "replay-01"


def load_recording(path):
    """Observations from a JSONL recording, in file order. Unreadable lines are skipped."""
    observations = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obs = json.loads(line)
                contract.parse_utc_iso(obs["timestamp"])
            except (ValueError, KeyError, TypeError) as e:
                sys.stderr.write("%s:%d skipped (%s)\n" % (path, n, e))
                continue
            observations.append(obs)
    return observations


def schedule(observations, speed=1.0):
    """Seconds after playback start at which each observation is sent."""
    if not observations:
        return []
    t0 = contract.parse_utc_iso(observations[0]["timestamp"])
    return [max(0.0, (contract.parse_utc_iso(o["timestamp"]) - t0) / speed) for o in observations]


def join_recordings(recordings, gap_s=0.1):
    """Several clips -> one continuous clip: each starts gap_s after the previous one ends.

    Original timestamps are shifted (copies are returned), so playback, the backend's hold
    timer and recovery all see one uninterrupted run. gap_s stays well under the backend's
    1 s observation-gap limit.
    """
    joined, end = [], None
    for observations in recordings:
        if not observations:
            continue
        first = contract.parse_utc_iso(observations[0]["timestamp"])
        shift = 0.0 if end is None else (end + gap_s) - first
        for o in observations:
            o = dict(o)
            o["timestamp"] = contract.utc_iso(contract.parse_utc_iso(o["timestamp"]) + shift)
            joined.append(o)
        end = contract.parse_utc_iso(joined[-1]["timestamp"])
    return joined


def together_recordings(recordings, spread=0.5):
    """Several clips -> one clip of several people at once, as the edge sends them.

    Clip i becomes person "p<i+1>" (any person_id in it is replaced); all start together, and
    each is moved sideways in the frame (keypoint x and model box shifted, clamped to 0-1) so
    the people stand apart: with spread 0.5, two people sit a quarter-frame left and right.
    """
    n = len(recordings)
    if n == 0:
        return []
    start = contract.parse_utc_iso(recordings[0][0]["timestamp"])
    merged = []
    for i, observations in enumerate(recordings):
        shift = 0.0 if n == 1 else spread * (i / (n - 1.0) - 0.5)
        first = contract.parse_utc_iso(observations[0]["timestamp"])
        for o in observations:
            o = dict(o)
            o["timestamp"] = contract.utc_iso(start + contract.parse_utc_iso(o["timestamp"]) - first)
            o["person_id"] = "p%d" % (i + 1)
            if o.get("keypoints"):
                o["keypoints"] = [dict(k, x=round(_unit(k["x"] + shift), 4)) for k in o["keypoints"]]
            box = (o.get("posture_prediction") or {}).get("bbox")
            if box:
                o["posture_prediction"] = dict(o["posture_prediction"], bbox=dict(
                    box, x1=_unit(box["x1"] + shift), x2=_unit(box["x2"] + shift)))
            merged.append(o)
    merged.sort(key=lambda o: contract.parse_utc_iso(o["timestamp"]))
    return merged


def add_posture_model(observations, mlp, width=640, height=480):
    """Copies with our posture model's call (posture_mlp.PostureMLP) on each observation's
    keypoints, as run_edge.py sends it: for recordings made before the model existed."""
    out = []
    for o in observations:
        o = dict(o)
        if o.get("keypoints"):
            kps = dict((k["name"], (k["x"] * width, k["y"] * height)) for k in o["keypoints"])
            xs, ys = [p[0] for p in kps.values()], [p[1] for p in kps.values()]
            pose = {"keypoints": kps, "bbox": (min(xs), min(ys), max(xs), max(ys))}
            o["posture_prediction"] = mlp.person_prediction(pose, mlp.predict(kps, width, height),
                                                            width, height)
        out.append(o)
    return out


def _unit(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def new_run_id(source, n=1, clock=time.time):
    """e.g. replay-01-20261010-043512-1: unique per playback pass."""
    return "%s-%s-%d" % (source, time.strftime("%Y%m%d-%H%M%S", time.localtime(clock())), n)


def rewrite_for_replay(obs, obs_id, ts, source=REPLAY_SOURCE, run_id=None):
    out = dict(obs)
    out["obs_id"] = obs_id
    out["timestamp"] = contract.utc_iso(ts)
    out["source"] = source
    out["source_type"] = "replay"
    out["scenario_id"] = None  # recordings are live captures, never simulator scenarios
    if run_id is not None:
        out["run_id"] = run_id
    else:
        out.pop("run_id", None)
    return out


class PlaybackStats(object):
    """Heartbeat data while playing; None (heartbeats pause) when not."""

    def __init__(self):
        self.playing = False
        self.fps = None

    def snapshot(self):
        if not self.playing:
            return None
        return {"fps": self.fps, "inference_ms": None, "mem_used_mb": None}


def recording_rate(observations):
    if len(observations) < 2:
        return None
    span = (contract.parse_utc_iso(observations[-1]["timestamp"])
            - contract.parse_utc_iso(observations[0]["timestamp"]))
    return (len(observations) - 1) / span if span > 0 else None


def play(client, observations, ids, speed=1.0, source=REPLAY_SOURCE, run_id=None,
         clock=time.time, sleep=time.sleep):
    """Submit each observation at its scheduled time. Returns how many were submitted."""
    t0 = clock()
    sent = 0
    for obs, offset in zip(observations, schedule(observations, speed)):
        wait = t0 + offset - clock()
        if wait > 0:
            sleep(wait)
        client.submit(rewrite_for_replay(obs, ids.next(), clock(), source, run_id))
        sent += 1
    return sent


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay recorded observations as source_type 'replay'.")
    ap.add_argument("files", nargs="+", help="recordings/*.jsonl files, played in order; "
                    "a+b+c plays a, b and c back to back (one person, with --together)")
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--source", default=REPLAY_SOURCE)
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    ap.add_argument("--loop", action="store_true", help="repeat until Ctrl+C")
    ap.add_argument("--join", action="store_true",
                    help="play all files back to back as ONE run (one run_id), e.g. stand, "
                         "lie_down, stand: needed for the backend to link a recovery to its alert")
    ap.add_argument("--together", action="store_true",
                    help="play the files at the same time as different people (p1, p2, ...), side "
                         "by side, in ONE run; join clips for one person with + (a+b+c)")
    ap.add_argument("--posture-mlp", action="store_true",
                    help="add our posture model's call to every observation (for recordings made "
                         "before run_edge.py sent it; needs numpy)")
    args = ap.parse_args(argv)
    if args.speed <= 0:
        ap.error("--speed must be positive")
    if args.join and args.together:
        ap.error("--join and --together don't mix: join one person's clips with +")

    recordings = []
    for arg in args.files:
        parts = [obs for obs in (load_recording(p) for p in arg.split("+")) if obs]
        if parts:
            recordings.append((arg, join_recordings(parts) if len(parts) > 1 else parts[0]))
    if not recordings:
        sys.stderr.write("nothing to replay\n")
        return 1
    if args.together:
        recordings = [(" & ".join("+".join(os.path.basename(f) for f in p.split("+")) for p, _ in recordings),
                       together_recordings([obs for _, obs in recordings]))]
    elif args.join and len(recordings) > 1:
        recordings = [(" + ".join(os.path.basename(p) for p, _ in recordings),
                       join_recordings([obs for _, obs in recordings]))]
    if args.posture_mlp:
        from hearth_edge.posture_mlp import PostureMLP
        mlp = PostureMLP()
        recordings = [(p, add_posture_model(obs, mlp)) for p, obs in recordings]

    room = recordings[0][1][0].get("room_id", "room-01")
    stats = PlaybackStats()
    client = EdgeClient(args.backend, args.source, room, stats=stats, source_type="replay",
                        idle_reason="replay finished")
    ids = contract.ObservationIds(args.source)
    client.start()
    total = 0
    passes = 0
    try:
        while True:
            for path, observations in recordings:
                passes += 1
                run_id = new_run_id(args.source, passes)
                client.run_id = run_id
                duration = schedule(observations, args.speed)[-1]
                print("replaying %s: %d observations over %.1f s (x%g) as %s/replay, run %s"
                      % (os.path.basename(path), len(observations), duration, args.speed,
                         args.source, run_id), flush=True)
                rate = recording_rate(observations)
                stats.fps = rate * args.speed if rate else None
                stats.playing = True
                total += play(client, observations, ids, args.speed, args.source, run_id)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stats.playing = False
        end = time.time() + 5.0
        while client.queued() and time.time() < end and client.online is not False:
            time.sleep(0.05)
        client.stop()
    c = client.counts
    print("done: submitted %d | sent %d dropped %d rejected %d | backend %s"
          % (total, c["sent"], c["dropped"], c["rejected"],
             {None: "not reached", True: "online", False: "OFFLINE"}[client.online]), flush=True)
    return 0 if c["rejected"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

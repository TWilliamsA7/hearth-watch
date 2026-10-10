#!/usr/bin/env python3
"""Hearth edge node: camera -> poseNet -> posture features -> Observations to the laptop backend.

Run on the Jetson (from the edge/ folder):
    python3 run_edge.py                      # send to the backend in config.py
    python3 run_edge.py --dry-run            # print observations instead of sending
    python3 run_edge.py --stream rtp://192.168.55.100:1234   # also stream the skeleton video
    python3 run_edge.py --help               # every option

Stop with Ctrl+C. Settings default to hearth_edge/config.py; options override them.
Face detection is not wired in yet, so face_present is null ("did not run") on every frame.
"""

import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hearth_edge import config, contract, features  # noqa: E402
from hearth_edge.client import EdgeClient, FrameStats  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Hearth edge node (Jetson).")
    ap.add_argument("--backend", default=config.BACKEND_URL, help="backend base URL")
    ap.add_argument("--source", default=config.SOURCE_ID)
    ap.add_argument("--room", default=config.ROOM_ID)
    ap.add_argument("--camera", default=config.CAMERA)
    ap.add_argument("--width", type=int, default=config.WIDTH)
    ap.add_argument("--height", type=int, default=config.HEIGHT)
    ap.add_argument("--threshold", type=float, default=config.POSE_THRESHOLD,
                    help="poseNet keypoint threshold")
    ap.add_argument("--obs-hz", type=float, default=config.OBS_HZ,
                    help="observations sent per second (frames beyond this are not sent)")
    ap.add_argument("--stream", default=None,
                    help="also stream the skeleton video, e.g. rtp://192.168.55.100:1234")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each observation as JSON instead of sending anything")
    ap.add_argument("--status-every", type=float, default=5.0, help="seconds between status lines")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = run until stopped)")
    return ap.parse_args(argv)


class RateLimiter(object):
    """Lets through about `hz` events per second from a faster, uneven stream of frames.

    Schedules the next slot from the previous slot (not from the last frame), so a 14 FPS
    camera still yields ~10 observations/s instead of every other frame (~7/s). After a stall
    it restarts from now instead of bursting to catch up.
    """

    def __init__(self, hz):
        self.period = 1.0 / hz if hz > 0 else 0.0
        self.next_t = None

    def ready(self, t):
        if self.next_t is None or self.period == 0.0:
            self.next_t = t + self.period
            return True
        if t < self.next_t:
            return False
        self.next_t += self.period
        if t - self.next_t > self.period:
            self.next_t = t + self.period
        return True


def observation_for_frame(ids, frame, pose, feats, source, room_id,
                          confidence=config.KEYPOINT_CONFIDENCE_PLACEHOLDER):
    """Build the Observation for one frame. pose/feats are None when nobody was detected."""
    if pose is None:
        return contract.build_observation(ids.next(), frame["t"], source, room_id,
                                          person_detected=False, face_present=None)
    keypoints = contract.normalize_keypoints(pose["keypoints"], frame["width"], frame["height"],
                                             confidence)
    return contract.build_observation(ids.next(), frame["t"], source, room_id,
                                      person_detected=True, keypoints=keypoints,
                                      features=feats, face_present=None)


def log(msg):
    sys.stderr.write("[edge %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stderr.flush()


def reserve_stdout_for_json():
    """jetson-inference's C++ code prints its log lines to stdout. Point file descriptor 1 at
    stderr so those go there, and return a private handle to the real stdout for our JSON."""
    sys.stdout.flush()
    json_out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    return json_out


def main(argv=None):
    args = parse_args(argv)
    json_out = reserve_stdout_for_json()  # before the Jetson libraries load and start printing
    from hearth_edge.pose import PoseCamera  # Jetson-only import, after --help works

    stop = {"flag": False}

    def request_stop(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    log("source=%s room=%s camera=%s %dx%d obs=%.0f/s %s"
        % (args.source, args.room, args.camera, args.width, args.height, args.obs_hz,
           "DRY RUN (nothing sent)" if args.dry_run else "backend=" + args.backend))

    cam = PoseCamera(camera=args.camera, width=args.width, height=args.height,
                     model=config.POSE_MODEL, threshold=args.threshold, stream=args.stream)
    stats = FrameStats()
    client = None
    if not args.dry_run:
        client = EdgeClient(args.backend, args.source, args.room, stats=stats,
                            observations_path=config.OBSERVATIONS_PATH,
                            heartbeat_path=config.HEARTBEAT_PATH,
                            batch_interval_s=config.BATCH_INTERVAL_S,
                            heartbeat_interval_s=config.HEARTBEAT_INTERVAL_S,
                            timeout_s=config.HTTP_TIMEOUT_S, queue_max=config.QUEUE_MAX)
        client.start()

    ids = contract.ObservationIds(args.source)
    velocity = features.VerticalVelocity(alpha=config.HIP_EMA_ALPHA,
                                         reset_gap_s=config.VELOCITY_RESET_GAP_S)
    limiter = RateLimiter(args.obs_hz)
    start = last_status = last_frame = time.time()
    frames = with_person = emitted = 0
    camera_quiet = False

    try:
        while not stop["flag"]:
            now = time.time()
            if args.seconds and now - start >= args.seconds:
                break
            frame = cam.read()
            if frame is None:
                if not camera_quiet and now - last_frame > 2.0:
                    log("no frames from the camera for 2 s (unplugged or busy?)")
                    camera_quiet = True
                continue
            if camera_quiet:
                log("camera frames resumed")
                camera_quiet = False
            last_frame = frame["t"]
            frames += 1
            stats.record(frame["inference_ms"], frame["t"])
            cam.render(frame)

            # Features every frame keeps the hip-velocity estimate smooth; only some are sent.
            pose = features.select_pose(frame["poses"])
            if pose is None:
                velocity.reset()
                feats = None
            else:
                with_person += 1
                feats = features.compute_features(pose, frame["width"], frame["height"],
                                                  velocity, frame["t"])

            if limiter.ready(frame["t"]):
                obs = observation_for_frame(ids, frame, pose, feats, args.source, args.room)
                emitted += 1
                if args.dry_run:
                    json_out.write(json.dumps(obs) + "\n")
                    json_out.flush()
                else:
                    client.submit(obs)

            if frame["t"] - last_status >= args.status_every:
                elapsed = frame["t"] - last_status
                snap = stats.snapshot() or {}
                line = "fps %.1f | person in %d%% of frames | observations %d" % (
                    snap.get("fps") or 0.0, 100 * with_person // max(frames, 1), emitted)
                if client is not None:
                    state = {None: "not tried", True: "online", False: "OFFLINE"}[client.online]
                    line += " | sent %(sent)d dropped %(dropped)d rejected %(rejected)d" % client.counts
                    line += " | queued %d | backend %s" % (client.queued(), state)
                log(line)
                last_status = frame["t"]
                frames = with_person = 0
    finally:
        log("stopping")
        if client is not None:
            client.stop()
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

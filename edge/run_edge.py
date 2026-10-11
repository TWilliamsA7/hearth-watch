#!/usr/bin/env python3
"""Hearth edge node: camera -> poseNet -> posture model + features -> Observations to the backend.

Run on the Jetson (from the edge/ folder):
    python3 run_edge.py                      # send to the backend in config.py
    python3 run_edge.py --dry-run            # print observations instead of sending
    python3 run_edge.py --stream rtp://192.168.55.100:1234   # also stream the skeleton video
    python3 run_edge.py --help               # every option

    python3 run_edge.py --dry-run --record sit --seconds 10   # save a clip to recordings/

Stop with Ctrl+C. Settings default to hearth_edge/config.py; options override them.
Face detection (presence only, never identity) starts a check every --face-interval seconds,
on a background thread for res10; each result goes on the next observation sent, and every
other observation carries face_present = null ("did not run"). --no-face turns it off.
Our posture model (hearth_edge/posture_mlp.py, on by default; --no-posture-mlp turns it off)
judges everyone in view each frame; its posture_prediction goes on every observation and is
what the team backend's alerts use (see edge/POSTURE_MODEL.md).
Everyone in view is followed (hearth_edge/people.py) and gets their own observations, each with
an anonymous person_id ("p1", "p2", ...) and that person's own posture_prediction, so the backend
watches each person separately. --single-person sends only the followed person, as before.
"""

import argparse
import collections
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hearth_edge import config, contract, features, pose_rotate  # noqa: E402
from hearth_edge.control import read_control  # noqa: E402,F401  (stdin commands)
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
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after this long (0 = run until stopped); with --record, the "
                         "length of the recording after the countdown")
    ap.add_argument("--record", metavar="LABEL", default=None,
                    help="save observations to recordings/LABEL-<time>.jsonl (keypoints only)")
    ap.add_argument("--record-dir", default=None, help="where recordings go (default <repo>/recordings)")
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="seconds of countdown before recording starts (default 3)")
    ap.add_argument("--control-stdin", action="store_true",
                    help="keep running and accept commands on stdin, one per line: "
                         "'record LABEL SECONDS [COUNTDOWN]' or 'quit' (used by record_clips.sh)")
    ap.add_argument("--no-face", action="store_true",
                    help="turn face detection off (face_present is then null on every frame)")
    ap.add_argument("--face-interval", type=float, default=config.FACE_INTERVAL_S,
                    help="seconds between face checks (default %(default)s)")
    ap.add_argument("--face-threshold", type=float, default=config.FACE_THRESHOLD,
                    help="face-model confidence needed to count as a face (default %(default)s)")
    ap.add_argument("--fall-model", action="store_true",
                    help="also run the Hugging Face fall model (TensorRT; edge/tools/get_fall_model.sh) "
                         "and send its posture_prediction on every observation")
    ap.add_argument("--fall-engine", default=None, help="TensorRT engine path (default models/fall/...)")
    ap.add_argument("--no-posture-mlp", dest="posture_mlp", action="store_false",
                    help="turn off our posture model (on by default: hearth_edge/posture_mlp.npz, "
                         "edge/POSTURE_MODEL.md; its call goes to the backend as posture_prediction "
                         "on every observation, and the HF --fall-model is then panel-only)")
    ap.add_argument("--posture-mlp", dest="posture_mlp", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-rotation", action="store_true",
                    help="never run rotated poseNet passes (used for people lying down)")
    ap.add_argument("--keypoint-hold", type=float, default=config.KEYPOINT_HOLD_S,
                    help="seconds a briefly missing joint is bridged for the posture numbers "
                         "(sent keypoints are never filled in); 0 = off (default %(default)s)")
    ap.add_argument("--camera-fps", type=int, default=config.CAMERA_FPS,
                    help="frames per second requested from the camera (default %(default)s)")
    ap.add_argument("--single-person", dest="multi_person", action="store_false",
                    default=config.MULTI_PERSON,
                    help="send only the followed person (no person_id), instead of one observation "
                         "per person in view")
    return ap.parse_args(argv)


def parse_control_line(line):
    """'record LABEL SECONDS [COUNTDOWN]' -> ('record', label, seconds, countdown or None);
    'quit' -> ('quit',); anything else -> None."""
    parts = line.split()
    if parts == ["quit"]:
        return ("quit",)
    if len(parts) in (3, 4) and parts[0] == "record":
        try:
            seconds = float(parts[2])
            countdown = float(parts[3]) if len(parts) == 4 else None
        except ValueError:
            return None
        if seconds > 0 and (countdown is None or countdown >= 0):
            return ("record", parts[1], seconds, countdown)
    return None


class RecordTimer(object):
    """A countdown, then a recording window of `seconds` (0 = until stopped). Frame times."""

    def __init__(self, start, countdown_s, seconds):
        self.rec_start = start + max(0.0, countdown_s)
        self.rec_end = self.rec_start + seconds if seconds else None

    def recording(self, t):
        return t >= self.rec_start and (self.rec_end is None or t < self.rec_end)

    def done(self, t):
        return self.rec_end is not None and t >= self.rec_end

    def countdown_left(self, t):
        """Whole seconds until recording starts (0 once it has started)."""
        left = self.rec_start - t
        return int(left) + (1 if left > int(left) else 0) if left > 0 else 0


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


def observation_for_frame(ids, frame, pose, feats, source, room_id, face_present=None,
                          face_confidence=None, confidence=config.KEYPOINT_CONFIDENCE,
                          posture_prediction=None):
    """Build the Observation for one frame. pose/feats are None when nobody was detected.

    face_present is None when no face check ran on this frame (not the same as False).
    posture_prediction (optional fall model): when it saw someone, person_detected is true even
    if poseNet didn't - otherwise the backend would ignore the model's evidence for that frame.
    """
    if pose is None:
        model_saw = bool(posture_prediction and posture_prediction.get("detection_count", 0) > 0)
        return contract.build_observation(ids.next(), frame["t"], source, room_id,
                                          person_detected=model_saw, face_present=face_present,
                                          face_confidence=face_confidence,
                                          posture_prediction=posture_prediction)
    keypoints = contract.normalize_keypoints(pose["keypoints"], frame["width"], frame["height"],
                                             confidence)
    return contract.build_observation(ids.next(), frame["t"], source, room_id,
                                      person_detected=True, keypoints=keypoints, features=feats,
                                      face_present=face_present, face_confidence=face_confidence,
                                      posture_prediction=posture_prediction)


def observations_for_people(ids, frame, people, ended, source, room_id, face_present=None,
                            face_confidence=None, briefly_missing=False,
                            confidence=config.KEYPOINT_CONFIDENCE):
    """Observations for one sent frame when everyone in view is reported separately.

    people: [(person_id, pose, feats, posture_prediction or None, is_tracked)]. Each gets its own
    observation with its person_id; the face check (aimed at the followed person) goes on the
    followed person's only. ended: person_ids that just left view; each gets one "not detected"
    observation so the backend knows they're gone. With nobody in view, one observation without
    a person_id says so (it keeps the camera's health fresh) - unless someone is only briefly
    missing, when that is skipped, as for a single followed person.
    """
    out = []
    for person_id, pose, feats, prediction, is_tracked in people:
        keypoints = contract.normalize_keypoints(pose["keypoints"], frame["width"], frame["height"],
                                                 confidence)
        out.append(contract.build_observation(
            ids.next(), frame["t"], source, room_id, person_detected=True, keypoints=keypoints,
            features=feats, face_present=face_present if is_tracked else None,
            face_confidence=face_confidence if is_tracked else None,
            posture_prediction=prediction, person_id=person_id))
    for person_id in ended:
        out.append(contract.build_observation(ids.next(), frame["t"], source, room_id,
                                              person_detected=False, person_id=person_id))
    if not people and not briefly_missing:
        out.append(contract.build_observation(ids.next(), frame["t"], source, room_id,
                                              person_detected=False, face_present=face_present,
                                              face_confidence=face_confidence))
    return out


def others_in_view(tracked, poses, max_dist):
    """Plain-pass poses of everyone except the tracked person.

    `tracked` is the followed person's pose (possibly from a rotated pass, with more keypoints);
    `poses` (the plain pass) include the tracked person's own plain-pass version: the one whose
    box centre is nearest the tracked pose (within max_dist) is dropped.
    """
    others = list(poses or [])
    if tracked is not None and others:
        tx, ty = features.bbox_center(tracked["bbox"])

        def dist(p):
            x, y = features.bbox_center(p["bbox"])
            return ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
        nearest = min(others, key=dist)
        if dist(nearest) <= max_dist:
            others.remove(nearest)
    return others


def order_people(tracked, others):
    """[(pose, is_tracked)] for everyone, left to right."""
    people = [(features.bbox_center(p["bbox"])[0], p, False) for p in others]
    if tracked is not None:
        people.append((features.bbox_center(tracked["bbox"])[0], tracked, True))
    people.sort(key=lambda item: item[0])
    return [(p, is_tracked) for _, p, is_tracked in people]


def people_in_view(tracked, poses, max_dist):
    """Everyone our posture model should look at, left to right: [(pose, is_tracked)]."""
    return order_people(tracked, others_in_view(tracked, poses, max_dist))


def log(msg):
    """Status line to stderr. Never raises: if whoever reads our output has gone away
    (closed ssh, killed pipe), logging must not crash the program before it shuts down."""
    try:
        sys.stderr.write("[edge %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        sys.stderr.flush()
    except (IOError, OSError, ValueError):
        pass


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
    signal.signal(signal.SIGHUP, request_stop)  # terminal/ssh closed: stop cleanly too

    log("source=%s room=%s camera=%s %dx%d obs=%.0f/s %s"
        % (args.source, args.room, args.camera, args.width, args.height, args.obs_hz,
           "DRY RUN (nothing sent)" if args.dry_run else "backend=" + args.backend))

    cam = PoseCamera(camera=args.camera, width=args.width, height=args.height,
                     model=config.POSE_MODEL, threshold=args.threshold, stream=args.stream,
                     fps=args.camera_fps)

    face = schedule = checker = None
    if not args.no_face:
        from hearth_edge.face import FaceDetector, FaceSchedule, head_box, summarize
        try:
            face = FaceDetector(config.FACE_MODEL, args.face_threshold, draw=bool(args.stream))
            schedule = FaceSchedule(args.face_interval)
            if face.model == "res10":
                checker = face.start_background()
            log("face detection on: %s, one check every %.1f s, threshold %.2f, %s"
                % (config.FACE_MODEL, args.face_interval, args.face_threshold,
                   "background thread" if checker else "in the camera loop"))
        except Exception as e:  # keep monitoring posture even if the face model can't load
            log("FACE DETECTION UNAVAILABLE (%s): face_present stays null" % e)
            face = None
    else:
        log("face detection off (--no-face): face_present stays null")
    face_checks = faces_seen = 0
    face_ms_total = 0.0

    fall_model = fall_runner = fall_schedule = None
    latest_fall = None      # (posture_prediction, time of the frame it checked)
    fall_checks = 0
    fall_ms_total = 0.0
    if args.fall_model:
        from hearth_edge.face import AsyncFaceChecker, FaceSchedule as Every
        from hearth_edge.fall_model import FallModel, frame_to_rgb
        try:
            fall_model = FallModel(args.fall_engine, config.FALL_MODEL_SIZE)
            # Same background-thread runner as the face check: copies the frame, runs off-loop.
            fall_runner = AsyncFaceChecker(lambda rgb, roi: fall_model.predict_rgb(rgb), frame_to_rgb)
            fall_schedule = Every(config.FALL_INTERVAL_S)
            log("fall model on: %s, every %.1f s, posture_prediction on every observation"
                % (fall_model.engine_path, config.FALL_INTERVAL_S))
        except Exception as e:
            log("FALL MODEL UNAVAILABLE (%s): continuing with pose evidence only" % e)
            fall_model = fall_runner = None
    mlp = None
    mlp_counts = {}         # labels from our posture model since the last status line
    mlp_people = []         # its call on every person in view, left to right
    mlp_backend = None      # its PosturePrediction for the backend (every observation)
    others_rot = pose_rotate.OthersRotation()   # rotated poses for people other than the tracked one
    crowd = None            # PeopleTracker: everyone in view, each with their own person_id
    crowd_now = []          # (person_id, pose, feats, posture_prediction, is_tracked) this frame
    if args.multi_person:
        from hearth_edge.people import PeopleTracker
        crowd = PeopleTracker(config.TRACK_MAX_DIST * args.width, config.TRACK_KEEP_S,
                              alpha=config.HIP_EMA_ALPHA, reset_gap_s=config.VELOCITY_RESET_GAP_S,
                              hold_s=args.keypoint_hold)
        log("everyone in view: one observation per person, each with its own person_id "
            "(--single-person to send only the followed person)")
    last_plain = None       # (plain-pass poses, time): reused on frames where that pass is skipped
    if args.posture_mlp:
        from hearth_edge.posture_mlp import PostureMLP
        try:
            mlp = PostureMLP()
            log("our posture model on: %s (from %s, cut-off %.2f); sent to the backend as %s"
                % (mlp.path, mlp.source, mlp.min_conf, "posture_prediction (%s)" % mlp.revision))
            if fall_runner is not None:
                log("the Hugging Face fall model is shown on the panel only (ours goes to the backend)")
        except Exception as e:
            log("OUR POSTURE MODEL UNAVAILABLE (%s): continuing without it" % e)
    last_face = None        # (present, confidence, frame time) of the latest face result
    pending_face = None     # finished background result waiting for the next sent observation
    face_failed_logged = False
    stats = FrameStats()

    # Preview panel (only when streaming): edge measurements + the backend's own view.
    overlay = backend_view = None
    emit_times = collections.deque()
    if args.stream:
        from hearth_edge import overlay as panel
        overlay = panel.Overlay()
        if not args.dry_run:
            from hearth_edge.backend_view import BackendView
            backend_view = BackendView(args.backend, args.source, config.BACKEND_POLL_S)
            backend_view.start()
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
    rotation = (pose_rotate.RotationPolicy()
                if config.POSE_ROTATION and not args.no_rotation else None)
    hold = features.KeypointHold(args.keypoint_hold)
    track = features.PersonTrack(config.TRACK_KEEP_S, config.TRACK_DROPOUT_S)
    held = []
    limiter = RateLimiter(args.obs_hz)
    start = last_status = last_frame = time.time()
    frames = with_person = emitted = 0
    camera_quiet = False

    # Recording: one clip from --record, or many on command with --control-stdin.
    rec = {"recorder": None, "timer": None, "label": None, "announced": None}

    def start_recording(label, seconds, countdown, t):
        from hearth_edge.recorder import Recorder
        rec["recorder"] = Recorder(label, args.record_dir)
        rec["timer"] = RecordTimer(t, countdown, seconds)
        rec["label"], rec["announced"] = label, None

    def finish_recording():
        r = rec["recorder"]
        r.close()
        log("recording done: %d observations -> %s" % (r.count, r.path))
        rec["recorder"] = rec["timer"] = rec["label"] = None

    if args.record:
        start_recording(args.record, args.seconds, args.countdown, start)
    control = sys.stdin if args.control_stdin else None
    if control is not None:
        log("control: ready (commands: record LABEL SECONDS [COUNTDOWN] | quit)")

    try:
        while not stop["flag"]:
            now = time.time()
            if control is not None:
                line = read_control(control)
                if line == "":  # the sender closed the pipe (script ended or ssh dropped)
                    break
                if line:
                    cmd = parse_control_line(line)
                    if cmd is None:
                        log("control: ignored %r" % line.strip())
                    elif cmd[0] == "quit":
                        break
                    elif rec["timer"] is not None:
                        log("control: already recording '%s'" % rec["label"])
                    else:
                        _, label, seconds, countdown = cmd
                        start_recording(label, seconds,
                                        args.countdown if countdown is None else countdown, now)
            timer = rec["timer"]
            if timer is not None:
                if timer.done(now):
                    finish_recording()
                    if control is None:
                        break  # --record: one clip, then stop
                    continue
                left = timer.countdown_left(now)
                if left != rec["announced"]:
                    rec["announced"] = left
                    log("RECORDING '%s' -> %s" % (rec["label"], rec["recorder"].path) if left == 0
                        else "recording '%s' starts in %d" % (rec["label"], left))
            elif args.seconds and now - start >= args.seconds:
                break
            skip_upright = rotation is not None and rotation.skip_upright(now)
            frame = cam.read(run_pose=not skip_upright)
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

            # Features every frame keeps the hip-velocity estimate smooth; only some are sent.
            # Follow the same person as a moment ago, so a background person can't take over.
            pose = features.select_pose(frame["poses"], track.near(frame["t"]),
                                        config.TRACK_MAX_DIST * frame["width"])
            pose_source, rotation_ms = None, 0.0
            rotated_seen = []       # every pose any rotated pass found this frame
            if rotation is not None:
                # People lying down: poseNet sees them far better on a 90-degree rotated frame.
                tries = rotation.directions(pose, frame["t"])
                for k in tries:
                    found, ms = cam.rotated_poses(frame, k)
                    rotated_seen += found
                    candidate = pose_rotate.largest(found)
                    rotation_ms += ms
                    if pose_rotate.better_rotated(pose, candidate):
                        pose = candidate
                        pose_source = "rotated CCW" if k == 1 else "rotated CW"
                        frame["pose_override"] = candidate  # what the preview draws
                        rotation.accepted(k, frame["t"])
                        break
                else:
                    if tries:
                        rotation.rejected(frame["t"])
            dropout = pose is None and track.brief_dropout(frame["t"])
            track.update(pose, frame["t"])
            if pose is None:
                velocity.reset()
                feats = None
                held = []
            else:
                with_person += 1
                # Posture numbers use briefly-held joints too; the observation's keypoints
                # (built from `pose` below) stay exactly what poseNet reported.
                merged, held = hold.update(pose["keypoints"], frame["t"])
                feats = features.compute_features({"keypoints": merged, "bbox": pose["bbox"]},
                                                  frame["width"], frame["height"],
                                                  velocity, frame["t"])

            people = []
            if mlp is not None or crowd is not None:
                # Everyone in view, for our own model (0.6 ms each, on poseNet's raw keypoints like
                # the field test, so someone on the floor is seen even while the edge tracks
                # someone else) and for per-person observations.
                # Someone else lying down gets rotated poses too (OthersRotation): reused from the
                # tracked person's passes, plus one extra pass at most every 0.5-1 s.
                if frame.get("pose_skipped") and last_plain and frame["t"] - last_plain[1] <= 0.5:
                    plain = last_plain[0]
                else:
                    plain = frame["poses"]
                    last_plain = (plain, frame["t"])
                others = others_in_view(pose, plain, config.TRACK_MAX_DIST * frame["width"])
                k = others_rot.wants_pass(others, frame["t"], bool(rotated_seen) or
                                          (rotation is not None and bool(tries)))
                if k is not None and rotation is not None:
                    found, ms = cam.rotated_poses(frame, k)
                    rotated_seen += found
                    rotation_ms += ms
                others_rot.remember(rotated_seen, frame["t"])
                others = others_rot.merge(pose, others, frame["t"])
                people = order_people(pose, others)
            if mlp is not None:
                mlp_people = [mlp.predict(p["keypoints"], frame["width"], frame["height"])
                              for p, _ in people]
                mlp_backend = mlp.backend_prediction(
                    [(p, tracked, r) for (p, tracked), r in zip(people, mlp_people)],
                    frame["width"], frame["height"])
                for r in mlp_people:
                    key = r["label"] if r is not None else "too few"
                    mlp_counts[key] = mlp_counts.get(key, 0) + 1
                if any(r is not None and r["label"] == "floor" for r in mlp_people):
                    mlp_counts["SOMEONE ON FLOOR (frames)"] = mlp_counts.get("SOMEONE ON FLOOR (frames)", 0) + 1
            if crowd is not None:
                # Each person's posture numbers come from their own history (velocity, held joints).
                crowd_now = []
                calls = mlp_people if mlp is not None else [None] * len(people)
                for (person, p), (_, is_tracked), r in zip(
                        crowd.assign([p for p, _ in people], frame["t"]), people, calls):
                    merged, _ = person.hold.update(p["keypoints"], frame["t"])
                    f = features.compute_features({"keypoints": merged, "bbox": p["bbox"]},
                                                  frame["width"], frame["height"],
                                                  person.velocity, frame["t"])
                    prediction = (mlp.person_prediction(p, r, frame["width"], frame["height"])
                                  if mlp is not None else None)
                    crowd_now.append((person.person_id, p, f, prediction, is_tracked))

            # A brief dropout of the followed person is not sent as "no person" (see PersonTrack).
            # With everyone reported, each person's own brief dropouts are skipped instead.
            emit = limiter.ready(frame["t"]) and (crowd is not None or not dropout)
            face_present = face_confidence = None
            inference_ms = frame["inference_ms"] + rotation_ms
            if face is not None:
                roi = (head_box(pose["keypoints"], frame["width"], frame["height"])
                       if pose is not None else None)
                if checker is not None:
                    # Background (res10): start a check when idle and due; collect finished ones.
                    if checker.idle() and schedule.due(frame["t"]):
                        checker.submit(frame, roi)
                    result = checker.poll()
                    if result is not None:
                        _t, boxes, face_ms = result
                        if boxes is None:
                            if not face_failed_logged:
                                log("a face check failed; face_present stays null for it")
                                face_failed_logged = True
                        else:
                            face.last = (frame["t"], boxes)  # what the preview draws
                            pending_face = summarize(boxes)
                            last_face = pending_face + (frame["t"],)
                            face_checks += 1
                            faces_seen += 1 if boxes else 0
                            face_ms_total += face_ms
                    if emit and pending_face is not None:
                        face_present, face_confidence = pending_face
                        pending_face = None
                elif emit and schedule.due(frame["t"]):
                    # In-loop (facenet, GPU): check only frames that are sent.
                    face_present, face_confidence, face_ms = face.check(frame, roi)
                    last_face = (face_present, face_confidence, frame["t"])
                    inference_ms += face_ms
                    face_checks += 1
                    faces_seen += 1 if face_present else 0
                    face_ms_total += face_ms
            if fall_runner is not None:
                if fall_runner.idle() and fall_schedule.due(frame["t"]):
                    fall_runner.submit(frame)
                done = fall_runner.poll()    # (time of the checked frame, prediction, ms)
                if done is not None and done[1] is not None:
                    latest_fall = (done[1], done[0])
                    fall_ms_total += done[2]
                    fall_checks += 1
            fall_now = (latest_fall[0] if latest_fall is not None
                        and frame["t"] - latest_fall[1] <= config.FALL_MAX_AGE_S else None)
            stats.record(inference_ms, frame["t"])
            if face is not None:
                face.draw(frame)  # holds the last check's boxes so they don't flicker
            if emit:
                emit_times.append(frame["t"])
            while emit_times and frame["t"] - emit_times[0] > 1.0:
                emit_times.popleft()
            if overlay is not None:
                if args.dry_run:
                    mode = "DRY RUN (nothing sent)"
                else:
                    mode = "backend " + {None: "not tried", True: "online",
                                         False: "OFFLINE"}[client.online]
                kps = pose["keypoints"] if pose is not None else {}
                lines = [
                    (panel.status_line((stats.snapshot() or {}).get("fps"), len(emit_times), mode,
                                       time.strftime("%H:%M:%S")), panel.WHITE),
                    (panel.person_line(pose is not None, len(kps),
                                       None if last_face is None else
                                       (last_face[0], last_face[1], frame["t"] - last_face[2]),
                                       len(held), pose_source),
                     panel.WHITE),
                    (panel.features_line(feats), panel.WHITE),
                ]
                if latest_fall is not None:
                    fall_line = panel.fall_model_line(latest_fall[0], frame["t"] - latest_fall[1])
                    lines.append((fall_line, panel.CYAN))
                if mlp is not None and mlp_people:
                    text, someone_down = panel.posture_mlp_people_line(mlp_people)
                    lines.append((text, panel.AMBER if someone_down else panel.GREEN))
                lines += panel.backend_lines(backend_view.snapshot() if backend_view else None,
                                             args.source)
                overlay.draw(frame, lines, features.head_point(kps) if kps else None,
                             features.hip_point(kps) if kps else None)
            cam.render(frame)

            if emit:
                if crowd is not None:
                    batch = observations_for_people(
                        ids, frame, crowd_now, [p.person_id for p in crowd.ended(frame["t"])],
                        args.source, args.room, face_present, face_confidence,
                        briefly_missing=bool(crowd.missing(frame["t"])))
                else:
                    batch = [observation_for_frame(ids, frame, pose, feats, args.source, args.room,
                                                   face_present, face_confidence,
                                                   posture_prediction=mlp_backend if mlp is not None
                                                   else fall_now)]
                emitted += 1
                for obs in batch:
                    if rec["timer"] is not None and rec["timer"].recording(frame["t"]):
                        rec["recorder"].write(obs)
                    if args.dry_run:
                        json_out.write(json.dumps(obs) + "\n")
                        json_out.flush()
                    else:
                        client.submit(obs)

            if frame["t"] - last_status >= args.status_every:
                snap = stats.snapshot() or {}
                line = "fps %.1f | person in %d%% of frames | observations %d" % (
                    snap.get("fps") or 0.0, 100 * with_person // max(frames, 1), emitted)
                if crowd is not None:
                    line += " | people in view %d" % len(crowd_now)
                if face is None:
                    line += " | face off"
                else:
                    line += " | face seen in %d of %d checks" % (faces_seen, face_checks)
                    if face_checks:
                        line += " (%.0f ms each)" % (face_ms_total / face_checks)
                if fall_runner is not None:
                    line += " | fall model %d checks" % fall_checks
                    if fall_checks:
                        line += " (%.0f ms each), last %s" % (fall_ms_total / fall_checks,
                                                              latest_fall[0]["label"])
                if mlp is not None:
                    line += " | our model %s" % (" ".join("%s %d" % kv for kv in sorted(mlp_counts.items()))
                                                 or "no person")
                    mlp_counts = {}
                if client is not None:
                    state = {None: "not tried", True: "online", False: "OFFLINE"}[client.online]
                    line += " | sent %(sent)d dropped %(dropped)d rejected %(rejected)d" % client.counts
                    line += " | queued %d | backend %s" % (client.queued(), state)
                log(line)
                last_status = frame["t"]
                frames = with_person = face_checks = faces_seen = fall_checks = 0
                face_ms_total = fall_ms_total = 0.0
    finally:
        log("stopping")
        if checker is not None:
            checker.stop()
        if fall_runner is not None:
            fall_runner.stop()
            fall_model.close()
        if backend_view is not None:
            backend_view.stop()
        if rec["recorder"] is not None:
            finish_recording()  # stopped mid-clip: keep what was recorded
        try:
            if client is not None:
                client.stop()  # sends what's queued while the backend is reachable
        finally:
            cam.close()        # always release /dev/video0 for the next program
    return 0


if __name__ == "__main__":
    sys.exit(main())

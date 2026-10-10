"""Stdlib-only integration stub; not the scenario simulator or clinical evidence."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def post(url, payload):
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def feature(angle, head, hip, aspect, velocity=0):
    return dict(torso_angle_deg=angle, head_y=head, hip_y=hip,
                bbox_aspect=aspect, vertical_velocity=velocity)


def trace(name):
    upright = feature(5, .2, .5, .4)
    floor = feature(90, .85, .85, 1.8)
    frames = [upright] * 5
    if name in ("gesture", "wave", "gesture_occlusion"):
        frames += [upright] * 20
        if name == "gesture_occlusion":
            frames[10:15] = [None] * 5
    elif name == "normal":
        frames += [feature(15, .35, .6, .65)] * 10  # sitting
        frames += [feature(75, .55, .5, 1.1)] * 10  # bending, not near floor
    elif name == "occlusion":
        frames += [floor] * 5 + [None] * 20 + [floor] * 5
    else:
        frames += [feature(45, .5, .6, .8, .3)] * 2 + [floor] * 21
    return frames + [upright] * 5


def arm_points(raised):
    return [dict(name="nose", x=.5, y=.3, confidence=.95),
            dict(name="left_shoulder", x=.4, y=.45, confidence=.95),
            dict(name="left_wrist", x=.4, y=.1 if raised else .65, confidence=.95),
            dict(name="right_shoulder", x=.6, y=.45, confidence=.95),
            dict(name="right_wrist", x=.6, y=.65, confidence=.95)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", choices=["fall", "normal", "occlusion", "gesture", "wave", "gesture_occlusion"], default="fall")
    parser.add_argument("--room", default="room-01")
    parser.add_argument("--fast", action="store_true", help="Send immediately with the same observation clock")
    parser.add_argument("--source-type", choices=["simulation", "replay"], default="simulation")
    args = parser.parse_args()
    frames, run_id = trace(args.scenario), str(uuid4())
    # Fast mode uses a historical observation clock, never future timestamps.
    start = datetime.now(timezone.utc)
    if args.fast:
        start -= timedelta(seconds=len(frames) * .2)
    origin = dict(source="fake-01", source_type=args.source_type, room_id=args.room,
                  scenario_id="stub-" + args.scenario, run_id=run_id)
    print("Sending {} as {} (run {})".format(args.scenario, args.source_type, run_id))
    for index, features in enumerate(frames):
        timestamp = (start + timedelta(seconds=index * .2)).isoformat()
        if index % 5 == 0:
            post(args.url + "/ingest/heartbeat", dict(origin, timestamp=timestamp, fps=5))
        observation = dict(origin, obs_id=run_id + "-" + str(index), timestamp=timestamp,
                           person_detected=features is not None, features=features,
                           face_present=None)
        if args.scenario in ("gesture", "wave", "gesture_occlusion"):
            end = 8 if args.scenario == "wave" else 20
            observation["keypoints"] = arm_points(5 <= index < end) if features else None
        result = post(args.url + "/ingest/observations", {"observations": [observation]})
        if result["accepted"] != 1:
            raise RuntimeError("Observation was rejected: {}".format(result))
        if not args.fast:
            time.sleep(.2)
    print("Complete. Inspect /alerts?run_id=" + run_id)
    print("Source will become offline after the configured silence timeout.")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as exc:
        raise SystemExit("HTTP {}: {}".format(exc.code, exc.read().decode("utf-8")))
    except URLError as exc:
        raise SystemExit("Cannot reach backend: {}".format(exc.reason))

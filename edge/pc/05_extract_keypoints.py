"""Step 6a: turn labelled public fall videos into a small file of keypoints + labels (no video).

    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\05_extract_keypoints.py --check      (match only, seconds)
    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\05_extract_keypoints.py --limit 5    (quick trial)
    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\05_extract_keypoints.py              (everything)

Inputs (see README "Step 6"): C:\\hearth-data\\labels\\{GMDCSA24,le2i}.csv from OmniFall, and the
videos in C:\\hearth-data\\gmdcsa24 and C:\\hearth-data\\le2i.

1. Reads OmniFall's timed labels and keeps only what we train on:
       walk, standing -> upright | sitting -> sitting | fallen, lying -> floor
   Transitions (fall, sit_down, lie_down, stand_up) and "other" are skipped: a frame halfway
   down is neither, and would only teach the network to be unsure.
2. Matches every label to its real video file. OmniFall writes "Subject_1/ADL/01" and
   "Coffee_room_01/video_14"; the files are "Subject 1\\ADL\\01.mp4" and
   "FallDataset\\Coffee_room_01\\Coffee_room_01\\Videos\\video (14).avi". Both sides are reduced to
   the same key (dataset, folder, number). A label without a video, or two videos with the same
   key, stops the script with a list: a label paired with the wrong video would quietly ruin
   training.
3. Reads each video once, front to back, and sends every --stride'th frame that falls inside a
   kept segment to YOLO11-pose on the GPU, in batches.
4. Keeps the most confident person per frame: 17 COCO keypoints (x, y in 0-1 of the frame like
   the Jetson sends, plus YOLO's confidence per keypoint), the label, the video, the person
   (subject) and the time. The subject lets training split by person, so the test never sees a
   person it trained on.
5. Writes C:\\hearth-data\\keypoints.npz (numbers only) and prints a summary per class.

YOLO11-pose (Ultralytics, AGPL-3.0) is only a tool here: its output becomes training data; the
classifier trained in the next step contains none of its code or weights.
"""

import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# OmniFall class ids (labels/LABELS.md) -> our three classes.
OMNIFALL_NAMES = {0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting", 5: "lie_down",
                  6: "lying", 7: "stand_up", 8: "standing", 9: "other"}
CLASSES = ["upright", "sitting", "floor"]
TO_OURS = {0: 0, 8: 0, 4: 1, 2: 2, 6: 2}       # everything else is skipped
DATASETS = ["GMDCSA24", "le2i"]                 # dataset index stored per frame
KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
VIDEO_EXT = {".mp4", ".avi"}


# --- step 2: label path <-> video file ------------------------------------------------------

def label_key(dataset, path):
    """OmniFall's path -> (dataset, folder..., number), e.g. ("le2i", "coffee_room_01", 14)."""
    parts = path.replace("\\", "/").split("/")
    number = int(re.findall(r"\d+", parts[-1])[-1])
    folders = tuple(p.lower().replace(" ", "_") for p in parts[:-1])
    return (dataset,) + folders + (number,)


def file_key(dataset, root, file):
    """A real video file -> the same kind of key as label_key, or None if it isn't one of ours."""
    parts = file.relative_to(root).parts
    number = re.findall(r"\d+", file.stem)
    if not number:
        return None
    if dataset == "GMDCSA24":
        # <top folder>\Subject 1\ADL\01.mp4 -> ("GMDCSA24", "subject_1", "adl", 1)
        subject = [p for p in parts if p.lower().startswith("subject")]
        if not subject or len(parts) < 2:
            return None
        folders = (subject[0], parts[-2])
    else:
        # FallDataset\Coffee_room_01\Coffee_room_01\Videos\video (14).avi -> room = the folder
        # right under FallDataset (the inner copy is spelled differently for Lecture_room)
        names = [p.lower() for p in parts]
        if "falldataset" not in names:
            return None
        folders = (parts[names.index("falldataset") + 1],)
    return (dataset,) + tuple(f.lower().replace(" ", "_") for f in folders) + (int(number[-1]),)


def index_videos(dataset, root):
    found, clashes = {}, defaultdict(list)
    for file in sorted(root.rglob("*")):
        if file.suffix.lower() in VIDEO_EXT:
            key = file_key(dataset, root, file)
            if key is None:
                continue
            if key in found:
                clashes[key].append(file)
            found.setdefault(key, file)
    for key, files in clashes.items():
        files.insert(0, found[key])
    return found, dict(clashes)


def load_segments(labels_dir, roots):
    """Kept segments grouped by video file, after matching every label to a file."""
    by_video, missing, skipped = defaultdict(list), [], Counter()
    for dataset in DATASETS:
        videos, clashes = index_videos(dataset, roots[dataset])
        if clashes:
            for key, files in clashes.items():
                print("  SAME KEY %s: %s" % (key, ", ".join(str(f) for f in files)))
            raise SystemExit("%s: several videos map to the same label name (listed above)" % dataset)
        rows = list(csv.DictReader(open(labels_dir / ("%s.csv" % dataset), newline="")))
        print("%-9s %4d labelled segments, %3d videos found under %s"
              % (dataset, len(rows), len(videos), roots[dataset]))
        for row in rows:
            key = label_key(dataset, row["path"])
            if key not in videos:
                missing.append("%s/%s" % (dataset, row["path"]))
                continue
            raw = int(row["label"])
            if raw not in TO_OURS:
                skipped[OMNIFALL_NAMES.get(raw, raw)] += 1
                continue
            by_video[videos[key]].append({
                "start": float(row["start"]), "end": float(row["end"]), "raw": raw,
                "label": TO_OURS[raw], "dataset": DATASETS.index(dataset),
                "subject": int(row["subject"]), "name": "%s/%s" % (dataset, row["path"])})
    if missing:
        for m in sorted(set(missing))[:30]:
            print("  NO VIDEO FOR LABEL %s" % m)
        raise SystemExit("%d labels have no matching video (first ones listed above)" % len(set(missing)))
    print("skipped segments (transitions / other): %s" % dict(skipped))
    return by_video


# --- steps 3-4: video -> keypoints ----------------------------------------------------------

def best_person(result, width, height):
    """Most confident person in one YOLO result -> (17 x 3 keypoints, box), or None."""
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None
    i = int(result.boxes.conf.argmax())
    xy = result.keypoints.xy[i].cpu().numpy()
    conf = result.keypoints.conf[i].cpu().numpy()
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 0], kp[:, 1], kp[:, 2] = xy[:, 0] / width, xy[:, 1] / height, conf
    kp[conf < 1e-6, :2] = 0.0          # YOLO reports unseen keypoints at (0, 0) anyway
    box = result.boxes.xyxy[i].cpu().numpy() / [width, height, width, height]
    return np.clip(kp, 0, 1), np.clip(box, 0, 1).astype(np.float32)


def extract(model, video, segments, stride, batch, device, out):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    last_end = max(s["end"] for s in segments)
    frames, metas, index, sent, people = [], [], 0, 0, 0

    def flush():
        nonlocal people
        if not frames:
            return
        for result, meta in zip(model(frames, device=device, verbose=False), metas):
            person = best_person(result, width, height)
            if person is None:
                continue
            people += 1
            out["keypoints"].append(person[0])
            out["box"].append(person[1])
            for k in ("label", "raw", "dataset", "subject", "video", "t"):
                out[k].append(meta[k])
        frames.clear()
        metas.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps
        if t > last_end:
            break
        if index % stride == 0:
            seg = next((s for s in segments if s["start"] <= t < s["end"]), None)
            if seg is not None:
                frames.append(frame)
                metas.append({"label": seg["label"], "raw": seg["raw"], "dataset": seg["dataset"],
                              "subject": seg["subject"], "video": len(out["videos"]), "t": t})
                sent += 1
                if len(frames) >= batch:
                    flush()
        index += 1
    flush()
    cap.release()
    out["videos"].append(segments[0]["name"])
    return sent, people


def main():
    ap = argparse.ArgumentParser(description="Public fall videos -> keypoints.npz for training.")
    ap.add_argument("--data", default=r"C:\hearth-data", type=Path, help="folder from the README")
    ap.add_argument("--gmdcsa", type=Path, help="GMDCSA24 folder (default <data>\\gmdcsa24)")
    ap.add_argument("--le2i", type=Path, help="Le2i folder (default <data>\\le2i)")
    ap.add_argument("--out", type=Path, help="output file (default <data>\\keypoints.npz)")
    ap.add_argument("--pose-model", default="yolo11m-pose.pt",
                    help="Ultralytics pose weights; a bare name is downloaded into --data")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth frame (default 2)")
    ap.add_argument("--batch", type=int, default=32, help="frames per GPU batch")
    ap.add_argument("--device", default="0", help="GPU index, or cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (a quick trial)")
    ap.add_argument("--check", action="store_true", help="only match labels to videos, no pose model")
    args = ap.parse_args()

    roots = {"GMDCSA24": args.gmdcsa or args.data / "gmdcsa24", "le2i": args.le2i or args.data / "le2i"}
    by_video = load_segments(args.data / "labels", roots)
    seconds = Counter()
    for segs in by_video.values():
        for s in segs:
            seconds[CLASSES[s["label"]]] += s["end"] - s["start"]
    print("matched: %d videos with kept segments; labelled seconds %s"
          % (len(by_video), {k: round(v) for k, v in seconds.items()}))
    if args.check:
        print("check only: every label has its video. Run again without --check to extract.")
        return

    from ultralytics import YOLO
    # A bare name would download into the current folder (the repo, where a .pt could get
    # committed by accident); keep it next to the data instead.
    weights = Path(args.pose_model)
    if weights.parent == Path("."):
        weights = args.data / weights
    model = YOLO(str(weights))
    videos = sorted(by_video)[: args.limit or None]
    out = defaultdict(list)
    start, sent_total = time.time(), 0
    for n, video in enumerate(videos, 1):
        sent, people = extract(model, video, by_video[video], args.stride, args.batch,
                               args.device, out)
        sent_total += sent
        print("[%3d/%d] %-34s %4d frames, person found in %3d%%  (%.0f s so far)"
              % (n, len(videos), by_video[video][0]["name"], sent, 100 * people // max(sent, 1),
                 time.time() - start))

    out_path = args.out or args.data / "keypoints.npz"
    np.savez_compressed(
        out_path,
        keypoints=np.array(out["keypoints"], dtype=np.float32).reshape(-1, 17, 3),
        box=np.array(out["box"], dtype=np.float32).reshape(-1, 4),
        label=np.array(out["label"], dtype=np.int64), raw_label=np.array(out["raw"], dtype=np.int64),
        dataset=np.array(out["dataset"], dtype=np.int64), subject=np.array(out["subject"], dtype=np.int64),
        video=np.array(out["video"], dtype=np.int64), t=np.array(out["t"], dtype=np.float32),
        videos=np.array(out["videos"]), classes=np.array(CLASSES), datasets=np.array(DATASETS),
        keypoint_names=np.array(KEYPOINT_NAMES), pose_model=np.array(args.pose_model))
    labels = Counter(CLASSES[i] for i in out["label"])
    print("\nwrote %s: %d frames with a person (of %d sent) in %.0f s"
          % (out_path, len(out["label"]), sent_total, time.time() - start))
    print("frames per class: %s" % dict(labels))
    for d, name in enumerate(DATASETS):
        subjects = sorted(set(s for s, ds in zip(out["subject"], out["dataset"]) if ds == d))
        print("  %-9s subjects %s" % (name, subjects))


if __name__ == "__main__":
    main()

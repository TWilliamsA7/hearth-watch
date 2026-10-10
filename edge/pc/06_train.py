"""Step 6b: train our own posture classifier (upright / sitting / floor) from keypoints.

    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\06_train.py

Input: C:\\hearth-data\\keypoints.npz from 05_extract_keypoints.py. Built in parts:
  part 1 (this file so far) - DATA: keypoints -> network inputs, split by person, Dataset and
                              DataLoader that hand out shuffled batches
  part 2                    - the network (nn.Module) and the training loop, with live progress
  part 3                    - evaluation on held-out people and on our own Jetson recordings
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# --- part 1: data ---------------------------------------------------------------------------

CLASSES = ["upright", "sitting", "floor"]
# YOLO gives every keypoint a confidence; the Jetson's poseNet only reports keypoints it found.
# A keypoint counts as "found" above this confidence, so training sees poses like poseNet's.
FOUND_CONF = 0.5
MIN_FOUND = 4            # frames with fewer found keypoints are dropped (too little to judge)
# Held-out people: never trained on, only used to check the network on people it hasn't seen.
# ~20% of each class (upright 165 s, sitting 132 s, floor 121 s of labelled video).
VAL_PEOPLE = {("GMDCSA24", 4), ("le2i", 3), ("le2i", 5)}
# Mirror image: left and right keypoints swap places (COCO order: nose, eyes, ears, shoulders,
# elbows, wrists, hips, knees, ankles - left before right).
MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def features(xy, found, aspect):
    """One frame -> the network's input: 17 x (dx, dy, found) = 51 numbers.

    xy: 17 x 2 in 0-1 of the frame (what the Jetson sends), found: 17 bools, aspect: width/height.
      1. x * aspect: back to true proportions (a 16:9 and a 4:3 frame no longer stretch the body
         differently); this keeps what matters most, which way the body points.
      2. subtract the centre of the found keypoints: where the person is in the room doesn't matter
      3. divide by the body's size (the larger side of its keypoint box): near or far doesn't matter
    Keypoints that weren't found are 0, with found = 0 telling the network they're missing.
    """
    p = xy.astype(np.float32) * np.array([aspect, 1.0], dtype=np.float32)   # stays float32
    pts = p[found]
    centre = pts.mean(axis=0)
    size = max((pts.max(axis=0) - pts.min(axis=0)).max(), 1e-3)
    rel = (p - centre) / size
    rel[~found] = 0.0
    out = np.concatenate([rel, found[:, None].astype(np.float32)], axis=1).reshape(-1)
    return out.astype(np.float32)        # networks use float32; float64 inputs would error


class PostureFrames(Dataset):
    """The frames of one split. Dataset is PyTorch's interface for "item number i":
    __len__ says how many there are, __getitem__ builds one (input, label) pair on demand.

    augment=True (training only) makes every pass over the data a little different:
      - hide each found keypoint with probability drop_p: poseNet on the Jetson misses joints
        that YOLO finds, so the network must cope with missing ones
      - mirror left/right half the time: a person lying head-left is the same posture as
        head-right
      - nudge keypoints by a tiny random amount: no two pose models agree to the pixel
    """

    def __init__(self, xy, found, aspect, labels, augment=False, drop_p=0.2, rng=None):
        self.xy, self.found, self.aspect, self.labels = xy, found, aspect, labels
        self.augment, self.drop_p = augment, drop_p
        self.rng = rng or np.random.default_rng()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        xy, found = self.xy[i].copy(), self.found[i].copy()
        if self.augment:
            drop = self.rng.random(17) < self.drop_p
            if (found & ~drop).sum() >= MIN_FOUND:      # never hide so much nothing is left
                found &= ~drop
            if self.rng.random() < 0.5:
                xy, found = xy[MIRROR], found[MIRROR]
                xy[:, 0] = 1.0 - xy[:, 0]
            xy = xy + self.rng.normal(0.0, 0.005, xy.shape)
        x = features(xy, found, self.aspect[i])
        return torch.from_numpy(x), int(self.labels[i])


def load_splits(path, drop_p):
    """keypoints.npz -> (training Dataset, validation Dataset, info for printing)."""
    d = np.load(path)
    found = d["keypoints"][:, :, 2] >= FOUND_CONF
    keep = found.sum(axis=1) >= MIN_FOUND
    size = d["video_size"][d["video"]]                     # each frame's video width, height
    aspect = (size[:, 0] / size[:, 1]).astype(np.float32)
    names = d["datasets"]
    person = [(str(names[ds]), int(s)) for ds, s in zip(d["dataset"], d["subject"])]
    is_val = np.array([p in VAL_PEOPLE for p in person])
    split = {}
    for name, mask, augment in (("train", keep & ~is_val, True), ("val", keep & is_val, False)):
        split[name] = PostureFrames(d["keypoints"][mask, :, :2], found[mask], aspect[mask],
                                    d["label"][mask], augment=augment, drop_p=drop_p)
    info = {"total": len(keep), "dropped": int((~keep).sum()),
            "sizes": sorted(set(tuple(int(v) for v in s) for s in d["video_size"]))}
    return split["train"], split["val"], info


def show_data(train, val, info, loader):
    """Part 1 check: what the network will be fed."""
    print("frames: %d, dropped (fewer than %d found keypoints): %d"
          % (info["total"], MIN_FOUND, info["dropped"]))
    print("video sizes (width x height): %s" % ", ".join("%dx%d" % s for s in info["sizes"]))
    for name, ds in (("train", train), ("val", val)):
        counts = np.bincount(ds.labels, minlength=3)
        print("  %-5s %6d frames  %s" % (name, len(ds), "  ".join(
            "%s %d (%d%%)" % (c, n, 100 * n // max(len(ds), 1)) for c, n in zip(CLASSES, counts))))
    x, y = next(iter(loader))
    print("one batch from the DataLoader: inputs %s %s, labels %s %s"
          % (tuple(x.shape), x.dtype, tuple(y.shape), y.dtype))
    print("first frame's input (17 keypoints x [dx, dy, found]):")
    print(np.array2string(x[0].view(17, 3).numpy(), precision=2, suppress_small=True))
    print("its label: %s; labels in this batch: %s"
          % (CLASSES[y[0]], dict(zip(CLASSES, np.bincount(y.numpy(), minlength=3).tolist()))))


def main():
    ap = argparse.ArgumentParser(description="Train the upright/sitting/floor classifier.")
    ap.add_argument("--data", default=r"C:\hearth-data\keypoints.npz", type=Path)
    ap.add_argument("--batch", type=int, default=256, help="frames per batch")
    ap.add_argument("--drop", type=float, default=0.2, help="chance of hiding each keypoint (training)")
    args = ap.parse_args()

    train, val, info = load_splits(args.data, args.drop)
    # DataLoader: asks the Dataset for `batch` random items and stacks them into one tensor.
    # shuffle=True: a new random order every epoch, so batches never repeat the same mix.
    # num_workers=0: build batches in this process (simplest; Windows-safe).
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)
    show_data(train, val, info, loader)
    print("\npart 1 (data) works. Part 2 adds the network and the training loop.")


if __name__ == "__main__":
    main()

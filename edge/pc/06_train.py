"""Step 6b: train our own posture classifier (upright / sitting / floor) from keypoints.

    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\06_train.py                (train, then field test)
    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\06_train.py --final-test   (the locked test, at the end)

Input: D:\\hearth-data\\keypoints.npz from 05_extract_keypoints.py. Built in parts:
  part 1 - DATA: keypoints -> network inputs; a LOCKED split by person into 60% train / 20%
           validation / 20% test (saved in split.json, never redrawn); Dataset and DataLoader
           that hand out shuffled batches (--show-data prints what the network is fed, then stops)
  part 2 - the network (nn.Module) and the training loop, with live progress: a line per batch,
           a summary per epoch, a chart window that redraws every epoch; saves the best model to
           D:\\hearth-data\\posture_mlp.pt (--out) and the curves next to it (..._curves.png)
  part 3 - the real test: the saved model on OUR Jetson recordings (recordings/*.jsonl, real
           poseNet output from our camera), per clip and overall. Runs after training, or alone
           with --test-only.
  --final-test - the locked 20%: scored once, when development is finished. Every run is
           logged in test_log.txt, so it's visible how often the test was looked at.
  "unknown" - the model may decline to answer: when its answer is "other" (a 4th class it
           learns: none of upright / sitting / floor), when the torso isn't visible (no shoulder
           or no hip, the edge's torso_visible rule), or when its top probability is under
           --min-conf.
           Reports then give two numbers: how often it decides, and how often it's right then.
           The cut-off is chosen on VALIDATION (a table of cut-offs is printed), never on test.
"""

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# --- part 1: data ---------------------------------------------------------------------------

ALL_CLASSES = ["upright", "sitting", "floor", "other"]
CLASSES = list(ALL_CLASSES)   # set by configure(): 3 classes (default) or 4 with --classes 4
RULES = {"torso": False}      # --torso-rule: also "unknown" when no shoulder or no hip
OTHER = 3                # "none of these": falling, sitting/lying down, getting up, kneeling,
                         # squatting, crawling, jumping, other activities (bending, picking up...)
# OmniFall's label (stored by 05 for every frame) -> our class. Mid-fall counts as "other": the
# model should say floor once the person is down, and the backend waits for 3 s of floor anyway.
FROM_OMNIFALL = dict((k, OTHER) for k in range(16))
FROM_OMNIFALL.update({0: 0, 8: 0, 4: 1, 2: 2, 6: 2})   # walk, standing | sitting | fallen, lying


def configure(classes, torso_rule):
    """3 classes: "other" frames are left out of training and the negative ("unknown") comes only
    from low confidence. 4 classes: "other" is a 4th answer the network learns, also reported as
    unknown. The list is changed in place so everything holding CLASSES sees the new one."""
    CLASSES[:] = ALL_CLASSES[:classes]
    RULES["torso"] = torso_rule
# YOLO gives every keypoint a confidence; the Jetson's poseNet only reports keypoints it found.
# A keypoint counts as "found" above this confidence, so training sees poses like poseNet's.
FOUND_CONF = 0.5
MIN_FOUND = 4            # frames with fewer found keypoints are dropped (too little to judge)
# The split, by PERSON (a person's frames are all in one part, so validation and test only ever
# show the network people it has never seen). Real datasets split by their subject ids;
# OF-Synthetic has none, so each of its clips counts as its own "person".
SPLIT_SEED = 6           # fixed: the same people land in the same part on every machine
VAL_SHARE = TEST_SHARE = 0.2
BIG_PERSON = 0.3         # anyone with >30% of a dataset's frames always trains (Le2i's subject 0
                         # groups many unnamed actors; in val/test it would swamp the score)
SYNTHETIC = {"of-syn"}   # scored separately: the number that matters is on real people
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


def person_of_each_frame(d):
    """'GMDCSA24/4', 'le2i/0', ... for real people; 'of-syn/<clip name>' for synthetic clips."""
    names, videos = d["datasets"], d["videos"]
    return np.array(["%s/%s" % (names[ds], videos[v].split("/")[-1] if names[ds] in SYNTHETIC else s)
                     for ds, s, v in zip(d["dataset"], d["subject"], d["video"])])


def locked_split(person, split_file):
    """{person: "train" | "val" | "test"}, drawn once and saved; later runs reuse it unchanged.

    Per dataset: shuffle its people with the fixed seed, give people to test until it holds 20%
    of that dataset's frames, then to validation until 20%, the rest train. People who appear
    later (e.g. more synthetic clips) in a dataset that is already split go to training, so the
    locked test never changes.
    """
    split = json.load(open(split_file)) if split_file.exists() else {}
    people, counts = np.unique(person, return_counts=True)
    frames = dict(zip(people, counts))
    by_dataset = {}
    for p in people:
        by_dataset.setdefault(p.split("/")[0], []).append(p)
    for dataset, members in sorted(by_dataset.items()):
        if any(m in split for m in members):
            for m in members:
                split.setdefault(m, "train")
            continue
        rng = np.random.default_rng(SPLIT_SEED + zlib.crc32(dataset.encode()))
        order = [members[i] for i in rng.permutation(len(members))]
        total = float(sum(frames[m] for m in members))
        filled = {"test": 0, "val": 0}
        for m in order:
            share = frames[m] / total
            part = "train"
            if share <= BIG_PERSON or len(members) > 20:
                if filled["test"] < TEST_SHARE * total:
                    part = "test"
                elif filled["val"] < VAL_SHARE * total:
                    part = "val"
            if part != "train":
                filled[part] += frames[m]
            split[m] = part
    json.dump(split, open(split_file, "w"), indent=1, sort_keys=True)
    return split


def load_splits(path, drop_p):
    """keypoints.npz -> {"train", "val", "val_syn", "test", "test_syn": Dataset} and info."""
    d = np.load(path)
    found = d["keypoints"][:, :, 2] >= FOUND_CONF
    keep = found.sum(axis=1) >= MIN_FOUND
    size = d["video_size"][d["video"]]                     # each frame's video width, height
    aspect = (size[:, 0] / size[:, 1]).astype(np.float32)
    person = person_of_each_frame(d)
    if "raw_label" in d.files:
        lookup = np.array([FROM_OMNIFALL[i] for i in range(16)])
        labels = lookup[d["raw_label"]]
    else:
        labels = d["label"]
    if len(CLASSES) == 3:
        keep &= labels != OTHER                            # 3 classes: "other" frames sit out
    split_file = Path(path).parent / "split.json"
    split = locked_split(person, split_file)
    part = np.array([split[p] for p in person])
    synthetic = np.array([p.split("/")[0] in SYNTHETIC for p in person])
    masks = {"train": part == "train",
             "val": (part == "val") & ~synthetic, "val_syn": (part == "val") & synthetic,
             "test": (part == "test") & ~synthetic, "test_syn": (part == "test") & synthetic}
    sets = {}
    for name, mask in masks.items():
        mask = mask & keep
        sets[name] = PostureFrames(d["keypoints"][mask, :, :2], found[mask], aspect[mask],
                                   labels[mask], augment=(name == "train"), drop_p=drop_p)
    people, present = {}, set(person)
    for p, prt in split.items():
        if p in present:
            dataset = p.split("/")[0]
            people.setdefault(prt, set()).add(dataset if dataset in SYNTHETIC else p)
    info = {"total": len(keep), "dropped": int((~keep).sum()), "split_file": split_file,
            "sizes": sorted(set(tuple(int(v) for v in s) for s in d["video_size"])),
            "people": dict((k, sorted(v)) for k, v in people.items())}
    return sets, info


def show_data(sets, info, loader):
    """Part 1 check: what the network will be fed."""
    print("frames: %d, dropped (fewer than %d found keypoints): %d"
          % (info["total"], MIN_FOUND, info["dropped"]))
    print("video sizes (width x height): %s" % ", ".join("%dx%d" % s for s in info["sizes"]))
    print("locked split (%s):" % info["split_file"])
    for prt in ("val", "test"):
        people = info["people"].get(prt, [])
        real = [p for p in people if p not in SYNTHETIC]
        print("  %-4s people: %s%s" % (prt, ", ".join(real),
                                      " + synthetic clips" if len(real) < len(people) else ""))
    for name, ds in sets.items():
        counts = np.bincount(ds.labels, minlength=len(CLASSES))
        print("  %-8s %6d frames  %s" % (name, len(ds), "  ".join(
            "%s %d (%d%%)" % (c, n, 100 * n // max(len(ds), 1)) for c, n in zip(CLASSES, counts))))
    x, y = next(iter(loader))
    print("one batch from the DataLoader: inputs %s %s, labels %s %s"
          % (tuple(x.shape), x.dtype, tuple(y.shape), y.dtype))
    print("first frame's input (17 keypoints x [dx, dy, found]):")
    print(np.array2string(x[0].view(17, 3).numpy(), precision=2, suppress_small=True))
    print("its label: %s; labels in this batch: %s"
          % (CLASSES[y[0]], dict(zip(CLASSES, np.bincount(y.numpy(), minlength=len(CLASSES)).tolist()))))


# --- part 2: the network and the training loop ---------------------------------------------

class PostureNet(nn.Module):
    """51 numbers in -> 4 scores out (one per class; the highest wins).

    nn.Module is PyTorch's base class for anything with learnable weights. __init__ creates the
    layers; forward() says how data flows through them. nn.Linear(a, b) is a b x a weight matrix
    plus b biases: every output is a weighted sum of every input. ReLU (max(0, x)) between them
    lets the network learn bends, not just straight lines. Dropout switches off a random 20% of
    the 64 values during training only, so no single one can be relied on (less memorising).
    51*64+64 + 64*64+64 + 64*4+4 = 7,748 weights: tiny on purpose, it has to run on the Jetson.
    """

    def __init__(self, inputs=51, hidden=64, classes=len(CLASSES), dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, classes))

    def forward(self, x):
        return self.layers(x)          # raw scores ("logits"); softmax would turn them into %


def class_weights(labels):
    """Rarer classes count more in the loss: weight = (frames per class on average) / (this class's
    frames). Unweighted, always answering the most common class would already score well."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    return torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)))


def run_epoch(model, loader, loss_fn, device, optimizer=None, progress=None):
    """One pass over a DataLoader. With an optimizer it trains; without, it only measures.
    Returns (mean loss, accuracy, per-class accuracy, confusion matrix)."""
    training = optimizer is not None
    model.train(training)              # switches Dropout on (train) or off (measure)
    total_loss, confusion = 0.0, np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    with torch.set_grad_enabled(training):     # no gradient bookkeeping when only measuring
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)  # the batch moves to the GPU
            scores = model(x)                  # 1. forward: 256 x 4 scores
            loss = loss_fn(scores, y)          # 2. loss: how wrong, as one number
            if training:
                optimizer.zero_grad()          # 3. forget the previous batch's gradients
                loss.backward()                # 4. backward: d(loss)/d(weight) for all 7,748
                optimizer.step()               # 5. nudge every weight downhill
            total_loss += loss.item() * len(y)
            predicted = scores.argmax(dim=1)
            np.add.at(confusion, (y.cpu().numpy(), predicted.cpu().numpy()), 1)
            if progress:
                progress(i, len(loader), loss.item())
    n = confusion.sum()
    per_class = confusion.diagonal() / np.maximum(confusion.sum(axis=1), 1)
    return total_loss / max(n, 1), confusion.diagonal().sum() / max(n, 1), per_class, confusion


class LiveChart(object):
    """A window with loss and accuracy curves, redrawn after every epoch."""

    def __init__(self, enabled):
        self.plt = None
        if not enabled:
            return
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()                              # interactive: draw without blocking the script
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
        self.fig.canvas.manager.set_window_title("Hearth posture training")

    def update(self, history):
        if self.plt is None:
            return
        epochs = range(1, len(history["train_loss"]) + 1)
        for ax, key, title in ((self.ax_loss, "loss", "loss (lower is better)"),
                               (self.ax_acc, "acc", "accuracy (higher is better)")):
            ax.clear()
            ax.plot(epochs, history["train_" + key], "o-", label="train (augmented)")
            ax.plot(epochs, history["val_" + key], "o-", label="validation (unseen people)")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend()
        self.ax_acc.set_ylim(0, 1)
        self.fig.tight_layout()
        self.plt.pause(0.05)                   # lets the window redraw

    def save(self, path):
        if self.plt is not None:
            self.fig.savefig(path, dpi=120)


def progress_line(epoch, epochs):
    def show(i, n, loss):
        bar = "#" * (20 * i // n)
        print("\r  epoch %2d/%d [%-20s] batch %3d/%d  loss %.3f" % (epoch, epochs, bar, i, n, loss),
              end="", flush=True)
    return show


def by_class(values, names=CLASSES):
    """[0.98, 0.88, ...] -> 'upright 98% sitting 88% ...'."""
    return " ".join("%s %.0f%%" % (n, 100 * v) for n, v in zip(names, values))


def confusion_table(confusion, columns=CLASSES):
    lines = ["              predicted: " + "".join("%9s" % c for c in columns)]
    for c, row in zip(CLASSES, confusion):
        lines.append("  actually %-8s     " % c + "".join("%9d" % v for v in row))
    return "\n".join(lines)


# --- "unknown": deciding not to answer --------------------------------------------------------

UNSURE_BELOW = 0.7       # default cut-off: top probability under this -> "unknown"
SHOULDERS, HIPS = [5, 6], [11, 12]     # COCO indices


DECIDED = ["upright", "sitting", "floor", "unknown"]    # what a decision can be


def decide(scores, x, min_conf):
    """Network scores + its inputs -> a class per frame, or -1 = unknown.

    Unknown when:
      - it isn't confident: softmax turns the raw scores into probabilities that add up to 1
        (e.g. 0.92 / 0.05 / 0.03); the top one is under min_conf
      - (4 classes) the network's answer is "other" (none of upright / sitting / floor)
      - (--torso-rule) no shoulder or no hip is visible. Off by default: on our recordings it
        threw away right answers, e.g. lying clips where poseNet found head and shoulders only.
    """
    probability = torch.softmax(scores, dim=1)
    confidence, predicted = probability.max(dim=1)
    found = x.view(-1, 17, 3)[:, :, 2] > 0.5           # the "found" flags inside the input
    unknown = confidence < min_conf
    if len(CLASSES) > OTHER:
        unknown |= predicted == OTHER
    if RULES["torso"]:
        unknown |= ~(found[:, SHOULDERS].any(dim=1) & found[:, HIPS].any(dim=1))
    predicted[unknown] = -1
    return predicted


def collect(model, loader, device):
    """Run a DataLoader through the model once: (scores, inputs, labels) for every frame."""
    model.eval()
    scores, inputs, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            scores.append(model(x.to(device)).cpu())
            inputs.append(x)
            labels.append(y)
    return torch.cat(scores), torch.cat(inputs), torch.cat(labels)


def decided_report(predicted, labels):
    """Decisions vs truth. Rows: the true class (upright, sitting, floor, other); columns:
    the decision (upright, sitting, floor, unknown). Returns a dict:
      coverage  - on upright/sitting/floor frames, how often it answered
      accuracy  - how often that answer was right
      per_class - right-when-answering for upright, sitting, floor
      floor_caught - of ALL floor frames, how many were called floor (missed falls are the
                     worst error, so this is watched on its own)
      other_unknown - of the "other" frames, how many came out unknown (the correct outcome)
    """
    predicted, labels = predicted.numpy(), labels.numpy()
    confusion = np.zeros((len(CLASSES), len(DECIDED)), dtype=np.int64)
    np.add.at(confusion, (labels, np.where(predicted < 0, 3, predicted)), 1)
    posture = confusion[:3]
    decided = posture[:, :3]
    return {"coverage": decided.sum() / max(posture.sum(), 1),
            "accuracy": decided.diagonal().sum() / max(decided.sum(), 1),
            "per_class": decided.diagonal() / np.maximum(decided.sum(axis=1), 1),
            "floor_caught": confusion[2, 2] / max(confusion[2].sum(), 1),
            "other_unknown": (confusion[3, 3] / confusion[3].sum()
                              if len(CLASSES) > OTHER and confusion[3].sum() else None),
            "confusion": confusion}


def describe(r):
    text = ("answers %.0f%% of upright/sitting/floor frames, right on %.0f%% of those (%s); "
            "floor caught %.0f%%" % (100 * r["coverage"], 100 * r["accuracy"],
                                     by_class(r["per_class"]), 100 * r["floor_caught"]))
    if r["other_unknown"] is not None:
        text += "; 'other' frames -> unknown %.0f%%" % (100 * r["other_unknown"])
    return text


def cutoff_table(scores, inputs, labels):
    """How the cut-off trades answering often against being right (on validation)."""
    print("choosing the cut-off on validation (real people):")
    print("   cut-off  answers  right when answering  (upright/sitting/floor)  floor caught  other->unknown")
    for cut in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        r = decided_report(decide(scores, inputs, cut), labels)
        other = "%5.0f%%" % (100 * r["other_unknown"]) if r["other_unknown"] is not None else "   -"
        print("   %5.2f   %5.1f%%        %5.1f%%           (%3.0f%% / %3.0f%% / %3.0f%%)       %5.1f%%        %s"
              % (cut, 100 * r["coverage"], 100 * r["accuracy"], *(100 * r["per_class"]),
                 100 * r["floor_caught"], other))
    print("   (0.00 = torso rule and 'other' only; answers / right count upright, sitting and floor frames)")


# --- part 3: the real test, on our own Jetson recordings -------------------------------------

KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
JETSON_ASPECT = 640 / 480      # the Jetson camera's frame (edge config); recordings store 0-1 only
# Clip name -> what the person was doing. lie_down clips start with the person still getting
# down, so their first LIE_SKIP_S seconds are left out rather than wrongly called "floor".
CLIP_LABEL = {"stand": 0, "walk": 0, "sit": 1, "lie_down": 2}
LIE_SKIP_S = 5.0


def load_recording(path):
    """One recordings/*.jsonl clip -> (xy 17x2 per frame, found 17 per frame, seconds from start).
    poseNet lists only the keypoints it found (plus "neck", which COCO-17 doesn't have)."""
    xys, founds, times, t0 = [], [], [], None
    for line in open(path):
        obs = json.loads(line)
        if not obs.get("person_detected") or not obs.get("keypoints"):
            continue
        h, rest = obs["timestamp"][11:-1].split(":", 1)
        t = int(h) * 3600 + int(rest.split(":")[0]) * 60 + float(rest.split(":")[1])
        t0 = t if t0 is None else t0
        xy, found = np.zeros((17, 2), np.float32), np.zeros(17, bool)
        for kp in obs["keypoints"]:
            if kp["name"] in KEYPOINT_NAMES:
                i = KEYPOINT_NAMES.index(kp["name"])
                xy[i], found[i] = (kp["x"], kp["y"]), True
        if found.sum() >= MIN_FOUND:
            xys.append(xy), founds.append(found), times.append(t - t0)
    return np.array(xys), np.array(founds), np.array(times)


def test_on_recordings(model, folder, device, min_conf=UNSURE_BELOW):
    """Per clip: how often the model decided, how often it was right then, and a timeline (one
    letter per frame: U upright / S sitting / F floor / ? unknown, . = getting down, not scored).
    Then the overall table."""
    clips = sorted(Path(folder).glob("*.jsonl"))
    if not clips:
        print("no recordings in %s (they come with git pull)" % folder)
        return
    model.eval()
    all_predicted, all_truth = [], []
    print("\nOUR JETSON RECORDINGS (poseNet, our camera) - the model has never seen these:")
    print("  (cut-off %.2f; decided = frames it answered, right = of those)" % min_conf)
    for clip in clips:
        kind = clip.name.split("-")[0]
        if kind not in CLIP_LABEL:
            continue
        xy, found, t = load_recording(clip)
        if len(t) == 0:
            continue
        x = torch.from_numpy(np.stack([features(a, f, JETSON_ASPECT) for a, f in zip(xy, found)]))
        with torch.no_grad():
            predicted = decide(model(x.to(device)).cpu(), x, min_conf).numpy()
        truth = CLIP_LABEL[kind]
        scored = t >= LIE_SKIP_S if kind == "lie_down" else np.ones(len(t), bool)
        p = predicted[scored]
        all_predicted.append(p)
        all_truth.append(np.full(len(p), truth))
        decided = p >= 0
        right = (p[decided] == truth).mean() if decided.any() else 0.0
        timeline = "".join(("USF?"[q] if s else ".") for q, s in zip(predicted, scored))
        print("  %-32s %-8s %4d frames  decided %3.0f%%  right %3.0f%%  %s"
              % (clip.name, CLASSES[truth], scored.sum(), 100 * decided.mean() if len(p) else 0,
                 100 * right, timeline[::3]))
    r = decided_report(torch.from_numpy(np.concatenate(all_predicted)),
                       torch.from_numpy(np.concatenate(all_truth)))
    print("overall (%d frames): %s" % (r["confusion"].sum(), describe(r)))
    print(confusion_table(r["confusion"][:3], DECIDED))
    print("(timelines show every 3rd frame, ~0.3 s per letter)")


def final_test(model, out, measure, device, min_conf):
    """The locked 20%: score the saved model on test people, log it, and say how often it ran."""
    saved = torch.load(out, map_location=device)
    model.load_state_dict(saved["state_dict"])
    log = out.parent / "test_log.txt"
    before = len(open(log).read().splitlines()) if log.exists() else 0
    print("LOCKED TEST - run %d. Use it to report the final number, not to tune: every look at"
          " it makes it a little less honest." % (before + 1))
    loss_fn = nn.CrossEntropyLoss()
    line = [time.strftime("%Y-%m-%d %H:%M"), "model epoch %d" % saved["epoch"]]
    for name, title in (("test", "real people"), ("test_syn", "synthetic clips")):
        if name not in measure:
            continue
        _, acc, per_class, _ = run_epoch(model, measure[name], loss_fn, device)
        print("\n%s, network alone (%d classes): %.0f%% right, balanced %.0f%% (%s)"
              % (title, len(CLASSES), 100 * acc, 100 * per_class.mean(), by_class(per_class)))
        scores, inputs, labels = collect(model, measure[name], device)
        r = decided_report(decide(scores, inputs, min_conf), labels)
        print("%s, as the system answers (cut-off %.2f): %s" % (title, min_conf, describe(r)))
        print(confusion_table(r["confusion"], DECIDED))
        line.append("%s %.1f%% balanced %.1f%% | @%.2f answers %.1f%% right %.1f%% floor caught %.1f%%"
                    % (name, 100 * acc, 100 * per_class.mean(), min_conf, 100 * r["coverage"],
                       100 * r["accuracy"], 100 * r["floor_caught"]))
    with open(log, "a") as fh:
        fh.write(" | ".join(line) + "\n")
    print("\nlogged in %s" % log)


def main():
    ap = argparse.ArgumentParser(description="Train the upright/sitting/floor classifier.")
    ap.add_argument("--data", default=r"D:\hearth-data\keypoints.npz", type=Path)
    ap.add_argument("--out", type=Path, help="model file (default: next to --data, posture_mlp.pt)")
    ap.add_argument("--batch", type=int, default=256, help="frames per batch")
    ap.add_argument("--epochs", type=int, default=30, help="passes over the training data")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate (step size)")
    ap.add_argument("--drop", type=float, default=0.2, help="chance of hiding each keypoint (training)")
    ap.add_argument("--show-data", action="store_true", help="part 1 only: print a batch, then stop")
    ap.add_argument("--test-only", action="store_true", help="part 3 only: field-test the saved model")
    ap.add_argument("--final-test", action="store_true",
                    help="score the saved model on the LOCKED test people (logged; do this last)")
    ap.add_argument("--classes", type=int, choices=(3, 4), default=3,
                    help="3: upright/sitting/floor, unknown from low confidence (default); "
                         "4: also a learned 'other' (negative) class")
    ap.add_argument("--torso-rule", action="store_true",
                    help="also unknown when no shoulder or no hip is visible (off by default)")
    ap.add_argument("--min-conf", type=float, default=None,
                    help="cut-off for 'unknown' (default: the saved model's, else %.2f)" % UNSURE_BELOW)
    ap.add_argument("--recordings", type=Path, default=Path("recordings"),
                    help="our Jetson clips (default: the repo's recordings folder)")
    ap.add_argument("--no-chart", action="store_true", help="no live chart window")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)                       # same starting weights every run: comparable runs
    out = args.out or args.data.parent / "posture_mlp.pt"
    saved = None
    if args.test_only or args.final_test:      # a saved model brings its own settings
        saved = torch.load(out, map_location="cpu")
        configure(len(saved["classes"]), saved.get("torso_rule", "torso" in saved.get("unknown_rule", "")))
    else:
        configure(args.classes, args.torso_rule)
    print("%d classes (%s); unknown = low confidence%s%s"
          % (len(CLASSES), ", ".join(CLASSES), " or 'other'" if len(CLASSES) == 4 else "",
             " or no torso" if RULES["torso"] else ""))

    sets, info = load_splits(args.data, args.drop)
    train, val = sets["train"], sets["val"]
    # DataLoader: asks the Dataset for `batch` random items and stacks them into one tensor.
    # shuffle=True: a new random order every epoch, so batches never repeat the same mix.
    # num_workers=0: build batches in this process (simplest; Windows-safe).
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)
    measure = dict((name, DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0))
                   for name, ds in sets.items() if name != "train" and len(ds))
    val_loader = measure["val"]
    if args.show_data:
        show_data(sets, info, loader)
        return

    device = torch.device(args.device)
    model = PostureNet(classes=len(CLASSES)).to(device)    # weights move to the GPU
    if saved is not None:
        model.load_state_dict(saved["state_dict"])
        min_conf = args.min_conf if args.min_conf is not None else saved.get("min_conf", UNSURE_BELOW)
        if args.final_test:
            final_test(model, out, measure, device, min_conf)
            return
        cutoff_table(*collect(model, val_loader, device))
        test_on_recordings(model, args.recordings, device, min_conf)
        return
    min_conf = args.min_conf if args.min_conf is not None else UNSURE_BELOW
    weights = class_weights(train.labels)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    # AdamW: a gradient-descent variant that adapts the step size per weight; weight_decay
    # gently pulls weights toward 0 (another guard against memorising).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print("device %s | %d weights | train %d frames, val %d | class weights %s"
          % (device, sum(p.numel() for p in model.parameters()), len(train), len(val),
             dict(zip(CLASSES, [round(float(w), 2) for w in weights]))))

    chart = LiveChart(not args.no_chart)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best = (-1.0, 0)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, loader, loss_fn, device, optimizer,
                                          progress_line(epoch, args.epochs))
        va_loss, va_acc, va_class, confusion = run_epoch(model, val_loader, loss_fn, device)
        for k, v in (("train_loss", tr_loss), ("train_acc", tr_acc), ("val_loss", va_loss),
                     ("val_acc", va_acc)):
            history[k].append(v)
        # "Balanced" accuracy (mean of the three per-class accuracies) picks the best epoch, so
        # being good at the common class can't hide being bad at a rare one.
        balanced = float(va_class.mean())
        mark = ""
        if balanced > best[0]:
            best = (balanced, epoch)
            torch.save({"state_dict": model.state_dict(), "classes": list(CLASSES),
                        "torso_rule": RULES["torso"],
                        "found_conf": FOUND_CONF, "min_found": MIN_FOUND, "epoch": epoch,
                        "val_balanced_acc": balanced, "split_file": str(info["split_file"]),
                        "min_conf": min_conf,
                        "unknown_rule": "top probability < min_conf"
                                        + (", or class 'other'" if len(CLASSES) == 4 else "")
                                        + (", or no shoulder or no hip" if RULES["torso"] else ""),
                        "features": "17 COCO keypoints x (dx, dy, found); see features()"}, out)
            mark = "  <- best so far, saved"
        print("\r  epoch %2d/%d  train loss %.3f acc %.0f%% | val loss %.3f acc %.0f%% (%s)  %.1f s%s"
              % (epoch, args.epochs, tr_loss, 100 * tr_acc, va_loss, 100 * va_acc,
                 by_class(va_class), time.time() - start, mark))
        chart.update(history)

    curves = out.with_name(out.stem + "_curves.png")     # posture_mlp.pt -> posture_mlp_curves.png
    chart.save(curves)
    model.load_state_dict(torch.load(out, map_location=device)["state_dict"])
    _, va_acc, va_class, confusion = run_epoch(model, val_loader, loss_fn, device)
    print("\nbest epoch %d: validation %.0f%% (balanced %.0f%%), saved to %s"
          % (best[1], 100 * va_acc, 100 * best[0], out))
    print("on the unseen (validation) people, frame counts:\n" + confusion_table(confusion))
    if "val_syn" in measure:
        _, syn_acc, syn_class, _ = run_epoch(model, measure["val_syn"], loss_fn, device)
        print("validation synthetic clips: %.0f%% (%s)" % (100 * syn_acc, by_class(syn_class)))
    scores, inputs, labels = collect(model, val_loader, device)
    cutoff_table(scores, inputs, labels)
    r = decided_report(decide(scores, inputs, min_conf), labels)
    print("validation as the system answers (cut-off %.2f): %s\n%s"
          % (min_conf, describe(r), confusion_table(r["confusion"], DECIDED)))
    print("(the locked test set was not touched; --final-test scores it once you're done)")
    test_on_recordings(model, args.recordings, device, min_conf)
    if chart.plt is not None:
        print("curves saved to %s. Close the chart window to finish." % curves)
        chart.plt.ioff()
        chart.plt.show()


if __name__ == "__main__":
    main()

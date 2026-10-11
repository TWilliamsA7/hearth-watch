"""Step 6d: teach the 3-class model to say "unsure" about close-ups - by fine-tuning, gently.

    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\07_finetune_closeups.py

Problem it fixes: with only head and shoulders in view (someone close to the camera, legs and
hips out of frame), the 3-class model must pick upright, sitting or floor, and on our clip
stand-...053107 it confidently said FLOOR: a false alarm. Yet head-and-shoulders alone is fine
when the person is LYING (lie_down-...061342: hips hidden, still clearly on the floor).

How:
1. Close-up examples are made from the training frames themselves (no new data, and our Jetson
   recordings stay out of training, so they remain a fair field test). Three kinds, as the
   camera sees people who are too close or half out of frame:
     upper body (hips, knees, ankles hidden; sometimes the arms too)
         upright / sitting -> "other"   floor -> still "floor" (lying with legs hidden is lying)
     head only (just the face keypoints)
         everyone -> "other" (a face alone can't tell standing from lying)
     legs only (no head, shoulders or arms)
         upright / sitting -> "other"   floor -> still "floor" (horizontal legs are lying, e.g.
         a fall that ends with the upper body behind a bed)
   So the network has to learn from the ORIENTATION of what is left.
2. Fine-tune, don't retrain: start from model_3class.pt, add a 4th output for "other" (the three
   old outputs keep their weights; the new one starts almost never firing), train a few epochs
   at a very low learning rate (1e-4, a tenth of normal) so it shifts instead of forgetting.
3. Saying "unsure" about a real posture is punished hard: on every frame that really is
   upright, sitting or floor, the loss gets an extra --unsure-penalty x (probability it gave
   to "other"), multiplied by (1 + --floor-penalty) on frames that really are FLOOR (genuine
   floor frames can look like close-ups; calling them "other" would miss a fall). Only the
   crops make "other" worth saying.
4. Keep it only if it's safe: each epoch is checked on the validation people (never trained on)
   and saved only if floor caught on REAL frames stays at or above the starting model's (minus
   --floor-tolerance points, default 0).

Run 2 (--floor-penalty 5, upper-body crops only): kept epoch 6, floor caught 93.2%, close-ups ->
unknown 85%, our lying clips 97% floor; but head-only (stand-...053107) stayed "floor" and
legs-only (walk-...034202) became "floor", so head-only and legs-only crops were added.
First run (2026-10-10, penalty 3, no floor extra): close-ups -> unknown 4% -> 95% in 6 epochs,
lying close-ups still floor 87%, real accuracy 94.4 -> 94.8%, but floor caught 93.1 -> 92.2%, so
nothing was kept under the zero-tolerance rule.

Reports per epoch: real validation frames (answers, right when answering, floor caught) and the
validation crops (upright/sitting close-ups -> unknown, lying close-ups -> floor). At the end,
the field test on our recordings. Output: D:\\hearth-data\\model_closeup.pt, which 06_train.py
--test-only / --final-test can read like any other model.
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 06_train.py's file name starts with a digit, so it's loaded by path instead of "import".
_spec = importlib.util.spec_from_file_location("train06", Path(__file__).with_name("06_train.py"))
t6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t6)

HEAD = [0, 1, 2, 3, 4]             # nose, eyes, ears (COCO indices)
SHOULDERS = [5, 6]
ARMS = [7, 8, 9, 10]               # elbows, wrists
LOWER = [11, 12, 13, 14, 15, 16]   # hips, knees, ankles
CROP_KINDS = ["upper body", "head only", "legs only"]
CROP_SHARES = [0.6, 0.2, 0.2]
UPRIGHT, SITTING, FLOOR, OTHER = 0, 1, 2, 3


def close_up(xy, found, label, rng):
    """One frame as a close-up of a random kind (see the top of the file). Returns
    (found, label), or None if too little would be left to judge."""
    found = found.copy()
    kind = rng.choice(len(CROP_KINDS), p=CROP_SHARES)
    if kind == 0:                                    # upper body
        found[LOWER] = False
        if rng.random() < 0.5:
            found[ARMS] = False
        label = FLOOR if label == FLOOR else OTHER
    elif kind == 1:                                  # head only
        found[SHOULDERS + ARMS + LOWER] = False
        label = OTHER
    else:                                            # legs only
        found[HEAD + SHOULDERS + ARMS] = False
        label = FLOOR if label == FLOOR else OTHER
    if found.sum() < t6.MIN_FOUND:
        return None
    return found, label


class WithCloseups(Dataset):
    """Training frames, a share of them turned into close-ups on the fly (then augmented as usual)."""

    def __init__(self, base, share, rng=None):
        self.base, self.share = base, share
        self.rng = rng or np.random.default_rng()
        self.labels = base.labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        b = self.base
        xy, found, label = b.xy[i].copy(), b.found[i].copy(), int(b.labels[i])
        if self.rng.random() < self.share:
            crop = close_up(xy, found, label, self.rng)
            if crop is not None:
                found, label = crop
        xy, found = t6.augment(xy, found, self.rng, b.drop_p)
        return torch.from_numpy(t6.features(xy, found, b.aspect[i])), label


class Crops(Dataset):
    """Fixed close-ups of the validation frames (same every run), keeping only those whose
    close-up label is `want`: OTHER (should come out unknown) or FLOOR (lying, should stay floor)."""

    def __init__(self, base, want):
        rng = np.random.default_rng(1)
        self.items = []
        for i in range(len(base)):
            crop = close_up(base.xy[i], base.found[i], int(base.labels[i]), rng)
            if crop is not None and crop[1] == want:
                self.items.append(torch.from_numpy(t6.features(base.xy[i], crop[0], base.aspect[i])))
        self.label = want

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.label


def grow_to_four(model3):
    """The 3-class network with a 4th output added. Old outputs keep their weights exactly; the
    new "other" output starts with zero weights and a bias of -4, so at first it almost never
    wins (softmax share ~2%) and the model behaves exactly like before."""
    model4 = t6.PostureNet(classes=4)
    last = "layers.%d." % (len(model4.layers) - 1)              # the output layer's weights
    hidden = dict((k, v) for k, v in model3.state_dict().items() if not k.startswith(last))
    model4.load_state_dict(hidden, strict=False)                 # copy all but the output layer
    old, new = model3.layers[-1], model4.layers[-1]
    with torch.no_grad():
        new.weight.zero_()
        new.bias.fill_(-4.0)
        new.weight[:3] = old.weight
        new.bias[:3] = old.bias
    return model4


def measure(model, loaders, device, min_conf):
    """Real validation frames + crops -> the numbers that decide whether to keep an epoch."""
    scores, inputs, labels = t6.collect(model, loaders["real"], device)
    r = t6.decided_report(t6.decide(scores, inputs, min_conf), labels)
    out = {"answers": r["coverage"], "right": r["accuracy"], "floor": r["floor_caught"],
           "per_class": r["per_class"]}
    for name in ("up_crops", "floor_crops"):
        scores, inputs, labels = t6.collect(model, loaders[name], device)
        decided = t6.decide(scores, inputs, min_conf).numpy()
        out[name] = float((decided < 0).mean() if name == "up_crops" else (decided == FLOOR).mean())
    return out


def line(m):
    return ("real: answers %.1f%% right %.1f%% floor caught %.1f%% | close-ups: should-be-unsure -> "
            "unknown %.0f%%, lying -> floor %.0f%%"
            % (100 * m["answers"], 100 * m["right"], 100 * m["floor"], 100 * m["up_crops"],
               100 * m["floor_crops"]))


def main():
    ap = argparse.ArgumentParser(description="Fine-tune the 3-class model with close-up negatives.")
    ap.add_argument("--data", default=r"D:\hearth-data\keypoints.npz", type=Path)
    ap.add_argument("--start", default=r"D:\hearth-data\model_3class.pt", type=Path)
    ap.add_argument("--out", default=r"D:\hearth-data\model_closeup.pt", type=Path)
    ap.add_argument("--epochs", type=int, default=10, help="slow learning rate: needs a few more passes")
    ap.add_argument("--lr", type=float, default=1e-4, help="very low on purpose (normal: 1e-3)")
    ap.add_argument("--share", type=float, default=0.2, help="share of training frames made close-ups")
    ap.add_argument("--unsure-penalty", type=float, default=3.0,
                    help="extra loss per unit of probability put on 'other' for a real posture")
    ap.add_argument("--floor-penalty", type=float, default=0.0,
                    help="extra multiplier on the unsure penalty for frames that are really floor")
    ap.add_argument("--floor-tolerance", type=float, default=0.0,
                    help="points of floor caught an epoch may lose and still be kept (default 0)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="cut-off for unknown")
    ap.add_argument("--recordings", type=Path, default=Path("recordings"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)

    # Data exactly as the 3-class model saw it (same locked split, OmniFall's "other" left out),
    # but with 4 outputs.
    t6.configure(4, False, other_frames=False)
    sets, info = t6.load_splits(args.data, drop_p=0.2)
    train = WithCloseups(sets["train"], args.share)
    loader = DataLoader(train, batch_size=256, shuffle=True, num_workers=0)
    loaders = {"real": DataLoader(sets["val"], batch_size=1024),
               "up_crops": DataLoader(Crops(sets["val"], OTHER), batch_size=1024),
               "floor_crops": DataLoader(Crops(sets["val"], FLOOR), batch_size=1024)}
    print("validation: %d real frames, %d should-be-unsure close-ups, %d lying close-ups"
          % (len(sets["val"]), len(loaders["up_crops"].dataset), len(loaders["floor_crops"].dataset)))

    saved = torch.load(args.start, map_location="cpu")
    if len(saved["classes"]) != 3:
        raise SystemExit("--start must be a 3-class model (06_train.py --classes 3)")
    model3 = t6.PostureNet(classes=3)
    model3.load_state_dict(saved["state_dict"])
    model = grow_to_four(model3).to(device)

    start = measure(model, loaders, device, args.min_conf)
    print("starting point (the 3-class model, nothing learned yet):\n  " + line(start))
    floor_bar = start["floor"]

    # Class weights for the four answers, from the training labels plus the expected crops.
    counts = np.bincount(sets["train"].labels, minlength=4).astype(np.float32)
    counts[OTHER] = args.share * (counts[UPRIGHT] + counts[SITTING])
    weights = torch.tensor(counts.sum() / (4 * counts)).to(device)
    ce = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = ce(scores, y)
            posture = y != OTHER
            if posture.any():           # punish "unsure" on frames that really are a posture
                p_other = torch.softmax(scores[posture], dim=1)[:, OTHER]
                extra = 1.0 + args.floor_penalty * (y[posture] == FLOOR).float()
                loss = loss + args.unsure_penalty * (p_other * extra).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 20 == 0 or i == len(loader):
                print("\r  epoch %d/%d batch %d/%d loss %.3f" % (epoch, args.epochs, i, len(loader),
                                                                loss.item()), end="", flush=True)
        m = measure(model, loaders, device, args.min_conf)
        safe = m["floor"] >= floor_bar - args.floor_tolerance / 100.0 - 1e-9
        mark = ""
        if safe and (best is None or m["up_crops"] > best[0]):
            best = (m["up_crops"], epoch)
            torch.save({"state_dict": model.state_dict(), "classes": list(t6.ALL_CLASSES),
                        "torso_rule": False, "other_frames": False, "min_conf": args.min_conf,
                        "found_conf": t6.FOUND_CONF, "min_found": t6.MIN_FOUND, "epoch": epoch,
                        "fine_tuned_from": str(args.start), "unsure_penalty": args.unsure_penalty,
                        "unknown_rule": "top probability < min_conf, or class 'other' (close-ups)",
                        "features": "17 COCO keypoints x (dx, dy, found); see 06 features()"}, args.out)
            mark = "  <- best so far, saved"
        print("\r  epoch %d/%d  %s%s%s  %.0f s" % (epoch, args.epochs, line(m),
                                                 "" if safe else "  (floor caught dropped: not kept)",
                                                 mark, time.time() - t0))

    if best is None:
        print("\nNo epoch kept floor caught at %.1f%% or better (tolerance %.1f points): keep "
              "model_3class.pt. Try --floor-penalty (e.g. 5), or allow --floor-tolerance 1."
              % (100 * floor_bar, args.floor_tolerance))
        return
    print("\nkept epoch %d -> %s" % (best[1], args.out))
    model.load_state_dict(torch.load(args.out, map_location=device)["state_dict"])
    t6.test_on_recordings(model, args.recordings, device, args.min_conf)


if __name__ == "__main__":
    main()

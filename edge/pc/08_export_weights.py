"""Step 7a: export the chosen posture model for the Jetson (numpy there, no PyTorch).

    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\08_export_weights.py

The Jetson (Python 3.6, numpy 1.13) can't run PyTorch, but it doesn't need to: the network is
three matrix multiplications. This saves the 7,683 learned numbers and the settings the Jetson
must copy exactly into edge\\hearth_edge\\posture_mlp.npz (~30 KB, committed with the code):

  W1 64x51, b1 64 | W2 64x64, b2 64 | W3 3x64, b3 3     the three Linear layers
  classes, min_conf (0.50, chosen on validation), min_found, keypoint_names, feature recipe
  source model's SHA-256, epoch, and the locked test result, so the file says what it is

Then it proves the copy is faithful: the same inputs go through PyTorch and through a plain
numpy version of the network, on validation frames and on our Jetson recordings, and the scores,
the chosen class and the final decision (with "unknown") are compared frame by frame. Finally the
written file is loaded through the Jetson's own module (edge\hearth_edge\posture_mlp.py) and
every frame of our recordings goes through it from raw keypoints to label: its decisions must
match the PyTorch pipeline's, which proves the Jetson builds the inputs exactly like training.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

_spec = importlib.util.spec_from_file_location("train06", Path(__file__).with_name("06_train.py"))
t6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t6)

REPO = Path(__file__).resolve().parents[2]
FEATURES = ("v1: 17 COCO keypoints x (dx, dy, found) = 51 numbers. x times frame width/height; "
            "subtract the mean of the found keypoints; divide by the larger side of their box "
            "(at least 1e-3); missing keypoints are 0 with found 0. Order: dx, dy, found per "
            "keypoint, keypoints in keypoint_names order.")
LOCKED_TEST = ("locked test run 1, 2026-10-10: real people right 97.7%, floor caught 96.1%, "
               "precision 97.9%, false positive rate 0.67% (edge/pc/RESULTS.md)")


def numpy_scores(x, w):
    """What the Jetson will compute: Linear -> ReLU -> Linear -> ReLU -> Linear (dropout is
    only active while training, so it isn't here)."""
    h = np.maximum(x @ w["W1"].T + w["b1"], 0.0)
    h = np.maximum(h @ w["W2"].T + w["b2"], 0.0)
    return h @ w["W3"].T + w["b3"]


def numpy_decide(scores, min_conf):
    e = np.exp(scores - scores.max(axis=1, keepdims=True))      # softmax, overflow-safe
    p = e / e.sum(axis=1, keepdims=True)
    best = p.argmax(axis=1)
    best[p.max(axis=1) < min_conf] = -1
    return best


def compare(name, x, model, w, min_conf):
    with torch.no_grad():
        torch_scores = model(torch.from_numpy(x)).numpy()
    np_scores = numpy_scores(x, w)
    diff = float(np.abs(torch_scores - np_scores).max())
    same_class = (torch_scores.argmax(1) == np_scores.argmax(1)).mean()
    torch_decision = t6.decide(torch.from_numpy(torch_scores), torch.from_numpy(x), min_conf).numpy()
    same_decision = (torch_decision == numpy_decide(np_scores, min_conf)).mean()
    ok = diff < 1e-4 and same_class == 1.0 and same_decision == 1.0
    print("  %-26s %6d frames | largest score difference %.1e | same class %.2f%% | "
          "same decision %.2f%%  %s" % (name, len(x), diff, 100 * same_class, 100 * same_decision,
                                       "OK" if ok else "MISMATCH"))
    return ok


def check_edge_module(out, model, min_conf, recordings):
    """Recordings -> the Jetson module (raw keypoints in pixels, as run_edge.py has them) vs the
    training pipeline (06 features + PyTorch + decide). Every decision must match."""
    spec = importlib.util.spec_from_file_location("posture_mlp", REPO / "edge" / "hearth_edge" / "posture_mlp.py")
    edge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edge)
    mlp = edge.PostureMLP(str(out))
    width, height = 640, 480                                 # the Jetson camera's frame
    names = t6.KEYPOINT_NAMES
    frames = agree = 0
    for clip in sorted(recordings.glob("*.jsonl")):
        for line in open(clip):
            obs = json.loads(line)
            if not obs.get("person_detected") or not obs.get("keypoints"):
                continue
            pixels = dict((k["name"], (k["x"] * width, k["y"] * height)) for k in obs["keypoints"])
            got = mlp.predict(pixels, width, height)
            xy, found = np.zeros((17, 2), np.float32), np.zeros(17, bool)
            for k in obs["keypoints"]:
                if k["name"] in names:
                    xy[names.index(k["name"])], found[names.index(k["name"])] = (k["x"], k["y"]), True
            if found.sum() < t6.MIN_FOUND:
                want = None
            else:
                x = torch.from_numpy(t6.features(xy, found, width / float(height)))[None]
                with torch.no_grad():
                    d = int(t6.decide(model(x), x, min_conf)[0])
                want = "unknown" if d < 0 else t6.CLASSES[d]
            frames += 1
            agree += (got is None and want is None) or (got is not None and got["label"] == want)
    ok = frames > 0 and agree == frames
    print("  %-26s %6d frames | Jetson module agrees on %d (%.2f%%)  %s"
          % ("edge module, end to end", frames, agree, 100.0 * agree / max(frames, 1),
             "OK" if ok else "MISMATCH"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Export the posture model's weights for the Jetson.")
    ap.add_argument("--model", default=r"D:\hearth-data\model_3class.pt", type=Path)
    ap.add_argument("--data", default=r"D:\hearth-data\keypoints.npz", type=Path,
                    help="for the validation-frame check")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="cut-off chosen on validation (default 0.50)")
    ap.add_argument("--recordings", type=Path, default=REPO / "recordings")
    ap.add_argument("--out", type=Path, default=REPO / "edge" / "hearth_edge" / "posture_mlp.npz")
    args = ap.parse_args()

    saved = torch.load(args.model, map_location="cpu")
    classes = list(saved["classes"])
    t6.configure(len(classes), saved.get("torso_rule", False), saved.get("other_frames"))
    if t6.RULES["torso"]:
        raise SystemExit("this model uses the torso rule; the Jetson export doesn't support it")
    model = t6.PostureNet(classes=len(classes))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    linear = [m for m in model.layers if isinstance(m, nn.Linear)]
    w = {}
    for i, layer in enumerate(linear, 1):
        w["W%d" % i] = layer.weight.detach().numpy().astype(np.float32)
        w["b%d" % i] = layer.bias.detach().numpy().astype(np.float32)
    count = sum(v.size for v in w.values())
    print("model %s: %s, epoch %d, %d weights, cut-off %.2f"
          % (args.model.name, "/".join(classes), saved.get("epoch", -1), count, args.min_conf))

    print("checking numpy against PyTorch:")
    ok = True
    if args.data.exists():
        sets, _ = t6.load_splits(args.data, 0.2)
        _, inputs, _ = t6.collect(model, torch.utils.data.DataLoader(sets["val"], batch_size=1024), "cpu")
        ok &= compare("validation frames", inputs.numpy(), model, w, args.min_conf)
    else:
        print("  (no %s: validation check skipped)" % args.data)
    rec = []
    for clip in sorted(args.recordings.glob("*.jsonl")):
        xy, found, _ = t6.load_recording(clip)
        rec += [t6.features(a, f, t6.JETSON_ASPECT) for a, f in zip(xy, found)]
    if rec:
        ok &= compare("our Jetson recordings", np.stack(rec), model, w, args.min_conf)
    if not ok:
        raise SystemExit("numpy and PyTorch disagree: not exported")

    model_sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, classes=np.array(classes), min_conf=np.float32(args.min_conf),
             min_found=np.int64(t6.MIN_FOUND), keypoint_names=np.array(t6.KEYPOINT_NAMES),
             features=np.array(FEATURES), source_model=np.array(args.model.name),
             source_sha256=np.array(model_sha), epoch=np.int64(saved.get("epoch", -1)),
             locked_test=np.array(LOCKED_TEST),
             info=np.array(json.dumps({"layers": [list(w["W%d" % i].shape) for i in (1, 2, 3)],
                                       "activation": "relu", "weights": int(count)})),
             **w)
    print("wrote %s (%.1f KB), sha256 %s"
          % (args.out, args.out.stat().st_size / 1024.0,
             hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]))
    if any(args.recordings.glob("*.jsonl")) and not check_edge_module(args.out, model, args.min_conf,
                                                                       args.recordings):
        args.out.unlink()
        raise SystemExit("the Jetson module disagrees with training: file removed, not exported")


if __name__ == "__main__":
    main()

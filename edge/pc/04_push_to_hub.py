"""Step 4: publish the ONNX exports to your Hugging Face account, with an honest model card.

    .\\.venv-pc\\Scripts\\python.exe -m pip install huggingface_hub     (once)
    .\\.venv-pc\\Scripts\\hf.exe auth login        (once: paste a token with "write" access from
                                                 https://huggingface.co/settings/tokens)
    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\04_push_to_hub.py [--repo NAME] [--private]

What you learn: a Hugging Face model repo is a git repo of files; every upload is a commit
with an id, and pinning a download to that id (as Person 2's model.json does for the original)
makes it reproducible. The model card (README.md in the repo) records provenance and license.

Uploads models/human-fall-detection/best_640.onnx, best_416.onnx, export_info.json and a
generated README.md to <your-username>/<repo> (default hearth-fall-yolo11n-onnx), then prints
the commit id and saves it to models/human-fall-detection/hub_info.json for the Jetson.
Public by default so the Jetson can download without a token (AGPL-3.0 allows redistribution
with attribution and the same license); --private keeps it private.
"""

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

EXPORTS = Path("models/human-fall-detection")


def model_card(info, repo_id):
    src = info["source"]
    rows = "\n".join("| `%s` | %d x %d | `%s` -> `%s` | %.1e |" % (
        e["file"], e["imgsz"], e["imgsz"], e["input"][1], e["output"][1], e["max_abs_diff_vs_pytorch"])
        for e in info["exports"])
    classes = ", ".join("%s = %s" % (k, v) for k, v in sorted(info["classes"].items(), key=lambda kv: int(kv[0])))
    return """---
license: agpl-3.0
base_model: {base}
library_name: onnx
tags: [onnx, yolo11, object-detection, fall-detection, tensorrt, jetson]
---

# Fall detection (YOLO11n) as ONNX for TensorRT 8.2 / Jetson Nano

ONNX exports of [{base}](https://huggingface.co/{base}) at revision `{rev}`
(file `{file}`, SHA-256 `{sha}`), made for the Hearth project (Knight Hacks IX) to run on a
Jetson Nano 2GB (JetPack 4.6, TensorRT 8.2). No retraining: same weights, different format.

## Files

| File | Input size | Input -> output | Max diff vs PyTorch |
|---|---|---|---|
{rows}

- Export: opset {opset}, fixed input size, no NMS inside the graph (do it after inference).
- Input: `images`, float32, 1x3xHxW, RGB, 0..1, letterboxed (aspect kept, grey 114 padding).
- Output: `output0`, 1 x (4 + 3) x N: per candidate position, box centre x, centre y, width,
  height (pixels of the input), then class scores. Classes: {classes}.
- Checked by running the same photo through PyTorch and ONNX Runtime: decisions above a 0.25
  score agree at every position; raw values differ only by float rounding (column above).

## Intended use and limits

Flags a *possible* fall for human review in a single-room monitoring demo. Not a medical
device, not clinically validated, and a person lying down on purpose looks the same as a fall.
Accuracy on the target camera has to be measured; see the base model's card for its training data.

## License

AGPL-3.0, as declared by the base model's author. This export keeps that license; credit to
the base model's author. Uploaded as `{repo}`.
""".format(base=src["repo_id"], rev=src["revision"], file=src["filename"], sha=src["sha256"],
           rows=rows, opset=info["opset"], classes=classes, repo=repo_id)


def main():
    ap = argparse.ArgumentParser(description="Publish the ONNX exports to the Hugging Face Hub.")
    ap.add_argument("--repo", default="hearth-fall-yolo11n-onnx", help="repo name under your account")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    info_path = EXPORTS / "export_info.json"
    if not info_path.exists():
        raise SystemExit("Run step 3 first (edge/pc/03_export_onnx.py): %s is missing" % info_path)
    info = json.loads(info_path.read_text())
    api = HfApi()
    user = api.whoami()["name"]                       # fails clearly if `hf auth login` wasn't done
    repo_id = "%s/%s" % (user, args.repo)
    print("1. Repo:", repo_id, "(private)" if args.private else "(public)")
    api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)

    print("2. Uploading as one commit:")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for e in info["exports"]:
            (tmp / e["file"]).write_bytes((EXPORTS / e["file"]).read_bytes())
            print("   ", e["file"])
        (tmp / "export_info.json").write_text(json.dumps(info, indent=2))
        (tmp / "README.md").write_text(model_card(info, repo_id), encoding="utf-8")
        print("    export_info.json, README.md (model card)")
        commit = api.upload_folder(repo_id=repo_id, folder_path=str(tmp), repo_type="model",
                                   commit_message="ONNX exports (opset %d) for TensorRT 8.2" % info["opset"])
    revision = commit.oid
    hub = {"repo_id": repo_id, "revision": revision,
           "files": {e["file"]: e["sha256"] for e in info["exports"]}}
    (EXPORTS / "hub_info.json").write_text(json.dumps(hub, indent=2))
    print("3. Done. Commit (pin downloads to this):", revision)
    print("   Page:", "https://huggingface.co/%s" % repo_id)
    print("   Example pinned download URL:")
    print("   https://huggingface.co/%s/resolve/%s/%s" % (repo_id, revision, info["exports"][-1]["file"]))
    print("   Saved %s - paste the commit id back for step 5." % (EXPORTS / "hub_info.json"))


if __name__ == "__main__":
    main()

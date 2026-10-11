#!/usr/bin/env bash
# Downloads the Hugging Face fall model (ONNX) pinned to a commit and builds its TensorRT engine.
# Run ON THE JETSON (needs internet for the download; the build itself is offline):
#     bash ~/hearth-watch/edge/tools/get_fall_model.sh            # 416x416 (default, faster)
#     SIZE=640 bash ~/hearth-watch/edge/tools/get_fall_model.sh   # the 640x640 export
#
# Source: Lunar255/hearth-fall-yolo11n-onnx @ 12dc1ac (made by edge/pc/03_export_onnx.py +
# 04_push_to_hub.py from melihuzunoglu/human-fall-detection @ 97261b0; AGPL-3.0). The SHA-256 of
# the ONNX file is checked against the export_info.json published in the same commit.
# Files go to <repo>/models/fall/ (on the Jetson ~/hearth-watch/models/fall/), outside edge/, so
# syncing edge/ never deletes them. The engine build takes ~9-10 min on the Nano (once).
set -euo pipefail
[ -f /etc/nv_tegra_release ] || { echo "run this on the Jetson"; exit 1; }
REPO_ID="${REPO_ID:-Lunar255/hearth-fall-yolo11n-onnx}"
REVISION="${REVISION:-12dc1ac7a68fbb879a6a0a97d6a5904cfed0abb8}"
SIZE="${SIZE:-416}"
here="$(cd "$(dirname "$0")" && pwd)"
dest="$(cd "$here/../.." && pwd)/models/fall"
mkdir -p "$dest"
cd "$dest"
onnx="best_${SIZE}.onnx"
engine="best_${SIZE}_fp16.engine"
base="https://huggingface.co/$REPO_ID/resolve/$REVISION"

python3 - "$base" "$onnx" <<'PY'
import hashlib, json, os, sys, urllib.request
base, name = sys.argv[1], sys.argv[2]
def get(fname):
    tmp = fname + ".part"
    urllib.request.urlretrieve(base + "/" + fname, tmp)
    os.replace(tmp, fname)
get("export_info.json")
info = json.load(open("export_info.json"))
want = {e["file"]: e["sha256"] for e in info["exports"]}
if name not in want:
    sys.exit("%s is not in this commit (have: %s)" % (name, ", ".join(want)))
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
if not (os.path.exists(name) and sha(name) == want[name]):
    get(name)
if sha(name) != want[name]:
    os.remove(name)
    sys.exit("checksum mismatch for %s" % name)
print("ok %s (sha256 %s..., classes %s)" % (name, want[name][:12], info["classes"]))
PY

if [ -f "$engine" ] && [ "$engine" -nt "$onnx" ]; then
  echo "ok $engine (already built)"
  exit 0
fi
while pgrep -x trtexec >/dev/null; do echo "waiting for another TensorRT build to finish..."; sleep 20; done
echo "building $engine with TensorRT (FP16), ~10 min on the Nano..."
/usr/src/tensorrt/bin/trtexec --onnx="$onnx" --saveEngine="$engine" --fp16 --workspace=512 \
  > "trtexec_${SIZE}.log" 2>&1 || { tail -20 "trtexec_${SIZE}.log"; exit 1; }
grep -E "GPU Compute Time:|Throughput:" "trtexec_${SIZE}.log" | sed 's/^/  /'
echo "fall model ready: $dest/$engine"

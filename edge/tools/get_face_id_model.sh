#!/usr/bin/env bash
# Downloads SFace (ONNX) pinned to a commit and builds its TensorRT engine, for --face-id.
# Run ON THE JETSON (needs internet for the download; the build itself is offline):
#     bash ~/hearth-watch/edge/tools/get_face_id_model.sh
#
# Source: opencv/face_recognition_sface @ 3d70824 (Apache-2.0), the same file and SHA-256 the
# laptop enrolls with (backend/hearth/perception/face_demo.py), so templates from both match.
# Input 1x3x112x112 raw 0-255 RGB (the model normalizes itself), output 1x128.
# Files go to <repo>/models/face-id/ (on the Jetson ~/hearth-watch/models/face-id/), outside
# edge/, so syncing edge/ never deletes them.
set -euo pipefail
[ -f /etc/nv_tegra_release ] || { echo "run this on the Jetson"; exit 1; }
REVISION="${REVISION:-3d7082438a6e4551e840c9b2bb60b71e8da4b524}"
SHA256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
here="$(cd "$(dirname "$0")" && pwd)"
dest="$(cd "$here/../.." && pwd)/models/face-id"
mkdir -p "$dest"
cd "$dest"
onnx="face_recognition_sface_2021dec.onnx"
engine="sface_fp16.engine"
url="https://huggingface.co/opencv/face_recognition_sface/resolve/$REVISION/$onnx"

python3 - "$url" "$onnx" "$SHA256" <<'PY'
import hashlib, os, sys, urllib.request
url, name, want = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
if not (os.path.exists(name) and sha(name) == want):
    urllib.request.urlretrieve(url, name + ".part")
    os.replace(name + ".part", name)
if sha(name) != want:
    os.remove(name)
    sys.exit("checksum mismatch for %s" % name)
print("ok %s (sha256 %s...)" % (name, want[:12]))
PY

if [ -f "$engine" ] && [ "$engine" -nt "$onnx" ]; then
  echo "ok $engine (already built)"
  exit 0
fi
while pgrep -x trtexec >/dev/null; do echo "waiting for another TensorRT build to finish..."; sleep 20; done
echo "building $engine with TensorRT (FP16)..."
/usr/src/tensorrt/bin/trtexec --onnx="$onnx" --saveEngine="$engine" --fp16 --workspace=256 \
  > trtexec_sface.log 2>&1 || { tail -20 trtexec_sface.log; exit 1; }
grep -E "GPU Compute Time:|Throughput:" trtexec_sface.log | sed 's/^/  /'
echo "face template model ready: $dest/$engine"

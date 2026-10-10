#!/usr/bin/env bash
# Downloads OpenCV's res10 SSD face-detection model (~10 MB) for edge/hearth_edge/face.py.
# Run ON THE JETSON (it refuses elsewhere so the binary never lands in the git repo):
#     bash ~/hearth-watch/edge/tools/get_face_model.sh
# Files go to <repo>/models/res10/ (on the Jetson: ~/hearth-watch/models/res10/), outside edge/,
# so syncing edge/ never deletes them. Both files come from OpenCV's own GitHub repositories.
set -euo pipefail
[ -f /etc/nv_tegra_release ] || { echo "run this on the Jetson, not the laptop"; exit 1; }
here="$(cd "$(dirname "$0")" && pwd)"
dest="$(cd "$here/../.." && pwd)/models/res10"
mkdir -p "$dest"
cd "$dest"
fetch() {  # name url sha256 (pinned from the first verified download, 2026-10-10)
  if [ -f "$1" ] && echo "$3  $1" | sha256sum --check --status; then echo "ok (cached) $1"; return; fi
  python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$2" "$1.part"
  echo "$3  $1.part" | sha256sum --check --status || { echo "checksum mismatch for $1"; rm -f "$1.part"; exit 1; }
  mv "$1.part" "$1"
  echo "downloaded $1"
}
fetch deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/4.1.1/samples/dnn/face_detector/deploy.prototxt \
  1e5a51172abe1e4052c27a3488f99d86dfec9b962191bc999a98261990be5d64
fetch res10_300x300_ssd_iter_140000.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel \
  2a56a11a57a4a295956b0660b4a3d76bbdca2206c4961cea8efe7d95c7cb2f2d
echo "face model ready in $dest"

#!/usr/bin/env bash
# One-time setup for the Jetson Nano 2GB (JetPack 4.6.x) used by Hearth (edge/).
# Run ON the Jetson in an interactive SSH session (it asks for your sudo password):
#   bash setup_jetson.sh
# Safe to re-run; each step skips itself if already done.
set -euo pipefail

step() { printf '\n==> %s\n' "$*"; }

step "Max power mode + clocks"
sudo nvpmodel -m 0
sudo jetson_clocks

step "4 GB swapfile (2 GB RAM is not enough for TensorRT engine builds)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

step "Boot headless (frees ~400 MB). Undo with: sudo systemctl set-default graphical.target"
sudo systemctl set-default multi-user.target

step "System packages"
sudo apt-get update
sudo apt-get install -y git cmake build-essential python3-dev python3-pip python3-numpy \
  libpython3-dev v4l-utils curl htop nano

step "jtop monitor (edge code itself is stdlib-only, no pip packages needed)"
sudo -H python3 -m pip install -U jetson-stats

step "jetson-inference (poseNet, detectNet, TensorRT) — build takes ~45-60 min"
if ! python3 -c 'import jetson_inference' 2>/dev/null && ! python3 -c 'import jetson.inference' 2>/dev/null; then
  cd ~
  [ -d jetson-inference ] || git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
  cd jetson-inference
  git submodule update --init
  mkdir -p build && cd build
  # Skip the interactive model/PyTorch dialogs; models download on first use.
  cmake -DBUILD_INTERACTIVE=NO ../
  # jetson-inference's Python bindings link -lnpymath without telling the linker where numpy
  # keeps it ("cannot find -lnpymath"). Point the linker at numpy's lib dir for py3 and py2.
  NPY_LIBS=""
  for py in python3 python2; do
    d=$("$py" -c 'import numpy, os; print(os.path.join(os.path.dirname(numpy.__file__), "core", "lib"))' 2>/dev/null || true)
    [ -n "$d" ] && [ -e "$d/libnpymath.a" ] && NPY_LIBS="${NPY_LIBS:+$NPY_LIBS:}$d"
  done
  echo "numpy npymath dirs: ${NPY_LIBS:-none found}"
  LIBRARY_PATH="${NPY_LIBS}${LIBRARY_PATH:+:$LIBRARY_PATH}" make -j"$(nproc)"
  sudo make install
  sudo ldconfig
fi

step "Pre-build the two models we use (first run builds the TensorRT engine; slow once)"
cd ~
python3 - <<'PY'
try:
    from jetson_inference import poseNet, detectNet
except ImportError:
    from jetson.inference import poseNet, detectNet
poseNet("resnet18-body")
detectNet("facenet")
print("models ready")
PY

step "Face detection model (OpenCV res10, ~10 MB) into ~/hearth-watch/models/res10"
bash "$(dirname "$0")/get_face_model.sh"

echo
echo "Done. Reboot now: sudo reboot   then re-run check_jetson.sh"

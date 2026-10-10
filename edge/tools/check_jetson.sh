#!/usr/bin/env bash
# Read-only health check for the Jetson Nano 2GB used by Hearth (edge/).
# Run on the Jetson:  bash check_jetson.sh
ok()   { printf '  [ OK ] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }
info() { printf '  [info] %s\n' "$*"; }

echo "== System =="
if [ -f /etc/nv_tegra_release ]; then info "$(head -1 /etc/nv_tegra_release)"; else bad "not an L4T system"; fi
grep -q 'R32' /etc/nv_tegra_release 2>/dev/null && ok "JetPack 4.x (R32) as expected" || bad "expected L4T R32 (JetPack 4.6.x)"
info "Ubuntu: $(lsb_release -ds 2>/dev/null)"
info "Model: $(tr -d '\0' </proc/device-tree/model 2>/dev/null)"
info "Python3: $(python3 --version 2>&1)"

echo "== Memory / storage =="
free -h | sed 's/^/  /'
swap_mb=$(free -m | awk '/Swap/{print $2}')
[ "${swap_mb:-0}" -ge 3500 ] && ok "swap ${swap_mb} MB" || bad "swap only ${swap_mb} MB (want >= 4 GB)"
avail=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "${avail:-0}" -ge 8 ] && ok "${avail} GB free on /" || bad "only ${avail} GB free on / (want >= 8 GB)"
[ "$(systemctl get-default)" = multi-user.target ] && ok "boots headless" || info "boots to desktop ($(systemctl get-default)); headless frees ~400 MB"
info "power mode: $(sudo -n nvpmodel -q 2>/dev/null | tail -1 || echo 'needs sudo to read')"

echo "== CUDA / TensorRT =="
[ -x /usr/local/cuda/bin/nvcc ] && ok "$(/usr/local/cuda/bin/nvcc --version | tail -1)" || bad "CUDA toolkit (nvcc) missing"
dpkg -l 2>/dev/null | grep -q '^ii  tensorrt ' && ok "TensorRT $(dpkg -l tensorrt | awk '/^ii/{print $3}')" || bad "TensorRT package missing"
python3 -c 'import tensorrt as t; print(t.__version__)' >/dev/null 2>&1 && ok "python tensorrt imports" || bad "python3 cannot import tensorrt"
python3 -c 'import cv2; print(cv2.__version__)' >/dev/null 2>&1 && ok "OpenCV $(python3 -c 'import cv2; print(cv2.__version__)')" || bad "python3 cannot import cv2"
python3 -c 'import numpy' >/dev/null 2>&1 && ok "numpy" || bad "numpy missing"

echo "== jetson-inference =="
if python3 -c 'import jetson_inference, jetson_utils' >/dev/null 2>&1 || python3 -c 'import jetson.inference, jetson.utils' >/dev/null 2>&1; then
  ok "jetson-inference python bindings installed"
else
  bad "jetson-inference not installed (build from github.com/dusty-nv/jetson-inference)"
fi
for m in resnet18-body facenet; do
  ls /usr/local/bin/networks 2>/dev/null | grep -qi "$m" || ls ~/jetson-inference/data/networks 2>/dev/null | grep -qi "${m%%-*}" \
    && ok "model present: $m" || bad "model missing: $m"
done

echo "== Cameras =="
if ls /dev/video* >/dev/null 2>&1; then
  for d in /dev/video*; do
    name=$(cat /sys/class/video4linux/$(basename "$d")/name 2>/dev/null)
    mj=$(v4l2-ctl -d "$d" --list-formats 2>/dev/null | grep -c MJPG)
    info "$d: $name  (MJPEG: $([ "$mj" -gt 0 ] && echo yes || echo no))"
  done
  n=$(ls /sys/class/video4linux | wc -l); [ "$n" -ge 1 ] && ok "$n camera(s); Hearth MVP needs one" || bad "no camera"
else
  bad "no /dev/video* devices; plug in the webcams"
fi
command -v v4l2-ctl >/dev/null && ok "v4l2-ctl installed" || bad "v4l-utils missing (sudo apt install v4l-utils)"


echo "== Network =="
ip -br addr | sed 's/^/  /'
python3 -c "import urllib.request; urllib.request.urlopen('https://github.com', timeout=8)" >/dev/null 2>&1 \
  && ok "internet reachable (HTTPS)" || bad "no internet over HTTPS (needed for apt/pip/model downloads); note UCF blocks ping"

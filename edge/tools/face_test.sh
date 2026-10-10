#!/usr/bin/env bash
# One-command face detection test with live video. Run on the LAPTOP:
#     edge/tools/face_test.sh                       # the 4 standard situations
#     edge/tools/face_test.sh --seconds 10 close 3m # pick your own labels
#     edge/tools/face_test.sh --no-video            # tables only, no window
#
# For each situation it tells you where to stand, waits for Enter, opens the video window
# (skeleton, yellow head close-up area, red faces found) and runs edge/tools/face_probe.py on the
# Jetson. Each run prints a table: how often res10 finds the face in the whole frame vs the head
# close-up, at cutoffs 0.3 / 0.4 / 0.5. All tables are also saved to
# recordings/face_test-<time>.md (git ignores recordings/). Nothing else is saved.
# Type s + Enter to skip a situation, q + Enter to stop. Ctrl+C stops the current run.
#
# JETSON is the ssh target (default: the ssh-config alias "jetson").
set -u
JETSON="${JETSON:-jetson}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
STREAM_URL="rtp://192.168.55.100:1234"

seconds=15
video=1
labels=()
while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) seconds="$2"; shift ;;
    --no-video) video=0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)"; exit 2 ;;
    *) labels+=("$1") ;;
  esac
  shift
done
[ ${#labels[@]} -gt 0 ] || labels=(close 3m mouth-covered 3m-mouth-covered)

instructions() {
  case "$1" in
    close) echo "sit or stand about 1 m from the camera, facing it" ;;
    3m) echo "stand about 3 m back, facing the camera" ;;
    mouth-covered) echo "about 1 m away, facing the camera, cover your mouth (hand or mask)" ;;
    3m-mouth-covered) echo "about 3 m back, facing the camera, cover your mouth (hand or mask)" ;;
    *) echo "get into position for: $1" ;;
  esac
}

player_pid=""
stop_player() { [ -n "$player_pid" ] && kill "$player_pid" 2>/dev/null; player_pid=""; }
trap 'stop_player' EXIT INT TERM

holder=$(ssh -o ConnectTimeout=5 "$JETSON" 'fuser /dev/video0 2>/dev/null | xargs -r ps -o pid=,args= -p') \
  || { echo "could not reach the Jetson as '$JETSON' (USB cable plugged in?)"; exit 1; }
if [ -n "$holder" ]; then
  echo "the Jetson camera is in use by:"; echo "  $holder"; echo "stop that program first"; exit 1
fi
echo "copying edge/ to the Jetson..."
rsync -a --delete --exclude __pycache__ --exclude tests "$REPO/edge/" "$JETSON:hearth-watch/edge/" || exit 1
ssh "$JETSON" 'test -f ~/hearth-watch/models/res10/deploy.prototxt' \
  || { echo "face model missing on the Jetson: ssh $JETSON bash hearth-watch/edge/tools/get_face_model.sh"; exit 1; }

mkdir -p "$REPO/recordings"
results="$REPO/recordings/face_test-$(date +%Y%m%d-%H%M%S).md"
printf '# Face probe results %s\n\n' "$(date '+%Y-%m-%d %H:%M')" > "$results"

stream_arg=""
[ "$video" = 1 ] && stream_arg="--stream $STREAM_URL"
for label in "${labels[@]}"; do
  printf '\n=== %s: %s (%s s)\n' "$label" "$(instructions "$label")" "$seconds"
  read -r -p "press Enter when ready (s = skip, q = quit): " answer
  case "$answer" in q|Q) break ;; s|S) continue ;; esac
  if [ "$video" = 1 ]; then
    setsid "$TOOLS/view_camera.sh" --player-only >/dev/null 2>&1 &
    player_pid=$!
  fi
  echo "loading models on the Jetson (~10 s), then probing..."
  ssh -tt "$JETSON" "cd ~/hearth-watch/edge && python3 tools/face_probe.py --label '$label' \
    --seconds $seconds $stream_arg 2>/dev/null" | tr -d '\r' | tee -a "$results"
  echo >> "$results"
  stop_player
done

echo
echo "saved: ${results#$REPO/}"

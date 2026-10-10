#!/usr/bin/env bash
# Shared runner for the guided probe tests (face_test.sh and pose_test.sh call this). LAPTOP:
#     edge/tools/probe_test.sh face [--seconds 15] [--countdown 3] [--no-video] [labels...]
#     edge/tools/probe_test.sh pose [--seconds 15] [--countdown 3] [--no-video] [labels...]
#
# One probe session runs on the Jetson for the whole test (models load once, ~10 s) and the
# camera window stays live, so you can see yourself and get into position BEFORE each run.
# Press Enter: a short countdown, then it measures for --seconds and prints the table, and the
# live view continues for the next situation. Type s + Enter to skip one, q + Enter to quit.
# All tables are also saved to recordings/<face|pose>_test-<time>.md (git ignores recordings/).
# Nothing else is saved; the window is a live view only.
# JETSON is the ssh target (default: the ssh-config alias "jetson").
set -u
JETSON="${JETSON:-jetson}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
STREAM_URL="rtp://192.168.55.100:1234"

mode="${1:-}"
shift || true
case "$mode" in
  face) probe="face_probe.py"; default_labels=(close 3m mouth-covered 3m-mouth-covered) ;;
  pose) probe="pose_probe.py"; default_labels=(standing across toward) ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
MATCH="^python3 -u tools/$probe --control-stdin"  # anchored: never matches pkill's own shell

instructions() {
  case "$1" in
    close) echo "sit or stand about 1 m from the camera, facing it" ;;
    3m) echo "stand about 3 m back, facing the camera" ;;
    mouth-covered) echo "about 1 m away, facing the camera, cover your mouth (hand or mask)" ;;
    3m-mouth-covered) echo "about 3 m back, facing the camera, cover your mouth (hand or mask)" ;;
    standing) echo "stand where you normally stand, whole body in view (baseline)" ;;
    across) echo "lie on the mat sideways ACROSS the picture (head left or right), stay still" ;;
    toward) echo "lie on the mat with your feet TOWARD the camera, head away, stay still" ;;
    *) echo "get into position for: $1" ;;
  esac
}

seconds=15
countdown=3
video=1
labels=()
while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) seconds="$2"; shift ;;
    --countdown) countdown="$2"; shift ;;
    --no-video) video=0 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)"; exit 2 ;;
    *) labels+=("$1") ;;
  esac
  shift
done
[ ${#labels[@]} -gt 0 ] || labels=("${default_labels[@]}")

tmp="$(mktemp -d)"
ctl="$tmp/control"
log="$tmp/session.log"
player_pid=""
session_pid=""
cleanup() {
  trap - EXIT INT TERM
  { echo quit >&3; } 2>/dev/null
  exec 3>&- 2>/dev/null
  if [ -n "$session_pid" ]; then
    sleep 1
    ssh "$JETSON" "pkill -INT -f '$MATCH'; sleep 2; pkill -KILL -f '$MATCH'" 2>/dev/null
    kill "$session_pid" 2>/dev/null
    wait "$session_pid" 2>/dev/null
  fi
  [ -n "$player_pid" ] && kill "$player_pid" 2>/dev/null
  rm -rf "$tmp"
}
trap cleanup EXIT INT TERM

wait_for_log() {  # pattern, minimum count, timeout seconds
  local end=$((SECONDS + $3))
  while [ "$(grep -c -- "$1" "$log" 2>/dev/null || true)" -lt "$2" ]; do
    kill -0 "$session_pid" 2>/dev/null || return 1
    [ "$SECONDS" -ge "$end" ] && return 1
    sleep 0.2
  done
}

holder=$(ssh -o ConnectTimeout=5 "$JETSON" 'fuser /dev/video0 2>/dev/null | xargs -r ps -o pid=,args= -p') \
  || { echo "could not reach the Jetson as '$JETSON' (USB cable plugged in?)"; exit 1; }
if [ -n "$holder" ]; then
  echo "the Jetson camera is in use by:"; echo "  $holder"; echo "stop that program first"; exit 1
fi
echo "copying edge/ to the Jetson..."
rsync -a --delete --exclude __pycache__ --exclude tests "$REPO/edge/" "$JETSON:hearth-watch/edge/" || exit 1
if [ "$mode" = face ]; then
  ssh "$JETSON" 'test -f ~/hearth-watch/models/res10/deploy.prototxt' \
    || { echo "face model missing on the Jetson: ssh $JETSON bash hearth-watch/edge/tools/get_face_model.sh"; exit 1; }
fi

stream_arg=""
if [ "$video" = 1 ]; then
  stream_arg="--stream $STREAM_URL"
  setsid "$TOOLS/view_camera.sh" --player-only >/dev/null 2>&1 &
  player_pid=$!
fi
mkfifo "$ctl"
: > "$log"
echo "starting the $mode probe session on the Jetson (models load in ~10 s)..."
ssh "$JETSON" "cd ~/hearth-watch/edge && python3 -u tools/$probe --control-stdin $stream_arg 2>/dev/null" \
  < "$ctl" | tee "$log" &  # no tr/grep here: anything that buffers would hide "control: ready"
session_pid=$!
exec 3>"$ctl"  # keep the command pipe open for the whole session
wait_for_log "control: ready" 1 90 || { echo "the probe session did not start"; exit 1; }
[ "$video" = 1 ] && echo "camera window is live: check the framing before each run"

mkdir -p "$REPO/recordings"
results="$REPO/recordings/${mode}_test-$(date +%Y%m%d-%H%M%S).md"
for label in "${labels[@]}"; do
  printf '\n=== %s: %s (%s s)\n' "$label" "$(instructions "$label")" "$seconds"
  read -r -p "get into position (watch the window), then press Enter (s = skip, q = quit): " answer
  case "$answer" in q|Q) break ;; s|S) continue ;; esac
  before=$(grep -c "probe done" "$log" 2>/dev/null || true)
  echo "probe $label $seconds $countdown" >&3
  if ! wait_for_log "probe done" $((before + 1)) $(( ${seconds%.*} + ${countdown%.*} + 30 )); then
    echo "the run did not finish (probe session stopped?)"; break
  fi
done

echo quit >&3
exec 3>&-
wait "$session_pid" 2>/dev/null
session_pid=""
{ printf '# %s probe results %s\n\n' "$mode" "$(date '+%Y-%m-%d %H:%M')"
  grep -vE '^(control:|probe .* starts in|measuring |probe done)' "$log"; } > "$results"
echo
echo "saved: ${results#$REPO/}"

#!/usr/bin/env bash
# Guided recording of staged clips, with a live camera view the whole time. Run on the LAPTOP:
#     edge/tools/record_clips.sh stand sit lie_down
#     edge/tools/record_clips.sh --takes 3 --seconds 8 sit bend lie_down
#     edge/tools/record_clips.sh --no-video stand        # no camera window
#
# One run_edge.py session runs on the Jetson for the whole script (models load once, ~10 s),
# and the camera window (skeleton, face boxes, info panel with our posture model) stays open.
# Before every clip you see yourself live and can position; press Enter and the countdown
# (--countdown, default 5 s) starts right away, then it records for --seconds and goes back to
# the live view. Type s + Enter to skip a clip, q + Enter to quit.
#   standing: whole body in view, head to feet
#   lying on the mat, sideways across the picture: whole body in view, side to side
# Clips are saved on the Jetson in ~/hearth-watch/recordings/ and copied to <repo>/recordings/
# at the end (git ignores that folder). Only keypoints and posture numbers are saved, never
# video (each observation includes our posture model's posture_prediction, as live); the window
# is a live view only. Uses --dry-run: nothing is sent to any backend.
# Check the clips afterwards: .venv/bin/python edge/tools/check_rules.py recordings/<clip>.jsonl
#
# Safety: for lie-down clips, lower yourself slowly onto a mat. Never stage a real fall.
# JETSON is the ssh target (default: the ssh-config alias "jetson").
set -u
JETSON="${JETSON:-jetson}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
STREAM_URL="rtp://192.168.55.100:1234"
EDGE_MATCH="^python3 -u run_edge.py --dry-run"  # anchored: never matches pkill's own shell

takes=1
seconds=10
countdown=5
video=1
labels=()
while [ $# -gt 0 ]; do
  case "$1" in
    --takes) takes="$2"; shift ;;
    --seconds) seconds="$2"; shift ;;
    --countdown) countdown="$2"; shift ;;
    --no-video) video=0 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)"; exit 2 ;;
    *) labels+=("$1") ;;
  esac
  shift
done
[ ${#labels[@]} -gt 0 ] || { sed -n '2,21p' "$0"; exit 2; }

tmp="$(mktemp -d)"
ctl="$tmp/control"   # commands to run_edge.py (via ssh stdin)
log="$tmp/edge.log"  # its log lines, to know when it is ready / a clip is done
player_pid=""
edge_pid=""
cleanup() {
  trap - EXIT INT TERM HUP
  { echo quit >&3; } 2>/dev/null
  exec 3>&- 2>/dev/null
  if [ -n "$edge_pid" ]; then
    sleep 1
    ssh "$JETSON" "pkill -INT -f '$EDGE_MATCH'; sleep 2; pkill -KILL -f '$EDGE_MATCH'" 2>/dev/null
    kill "$edge_pid" 2>/dev/null
    wait "$edge_pid" 2>/dev/null
  fi
  [ -n "$player_pid" ] && kill "$player_pid" 2>/dev/null
  rm -rf "$tmp"
}
trap cleanup EXIT INT TERM HUP

wait_for_log() {  # pattern, minimum count, timeout seconds
  local end=$((SECONDS + $3))
  while [ "$(grep -c -- "$1" "$log" 2>/dev/null || true)" -lt "$2" ]; do
    kill -0 "$edge_pid" 2>/dev/null || return 1
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

stream_arg=""
if [ "$video" = 1 ]; then
  stream_arg="--stream $STREAM_URL"
  setsid "$TOOLS/view_camera.sh" --player-only >/dev/null 2>&1 &
  player_pid=$!
fi
mkfifo "$ctl"
: > "$log"
echo "starting the camera session on the Jetson (models load in ~10 s)..."
ssh "$JETSON" "cd ~/hearth-watch/edge && python3 -u run_edge.py --dry-run $stream_arg --control-stdin \
  --status-every 9999 2>&1 >/dev/null | grep --line-buffered -aE 'control|RECORDING|recording|no frames|Traceback|Error'" \
  < "$ctl" | tee "$log" &
edge_pid=$!
exec 3>"$ctl"  # keep the command pipe open for the whole session
wait_for_log "control: ready" 1 90 || { echo "the camera session did not start (see messages above)"; exit 1; }
[ "$video" = 1 ] && echo "camera window is live: position the camera and the person now"

recorded=0
for label in "${labels[@]}"; do
  for take in $(seq 1 "$takes"); do
    printf '\n=== next: "%s"  take %d/%d  (%s s, %s s countdown)\n' "$label" "$take" "$takes" "$seconds" "$countdown"
    read -r -p "get into position (watch the window), then press Enter to start (s = skip clip, q = quit): " answer
    case "$answer" in
      q|Q) break 2 ;;
      s|S) break ;;
    esac
    before=$(grep -c "recording done" "$log" 2>/dev/null || true)
    echo "record $label $seconds $countdown" >&3
    if ! wait_for_log "recording done" $((before + 1)) $(( ${seconds%.*} + ${countdown%.*} + 30 )); then
      echo "the clip did not finish (camera session stopped?)"; break 2
    fi
    recorded=$((recorded + 1))
  done
done

echo quit >&3
exec 3>&-
wait "$edge_pid" 2>/dev/null
edge_pid=""
echo
echo "copying recordings to $REPO/recordings/ ..."
mkdir -p "$REPO/recordings"
rsync -a "$JETSON:hearth-watch/recordings/" "$REPO/recordings/" || exit 1
echo "$recorded clip(s) recorded this session. Newest files:"
ls -t "$REPO/recordings/"*.jsonl 2>/dev/null | head -n "$((recorded > 0 ? recorded : 5))" \
  | while read -r f; do printf '  %4d observations  %s\n' "$(wc -l < "$f")" "${f#$REPO/}"; done
echo "check them: .venv/bin/python edge/tools/check_rules.py <clip>.jsonl --frames"

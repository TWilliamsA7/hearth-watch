#!/usr/bin/env bash
# Continuous live view of the Jetson camera on the laptop, through the real edge pipeline:
# skeleton, red face boxes, and a status line every 5 s in this terminal.
# Runs run_edge.py --dry-run --stream on the Jetson: nothing is sent to a backend, nothing is
# recorded; the video only goes to this window over the USB link.
#
# Run on the LAPTOP:
#   edge/tools/view_camera.sh                 # until you close the window or press Ctrl+C
#   edge/tools/view_camera.sh --player-only   # window only, for something already streaming
#                                             # (e.g. run_edge.py --stream rtp://192.168.55.100:1234)
#
# Needs ffplay (ffmpeg). JETSON is the ssh target (default: the ssh-config alias "jetson").
# Only one program can use the camera: stop run_edge.py / live_test.sh first.
JETSON="${JETSON:-jetson}"
here="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$here/../.." && pwd)"
STREAM_URL="rtp://192.168.55.100:1234"
# Anchored with ^ so the remote pkill never matches its own "bash -c 'pkill ...'" shell.
EDGE_MATCH="^python3 -u run_edge.py --dry-run --stream"

# Low-latency playback: no input buffering or stream analysis, low-delay decoding, drop late
# frames, and show frames as they arrive instead of pacing them (-sync ext).
FFPLAY=(ffplay -loglevel error -window_title "Jetson camera (Hearth edge)"
  -protocol_whitelist file,udp,rtp -fflags nobuffer -flags low_delay -framedrop
  -probesize 32 -analyzeduration 0 -sync ext
  -i "$here/stream.sdp")

if [ "${1:-}" = "--player-only" ]; then
  exec "${FFPLAY[@]}"   # exec: stopping this script closes the window too
fi

holder=$(ssh -o ConnectTimeout=5 "$JETSON" 'fuser /dev/video0 2>/dev/null | xargs -r ps -o pid=,args= -p') \
  || { echo "could not reach the Jetson as '$JETSON' (USB cable plugged in?)"; exit 1; }
if [ -n "$holder" ]; then
  echo "the Jetson camera is in use by:"; echo "  $holder"; echo "stop that program first"; exit 1
fi
rsync -a --delete --exclude __pycache__ --exclude tests "$REPO/edge/" "$JETSON:hearth-watch/edge/" || exit 1

edge_pid=""
cleanup() {
  trap - EXIT INT TERM HUP
  echo "stopping the Jetson side..."
  ssh "$JETSON" "pkill -INT -f '$EDGE_MATCH'; sleep 2; pkill -KILL -f '$EDGE_MATCH'" 2>/dev/null
  [ -n "$edge_pid" ] && kill "$edge_pid" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM HUP

ssh "$JETSON" "cd ~/hearth-watch/edge && python3 -u run_edge.py --dry-run --stream $STREAM_URL \
  2>&1 >/dev/null | grep --line-buffered -aE '\[edge'" &
edge_pid=$!
echo "starting (models load in ~10 s). Close the window or press Ctrl+C to stop."
"${FFPLAY[@]}"   # returns when the window is closed

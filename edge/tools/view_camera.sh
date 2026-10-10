#!/usr/bin/env bash
# Live view of the Jetson camera on the laptop, with the poseNet skeleton drawn on top.
# The Jetson encodes H.264 and streams RTP over the USB link to 192.168.55.100:1234.
# Nothing is recorded; the video only goes to this window.
#
# Run on the LAPTOP:
#   edge/tools/view_camera.sh                 # starts NVIDIA's posenet demo on the Jetson + this window
#   edge/tools/view_camera.sh --player-only   # window only, for `run_edge.py --stream rtp://192.168.55.100:1234`
# Close the window or press Ctrl+C to stop.
#
# Needs ffplay (ffmpeg) on the laptop. JETSON is the ssh target (default: an ssh-config alias
# named "jetson"); e.g.  JETSON=g@192.168.55.1 edge/tools/view_camera.sh
# Only one program can use the camera: stop run_edge.py before using the default mode.
JETSON="${JETSON:-jetson}"
here="$(cd "$(dirname "$0")" && pwd)"

FFPLAY=(ffplay -loglevel error -window_title "Jetson camera + pose"
  -protocol_whitelist file,udp,rtp -fflags nobuffer -flags low_delay -framedrop
  -i "$here/stream.sdp")

if [ "${1:-}" = "--player-only" ]; then
  exec "${FFPLAY[@]}"   # exec: stopping this script closes the window too
fi

player() { "${FFPLAY[@]}"; }

# Stop the Jetson side for sure: SIGINT alone once left posenet holding the camera.
cleanup() {
  trap - EXIT INT TERM
  ssh "$JETSON" 'pkill -TERM -f "^posenet /dev/video0"; sleep 1; pkill -KILL -f "^posenet /dev/video0"' 2>/dev/null
  kill "$player_pid" 2>/dev/null
}
trap cleanup EXIT INT TERM

player &
player_pid=$!

# -tt ties posenet to this ssh session, so closing the connection also stops it.
ssh -tt "$JETSON" 'pkill -f "^posenet /dev/video0" 2>/dev/null; posenet /dev/video0 --input-codec=mjpeg --input-width=640 --input-height=480 \
  --headless rtp://192.168.55.100:1234 2>&1 | grep --line-buffered -E "detected [0-9]+ objects" '

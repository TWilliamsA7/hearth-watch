#!/usr/bin/env bash
# One-command live test, run on the LAPTOP from anywhere:
#     edge/tools/live_test.sh              # backend + video window + edge on the Jetson
#     edge/tools/live_test.sh --no-video   # skip the video stream/window
#     edge/tools/live_test.sh --real       # Person 2's real backend (backend/) instead of the stub
#     edge/tools/live_test.sh --real --config recordings/config_framing.json
#                                          # ...with another monitor config (HEARTH_CONFIG)
#     edge/tools/live_test.sh -- --obs-hz 5   # anything after -- goes to run_edge.py
#
# It does, in order:
#   1. checks the Jetson camera and laptop port 8000 are free (nothing changes if not)
#   2. copies edge/ to the Jetson (~/hearth-watch/edge), so the Jetson runs your latest code
#   3. starts the stand-in backend (edge/tools/stub_backend.py) on the laptop, or with --real
#      the team backend (uvicorn hearth.api.app:app); then open http://127.0.0.1:8000/nurses
#   4. opens the video window (ffplay) unless --no-video
#   5. runs run_edge.py on the Jetson, streaming the skeleton video to the window. Our posture
#      model is on by default and its posture_prediction is what the team backend's alerts use
#      (edge/POSTURE_MODEL.md); "-- --no-posture-mlp" turns it off
# Press Ctrl+C once to stop the Jetson side. The backend keeps running ~5 s so you can see
# "jetson-01 -> source_offline" (silence is not safety), then everything is shut down.
#
# JETSON is the ssh target (default: the ssh-config alias "jetson"), e.g. JETSON=g@192.168.55.1
set -u
JETSON="${JETSON:-jetson}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
PY="$REPO/.venv/bin/python"
STREAM_URL="rtp://192.168.55.100:1234"

video=1
real=0
monitor_config=""
edge_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-video) video=0 ;;
    --real) real=1 ;;
    --config) monitor_config="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift ;;
    --) shift; edge_args=("$@"); break ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

say() { printf '\n=== %s\n' "$*"; }

stub_pid=""
player_pid=""
edge_started=0
cleanup() {
  trap - EXIT INT TERM HUP
  # Only stop a Jetson program this script started (never someone else's run).
  [ "$edge_started" = 1 ] && ssh "$JETSON" 'pkill -INT -f "^python3 -u run_edge.py"' 2>/dev/null
  [ -n "$player_pid" ] && kill "$player_pid" 2>/dev/null
  [ -n "$stub_pid" ] && kill "$stub_pid" 2>/dev/null
  wait 2>/dev/null
  say "all stopped"
}
trap cleanup EXIT INT TERM HUP

[ -x "$PY" ] || { echo "missing $PY - create it: python3 -m venv .venv && .venv/bin/pip install -e shared/python pytest"; exit 1; }

say "1/5 checking the camera and port 8000"
holder=$(ssh -o ConnectTimeout=5 "$JETSON" 'fuser /dev/video0 2>/dev/null | xargs -r ps -o pid=,args= -p') \
  || { echo "could not reach the Jetson as '$JETSON' (USB cable plugged in? try: ssh $JETSON)"; exit 1; }
if [ -n "$holder" ]; then
  echo "the Jetson camera is in use by:"; echo "  $holder"
  echo "stop that program first (Ctrl+C where it runs, or: ssh $JETSON pkill -INT -f run_edge.py)"
  exit 1
fi
if ss -ltn 'sport = :8000' | grep -q LISTEN; then
  pid=$(ss -ltnp 'sport = :8000' 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  args=$([ -n "$pid" ] && ps -o args= -p "$pid")
  parent=$([ -n "$pid" ] && ps -o comm= -p "$(ps -o ppid= -p "$pid" | tr -d ' ')")
  # A backend these tools started whose terminal went away (parent is now systemd/init): stop it.
  if [ -n "$pid" ] && echo "$args" | grep -qE 'uvicorn hearth\.api\.app:app|stub_backend\.py' \
     && { [ "$parent" = systemd ] || [ "$parent" = init ]; }; then
    echo "stopping a leftover backend from an earlier run (pid $pid)"
    kill "$pid"; sleep 2
  else
    echo "port 8000 on the laptop is already in use${pid:+ by pid $pid: $args}"
    echo "stop it first${pid:+ (kill $pid)}, or use the backend that is already running"
    exit 1
  fi
fi
echo "ok"

say "2/5 copying edge/ to the Jetson"
rsync -a --delete --exclude __pycache__ --exclude tests "$REPO/edge/" "$JETSON:hearth-watch/edge/" \
  || { echo "could not reach the Jetson as '$JETSON' (USB cable plugged in? try: ssh $JETSON)"; exit 1; }

if [ "$real" = 1 ]; then
  "$PY" -c "import fastapi, uvicorn, hearth.api.app" 2>/dev/null \
    || { echo "team backend not installed: $PY -m pip install -e '$REPO/backend[dev]'"; exit 1; }
  say "3/5 starting the TEAM backend on the laptop (uvicorn, port 8000)"
  if [ -n "$monitor_config" ]; then
    [ -f "$monitor_config" ] || { echo "config not found: $monitor_config"; exit 1; }
    export HEARTH_CONFIG="$monitor_config"
    echo "monitor config: $monitor_config"
  fi
  # setsid: keep it out of the terminal's Ctrl+C; it is stopped in cleanup.
  setsid "$PY" -m uvicorn hearth.api.app:app --app-dir "$REPO/backend" --host 0.0.0.0 \
    --port 8000 --log-level warning --no-access-log &
  stub_pid=$!
  sleep 3
  echo "open in a browser:  http://127.0.0.1:8000/nurses  (pick room-01 + Possible falls)"
  echo "                    http://127.0.0.1:8000/sources (live Jetson telemetry)"
else
  say "3/5 starting the stand-in backend on the laptop (lines from it start with [HH:MM:SS] jetson-01)"
  # setsid: keep it out of the terminal's Ctrl+C so it can report source_offline after the edge stops.
  setsid "$PY" -u "$TOOLS/stub_backend.py" &
  stub_pid=$!
  sleep 1
fi

stream=()
if [ "$video" = 1 ]; then
  say "4/5 opening the video window"
  setsid "$TOOLS/view_camera.sh" --player-only >/dev/null 2>&1 &
  player_pid=$!
  stream=(--stream "$STREAM_URL")
else
  say "4/5 video skipped (--no-video)"
fi

edge_started=1
say "5/5 starting run_edge.py on the Jetson (lines start with [edge ...]); Ctrl+C to stop"
# -tt sends Ctrl+C to the Jetson program itself, which then shuts down cleanly.
# The grep keeps our own log lines and drops NVIDIA's setup messages. It ignores Ctrl+C
# (trap '' INT) so it keeps reading until run_edge.py has finished shutting down; otherwise
# it dies first and run_edge.py loses its output mid-shutdown.
ssh -tt "$JETSON" "cd ~/hearth-watch/edge && python3 -u run_edge.py ${stream[*]} ${edge_args[*]} 2>&1 >/dev/null | (trap '' INT; grep --line-buffered -aE '\[edge')"

if [ "$real" = 1 ]; then
  say "edge stopped; the team backend stays up until you press Enter (check /nurses and /sources)"
  read -r _
else
  say "edge stopped; backend keeps listening 5 s - watch for 'jetson-01 -> source_offline'"
  sleep 5
fi

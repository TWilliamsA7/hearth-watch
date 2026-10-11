#!/usr/bin/env bash
# Face recognition on the Jetson, end to end. Run on the LAPTOP (Git Bash on Windows, or Linux)
# from anywhere in the repo:
#     scripts/face_id_demo.sh                 # set up what's missing, start the backend, then the Jetson
#     scripts/face_id_demo.sh --sync-clock    # ...also copy the laptop's clock to the Jetson (sudo there)
#     scripts/face_id_demo.sh --laptop-only   # backend + enrollment only, no Jetson
#     scripts/face_id_demo.sh --setup-only    # do every one-time step, start nothing on the Jetson
#     scripts/face_id_demo.sh -- --obs-hz 5   # anything after -- goes to run_edge.py
#
# Every step skips itself when already done, so it is safe to re-run:
#   1. laptop Python env (.venv) with the backend and its face-demo extra
#   2. YuNet + SFace models for enrollment (pinned, checksummed) in models/face-demo/
#   3. the encryption key and the Jetson token, made once and kept in .env.face-demo
#      (git-ignored; never printed). Losing the key loses the enrollments.
#   4. the backend with face recognition on, in its own window, on 0.0.0.0:$PORT
#   5. the Jetson: reachable, camera free, clock within 1 s of the laptop
#   6. the Jetson: latest edge/ code, the SFace file (copied from the laptop, so no internet
#      needed there), its TensorRT engine (built once, a few minutes), the token file
#   7. after you enroll on the laptop: run_edge.py --face-id on the Jetson; Ctrl+C stops it
#
# JETSON is the ssh target (default: the ssh-config alias "jetson"). LAPTOP_IP is the laptop as
# the Jetson sees it (default 192.168.55.100, its USB link). PORT defaults to 8000.
set -u
JETSON="${JETSON:-jetson}"
LAPTOP_IP="${LAPTOP_IP:-192.168.55.100}"
PORT="${PORT:-8000}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SECRETS="$REPO/.env.face-demo"
SFACE="face_recognition_sface_2021dec.onnx"

laptop_only=0
setup_only=0
sync_clock=0
edge_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --laptop-only) laptop_only=1 ;;
    --setup-only) setup_only=1 ;;
    --sync-clock) sync_clock=1 ;;
    --) shift; edge_args=("$@"); break ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

say() { printf '\n=== %s\n' "$*"; }
die() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }
windows=0
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) windows=1 ;; esac

backend_pid=""
edge_started=0
cleanup() {
  trap - EXIT INT TERM HUP
  # Only stop a Jetson program this script started (never someone else's run).
  [ "$edge_started" = 1 ] && ssh "$JETSON" 'pkill -INT -f "^python3 -u run_edge.py"' 2>/dev/null
  if [ -n "$backend_pid" ]; then
    kill "$backend_pid" 2>/dev/null
    echo "backend stopped"
  elif [ "$windows" = 1 ] && [ "$started_backend" = 1 ]; then
    echo "the backend keeps running in its own window; close that window to stop it"
  fi
}
started_backend=0
trap cleanup EXIT INT TERM HUP

# --- 1. laptop Python env ----------------------------------------------------------------------
say "1/7 laptop Python env"
if [ -x "$REPO/.venv/Scripts/python.exe" ]; then PY="$REPO/.venv/Scripts/python.exe"
elif [ -x "$REPO/.venv/bin/python" ]; then PY="$REPO/.venv/bin/python"
else
  base=""
  for c in python3 python py; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
      base="$c"; break
    fi
  done
  [ -n "$base" ] || die "need Python 3.11 or newer on PATH to create .venv"
  echo "creating .venv with $base"
  "$base" -m venv "$REPO/.venv" || die "could not create .venv"
  if [ -x "$REPO/.venv/Scripts/python.exe" ]; then PY="$REPO/.venv/Scripts/python.exe"; else PY="$REPO/.venv/bin/python"; fi
fi
if ! (cd "$REPO" && "$PY" -c "import hearth.api.app, cv2, cryptography, PIL, uvicorn") 2>/dev/null; then
  echo "installing the backend with its face-demo extra (a few minutes the first time)"
  (cd "$REPO" && "$PY" -m pip install -q -e ./shared/python -e "./backend[dev,face-demo]") \
    || die "pip install failed (see above)"
fi
echo "ok ($PY)"

# --- 2. enrollment models ------------------------------------------------------------------------
say "2/7 YuNet + SFace models"
(cd "$REPO" && "$PY" -m hearth.perception.face_demo --download) || die "model download/check failed"

# --- 3. secrets ----------------------------------------------------------------------------------
say "3/7 encryption key and Jetson token"
if [ ! -f "$SECRETS" ]; then
  (umask 077 && "$PY" - > "$SECRETS" <<'PY'
import secrets
from cryptography.fernet import Fernet
print("HEARTH_FACE_KEY=" + Fernet.generate_key().decode())
print("HEARTH_EDGE_FACE_TOKEN=" + secrets.token_urlsafe(32))
PY
  ) || die "could not write $SECRETS"
  echo "made $SECRETS (keep it: a new key cannot read existing enrollments)"
else
  echo "using $SECRETS"
fi
set -a
# Python on Windows writes CRLF; a stray \r would end up inside the key and token.
eval "$(tr -d '\r' < "$SECRETS")"
set +a
[ -n "${HEARTH_FACE_KEY:-}" ] && [ -n "${HEARTH_EDGE_FACE_TOKEN:-}" ] || die "$SECRETS is missing a value; delete it to make a new one"
export HEARTH_FACE_DEMO=1

# --- 4. backend ----------------------------------------------------------------------------------
say "4/7 backend with face recognition on port $PORT"
status() { curl -s -m 3 "http://127.0.0.1:$PORT/face-demo/status"; }
if curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "a backend is already running on port $PORT"
else
  if [ "$windows" = 1 ]; then
    # Its own window, like scripts/start.ps1; it inherits the variables exported above.
    # cmd /k keeps the window open if it fails, so the error stays readable.
    (cd "$REPO" && MSYS2_ARG_CONV_EXCL='*' cmd.exe /c start "Hearth backend (face recognition)" \
      cmd.exe /k "$(cygpath -w "$PY")" -m uvicorn hearth.api.app:app --host 0.0.0.0 --port "$PORT" \
      < /dev/null > /dev/null 2>&1)   # the window must not hold this script's output open
  else
    mkdir -p "$REPO/data"
    (cd "$REPO" && exec "$PY" -m uvicorn hearth.api.app:app --host 0.0.0.0 --port "$PORT" \
      > "$REPO/data/backend.log" 2>&1) &
    backend_pid=$!
    echo "log: data/backend.log"
  fi
  started_backend=1
  for _ in $(seq 60); do
    curl -s -m 1 "http://127.0.0.1:$PORT/health" >/dev/null && break
    sleep 0.5
  done
  curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null || die "the backend did not start (see its window/log)"
fi
s="$(status)"
case "$s" in
  *'"enabled":true'*'edge_templates'*) echo "ok: face recognition on, Jetson templates accepted" ;;
  *'"enabled":true'*) die "that backend has no HEARTH_EDGE_FACE_TOKEN. Stop it and re-run this script" ;;
  *'"enabled":false'*) die "face recognition is off in that backend: $s
  (a backend started without these settings? stop it and re-run this script)" ;;
  *) die "unexpected answer from /face-demo/status: $s" ;;
esac

enroll_help() {
  cat <<EOF

  Enroll on THIS laptop (its webcam, in the browser):
    http://127.0.0.1:$PORT/room#recognition
    - type a name, tick consent AND "Also show my name", start the camera, enroll 3 views
    - then click "Stop camera" (enrollment stays loopback-only; the Jetson never sees it)
  Watch alerts:
    http://127.0.0.1:$PORT/   (start a nurse session, pick room-01 and Possible falls)
EOF
}

keep_backend() {   # a backend this script runs in the background (Linux) lives until Enter
  if [ -n "$backend_pid" ]; then
    printf '\n  Press Enter to stop the backend... '
    read -r _
  fi
}

if [ "$laptop_only" = 1 ]; then
  enroll_help
  echo
  echo "Jetson later: scripts/face_id_demo.sh"
  keep_backend
  exit 0
fi

# --- 5. Jetson reachable, camera free, clock ------------------------------------------------------
say "5/7 Jetson ($JETSON): reachable, camera free, clock"
ssh -o ConnectTimeout=5 "$JETSON" true 2>/dev/null \
  || die "cannot reach the Jetson as '$JETSON' (USB cable in? powered on? try: ssh $JETSON)"
holder=$(ssh "$JETSON" 'fuser /dev/video0 2>/dev/null | xargs -r ps -o pid=,args= -p')
[ -z "$holder" ] || die "the Jetson camera is in use by: $holder
  stop it first (Ctrl+C where it runs, or: ssh $JETSON pkill -INT -f run_edge.py)"
offset() {   # Jetson clock minus laptop clock, in seconds (includes about half an ssh round trip)
  local before after remote
  before=$(date +%s.%N); remote=$(ssh "$JETSON" 'date +%s.%N'); after=$(date +%s.%N)
  awk -v b="$before" -v a="$after" -v r="$remote" 'BEGIN { printf "%.2f", r - (a + b) / 2 }'
}
off=$(offset)
if awk -v o="$off" 'BEGIN { exit !(o > 1 || o < -1) }'; then
  if [ "$sync_clock" = 1 ]; then
    echo "Jetson clock is ${off}s off; setting it from the laptop (asks for the Jetson password)"
    ssh -t "$JETSON" "sudo date -u -s @$(date -u +%s.%N) >/dev/null" || die "could not set the Jetson clock"
    off=$(offset)
    awk -v o="$off" 'BEGIN { exit !(o > 1 || o < -1) }' && die "Jetson clock still ${off}s off"
  else
    die "the Jetson clock is ${off}s off the laptop's; the backend drops data older than 3 s.
  Re-run with --sync-clock (asks for the Jetson password), or use edge/tools/sync_clock.sh"
  fi
fi
echo "ok (clock offset ${off}s)"

# --- 6. Jetson code, model, engine, token -------------------------------------------------------
say "6/7 Jetson: edge code, SFace engine, token"
# No rsync on Git Bash: tar over ssh, swapped in whole like rsync --delete. Windows checkouts
# have CRLF line endings, which bash on the Jetson cannot run, so they are stripped there.
tar -C "$REPO" --exclude=__pycache__ --exclude=tests -cf - edge | ssh "$JETSON" '
  set -e; mkdir -p ~/hearth-watch && cd ~/hearth-watch
  rm -rf edge.new && mkdir edge.new && tar -xf - -C edge.new --strip-components=1
  find edge.new \( -name "*.sh" -o -name "*.py" \) -exec sed -i "s/\r\$//" {} +
  rm -rf edge && mv edge.new edge' || die "could not copy edge/ to the Jetson"
echo "edge/ copied"
ssh "$JETSON" 'test -f ~/hearth-watch/models/res10/res10_300x300_ssd_iter_140000.caffemodel' \
  || die "the Jetson has no res10 face detector; run once (needs internet there):
  ssh $JETSON 'bash ~/hearth-watch/edge/tools/get_face_model.sh'"
if ! ssh "$JETSON" "test -f ~/hearth-watch/models/face-id/$SFACE"; then
  ssh "$JETSON" 'mkdir -p ~/hearth-watch/models/face-id' \
    && scp -q "$REPO/models/face-demo/$SFACE" "$JETSON:hearth-watch/models/face-id/$SFACE" \
    || die "could not copy SFace to the Jetson"
  echo "SFace copied from the laptop (checksum verified on the Jetson next)"
fi
echo "building/checking the SFace TensorRT engine (the first build takes a few minutes)"
ssh "$JETSON" 'bash ~/hearth-watch/edge/tools/get_face_id_model.sh' || die "SFace engine build failed (see above)"
printf '%s\n' "$HEARTH_EDGE_FACE_TOKEN" | ssh "$JETSON" 'umask 077; cat > ~/.hearth-face-token' \
  || die "could not write the token on the Jetson"
echo "token in ~/.hearth-face-token on the Jetson (only you can read it)"
ssh "$JETSON" "curl -s -m 3 http://$LAPTOP_IP:$PORT/health" >/dev/null || die "the Jetson cannot reach the backend at http://$LAPTOP_IP:$PORT
  (Windows firewall: allow Python on private networks, or LAPTOP_IP=<laptop address as the Jetson sees it>)"
echo "ok: the Jetson reaches the backend at http://$LAPTOP_IP:$PORT"

if [ "$setup_only" = 1 ]; then
  say "set up; nothing started on the Jetson (--setup-only)"
  enroll_help
  keep_backend
  exit 0
fi

# --- 7. enroll, then run ---------------------------------------------------------------------------
say "7/7 enroll, then start the Jetson"
enroll_help
printf '\n  Press Enter when enrolled (and the browser camera is stopped) to start the Jetson... '
read -r _
cat <<EOF

  Stand in view of the Jetson camera, facing it, close enough that it sees your face clearly.
  The status line counts "face templates"; after 3 matching checks your name is attached.
  Then lie down on the mat: the nurse alert reads "Possible fall · <your demo name>".
  Anyone not enrolled stays unnamed. Ctrl+C stops the Jetson.
EOF
edge_started=1
# -tt sends Ctrl+C to run_edge.py itself so it shuts down cleanly; the grep keeps our log lines
# and drops NVIDIA's, and ignores Ctrl+C so it reads until run_edge.py has finished.
ssh -tt "$JETSON" "cd ~/hearth-watch/edge && python3 -u run_edge.py --face-id \
  --face-id-token-file ~/.hearth-face-token --backend http://$LAPTOP_IP:$PORT ${edge_args[*]} \
  2>&1 >/dev/null | (trap '' INT; grep --line-buffered -aE '\[edge')"
say "Jetson stopped"
keep_backend

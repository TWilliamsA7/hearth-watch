#!/usr/bin/env bash
# Guided recording of staged clips for tuning the backend's fall logic. Run on the LAPTOP:
#     edge/tools/record_clips.sh walk stand sit bend lie_down lie_then_up out_of_frame face_away
#     edge/tools/record_clips.sh --takes 3 --seconds 8 sit bend lie_down
#
# For each clip and take: press Enter, the Jetson loads the model (~10 s), counts down 3-2-1,
# records for --seconds, and stops. Type s + Enter to skip a clip, q + Enter to quit.
# Clips are saved on the Jetson in ~/hearth-watch/recordings/ and copied to <repo>/recordings/
# at the end (git ignores that folder). Only keypoints and posture numbers are saved, never
# video. Uses --dry-run, so nothing is sent to any backend while recording.
#
# Safety: for lie-down clips, lower yourself slowly onto a mat. Never stage a real fall.
# JETSON is the ssh target (default: the ssh-config alias "jetson").
set -u
JETSON="${JETSON:-jetson}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"

takes=1
seconds=10
labels=()
while [ $# -gt 0 ]; do
  case "$1" in
    --takes) takes="$2"; shift ;;
    --seconds) seconds="$2"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)"; exit 2 ;;
    *) labels+=("$1") ;;
  esac
  shift
done
[ ${#labels[@]} -gt 0 ] || { sed -n '2,15p' "$0"; exit 2; }

holder=$(ssh -o ConnectTimeout=5 "$JETSON" 'fuser /dev/video0 2>/dev/null | xargs -r ps -o pid=,args= -p') \
  || { echo "could not reach the Jetson as '$JETSON' (USB cable plugged in?)"; exit 1; }
if [ -n "$holder" ]; then
  echo "the Jetson camera is in use by:"; echo "  $holder"; echo "stop that program first"; exit 1
fi

echo "copying edge/ to the Jetson..."
rsync -a --delete --exclude __pycache__ --exclude tests "$REPO/edge/" "$JETSON:hearth-watch/edge/" || exit 1

recorded=0
for label in "${labels[@]}"; do
  for take in $(seq 1 "$takes"); do
    printf '\n=== next: "%s"  take %d/%d  (%s s)\n' "$label" "$take" "$takes" "$seconds"
    read -r -p "get into position, then press Enter (s = skip clip, q = quit): " answer
    case "$answer" in
      q|Q) break 2 ;;
      s|S) break ;;
    esac
    ssh -tt "$JETSON" "cd ~/hearth-watch/edge && python3 -u run_edge.py --dry-run --record '$label' \
      --seconds $seconds --countdown 3 --status-every 9999 2>&1 >/dev/null \
      | (trap '' INT; grep --line-buffered -aE 'RECORDING|recording|no frames')"
    recorded=$((recorded + 1))
  done
done

echo
echo "copying recordings to $REPO/recordings/ ..."
mkdir -p "$REPO/recordings"
rsync -a "$JETSON:hearth-watch/recordings/" "$REPO/recordings/" || exit 1
echo "$recorded clip(s) recorded this session. Newest files:"
ls -t "$REPO/recordings/"*.jsonl 2>/dev/null | head -n "$((recorded > 0 ? recorded : 5))" \
  | while read -r f; do printf '  %4d observations  %s\n' "$(wc -l < "$f")" "${f#$REPO/}"; done

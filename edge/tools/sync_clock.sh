#!/usr/bin/env bash
# Copy the laptop's clock to the Jetson. Run on the LAPTOP before a demo, especially when the
# Jetson has no internet (no NTP), e.g. at the venue:
#     edge/tools/sync_clock.sh           # measure, and correct the Jetson if off by > 50 ms
#     edge/tools/sync_clock.sh --check   # only measure, change nothing
#     edge/tools/sync_clock.sh --force   # correct it even if already close
#
# Why: the team backend times falls with the timestamps on the Jetson's observations (the
# 3 s floor hold) and checks freshness against its own clock. A Jetson clock that drifted
# seconds or minutes makes data look stale or shifts alert timing.
# How: the offset is measured over one open ssh connection (round trips of a few ms, not the
# ~250 ms of opening a connection), then the Jetson subtracts that offset from its own clock,
# so connection delays never leak into the result. Correcting needs sudo on the Jetson: you
# type the Jetson password once, before anything is changed.
# The laptop's own clock should be right (it syncs over the internet when online).
# JETSON is the ssh target (default: the ssh-config alias "jetson").
set -u
JETSON="${JETSON:-jetson}"
mode="${1:-auto}"
case "$mode" in auto|--check|--force) ;; *) sed -n '2,17p' "$0"; exit 2 ;; esac
LIMIT_MS=50

offset() {
  # Prints "<offset_ms> <round_trip_ms>": Jetson time minus laptop time, from the fastest of
  # 20 request/reply round trips over a single ssh connection.
  python3 - "$JETSON" <<'PY'
import subprocess, sys, time
remote = ("python3 -c \"import sys, time\n"
          "for line in sys.stdin:\n"
          "    sys.stdout.write(repr(time.time()) + '\\n'); sys.stdout.flush()\"")
p = subprocess.Popen(["ssh", "-o", "ConnectTimeout=5", sys.argv[1], remote],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1)
best = None
try:
    for _ in range(20):
        t0 = time.time()
        p.stdin.write("t\n"); p.stdin.flush()
        reply = p.stdout.readline()
        t1 = time.time()
        if not reply:
            sys.exit(1)
        rtt = t1 - t0
        off = float(reply) - (t0 + t1) / 2
        if best is None or rtt < best[0]:
            best = (rtt, off)
finally:
    p.stdin.close(); p.wait()
print("%.0f %.1f" % (best[1] * 1000, best[0] * 1000))
PY
}

read -r off rtt < <(offset) || { echo "could not reach the Jetson as '$JETSON'"; exit 1; }
printf 'Jetson clock is %+d ms vs the laptop (measured with a %s ms round trip)\n' "$off" "$rtt"
if [ "$mode" = --check ]; then exit 0; fi
if [ "$mode" = auto ] && [ "${off#-}" -le "$LIMIT_MS" ]; then
  echo "within $LIMIT_MS ms: nothing to do (use --force to correct it anyway)"
  exit 0
fi

echo "correcting the Jetson clock by $((-off)) ms (enter the Jetson password if asked)..."
ssh -t "$JETSON" "sudo -v" || { echo "sudo failed"; exit 1; }
read -r off rtt < <(offset)  # re-measure right before correcting
# The Jetson reads its own clock and subtracts the offset: no ssh delay in the new time.
ssh "$JETSON" "sudo -n date -u -s @\$(python3 -c 'import time; print(\"%.3f\" % (time.time() - $off / 1000.0))') >/dev/null" \
  || { echo "correcting the clock failed"; exit 1; }
read -r off rtt < <(offset)
printf 'now %+d ms vs the laptop (measured with a %s ms round trip)\n' "$off" "$rtt"

#!/usr/bin/env bash
# Guided test: does rotating the frame help poseNet see people lying down? Camera window live
# the whole time. Run on the LAPTOP:
#     edge/tools/pose_test.sh                        # standing, across, toward
#     edge/tools/pose_test.sh --seconds 10 across    # pick your own situations
#     edge/tools/pose_test.sh --no-video             # tables only, no window
# Watch yourself in the window (green dots = poseNet), get into position, press Enter: a 3 s
# countdown, then 15 s comparing poseNet as-is vs rotated 90 degrees each way.
# Tables are saved to recordings/pose_test-<time>.md. See probe_test.sh for details.
# Safety: lower yourself slowly onto a mat. Never stage a real fall.
exec "$(dirname "$0")/probe_test.sh" pose "$@"

#!/usr/bin/env bash
# Guided face detection test with the camera window live the whole time. Run on the LAPTOP:
#     edge/tools/face_test.sh                       # close, 3m, mouth-covered, 3m-mouth-covered
#     edge/tools/face_test.sh --seconds 10 close 3m # pick your own situations
#     edge/tools/face_test.sh --no-video            # tables only, no window
# Watch yourself in the window, get into position, press Enter: a 3 s countdown, then 15 s of
# measuring res10 on the whole frame vs the head close-up (cutoffs 0.3 / 0.4 / 0.5).
# Tables are saved to recordings/face_test-<time>.md. See probe_test.sh for details.
exec "$(dirname "$0")/probe_test.sh" face "$@"

# Edge benchmarks (measured)

Measured numbers only, produced by `edge/tools/bench.py` on the Jetson. To refresh, run on the
Jetson with someone in view and paste the output below:

    cd ~/hearth-watch/edge && python3 tools/bench.py --seconds 20

## 2026-10-10 06:40 — end to end: the backend's real FallDetector alerts on a live clip

recordings/lie_down-20261010-063944.jsonl (lying across the mat, camera ~3.5 m away), recorded
with the full edge pipeline (rotated poseNet passes, keypoint hold, one-person tracking), checked
with `tools/check_rules.py --config recordings/config_framing.json`: Person 2's config with only
`floor_head_y_min` 0.40 and `floor_hip_y_min` 0.45 (the change requested for this camera).

| Measure | Result |
|---|---|
| Verdict (real FallDetector) | possible_fall at 3.2 s |
| Frames at floor level | 83 of 83 |
| Head / shoulders / hips / legs found | 100% / 100% / 100% / 99% |
| Torso deg | 69-76 |
| Head / hip height (median) | 0.62 / 0.75 |
| Observations per second | 8.3 |

With the unchanged config (0.65 line) the same clip does not alert: head height 0.59-0.65.
Caveat of the lowered config: the backend's upright rule shares `floor_hip_y_min`, so standing
(hips ~0.60 here) reads unknown and recovery can't be recorded until that rule gets its own bound.

## 2026-10-10 06:18 — lying across, camera ~3.5 m away: rotation fixes it (adopted)

edge/tools/pose_test.sh across, 39 frames, whole body visible in the camera image (confirmed by a
screenshot: plain poseNet drew head and shoulders only, nothing on hips or legs).

| Method | Person found | Keypoints (median) | Head+shoulders+hips | Torso deg (median) |
|---|---|---|---|---|
| as-is (threshold 0.15) | 62% | 4.5 | 0% | - |
| threshold 0.08 | 90% | 8 | 0% | - |
| threshold 0.04 | 100% | 10 | 15% | 83 |
| rotated CCW (head was on the right) | 100% | 13 | **100%** | **94** |
| rotated CW | 0% | - | 0% | - |

Lower thresholds do not recover the lying body; rotating the frame so the person looks upright
does. Decision: added to the pipeline (hearth_edge/pose_rotate.RotationPolicy), only when the
upright pose looks lying or incomplete, head side first, accepted only with >= 8 keypoints
including a torso and more keypoints than the upright pose. Cost on the Nano: ~95 ms per rotated
pass. Measured dry run with nobody in view: 13.0-13.1 FPS, ~10 observations/s (one alternating
try per second); standing costs nothing extra.

Speed-up (06:40): a lying-down frame was capture+upright 71 ms + CPU rotate 27 ms + rotated
poseNet 68 ms = ~5.1 FPS (recordings then held ~4.2 observations/s). Now the frame is shrunk to
half size on the GPU (cudaResize) before the CPU rotation (poseNet resizes to 224x224 anyway),
and while a rotated pose keeps being accepted the upright pass runs on 1 frame in 3. Measured:
rotated pass 75 ms, lying-down pattern **10.3 FPS** (was 5.1).

## 2026-10-10 05:52 — does rotating the frame help poseNet on lying poses? (not here; see 06:18)

edge/tools/pose_test.sh, camera repositioned to see a whole standing body, ~57 frames per row.
poseNet run on the frame as-is and rotated 90 degrees each way; "best of 3" keeps whichever
found the most keypoints. "All three" = head, neck/shoulders and hips in the same frame (what the
backend's posture rules need).

| Situation | Method | Person found | Keypoints (median) | All three | Torso deg (median) |
|---|---|---|---|---|---|
| standing | as-is | 100% | 17 | 100% | 5 |
| standing | best of 3 | 100% | 17 | 100% | 5 |
| lying across the picture | as-is | 100% | 14 | 96% | 62 |
| lying across the picture | best of 3 (CCW picked 38/57) | 100% | 15 | 96% | 69 |
| lying feet toward camera | as-is | 100% | 10 | 28% | 19 |
| lying feet toward camera | best of 3 (as-is picked 58/58) | 100% | 10 | 28% | 19 |

Rotation costs ~188 ms per frame (both directions) and, in an empty scene, produced partial
false "people" (2-3 keypoints) that plain poseNet did not. In THIS position plain poseNet already
worked, so rotation added little; the 06:18 run (camera farther away) showed the opposite.
Lying toward the camera fails because the legs hide the hips and the body is foreshortened
(torso reads ~19 degrees), which rotation cannot fix.

## 2026-10-10 04:08 — res10 face model (current default), person in view

Jetson Nano 2GB (JetPack 4.6.1), /dev/video0 640x480 MJPEG, resnet18-body + res10 (OpenCV DNN,
CPU), 12 s per mode after a 30-frame warm-up.

Memory in use (system, MB): before models 377, after poseNet 1322, after the face model 1342.

| Mode | FPS | Pose ms (mean / p95) | Face ms (mean) | Frame ms p95 | Peak memory MB | Frames with person | Face checks / faces found |
|---|---|---|---|---|---|---|---|
| pose only | 14.5 | 56.0 / 56.3 | - | 68.9 | 1366 | 175 of 175 | - |
| pose + face every 0.5 s | 9.4 | 56.0 / 56.3 | 183.5 | 255.8 | 1366 | 113 of 113 | 23 / 23 |
| pose + face every frame | 4.0 | 56.1 / 56.4 | 182.0 | 260.7 | 1366 | 48 of 48 | 48 / 48 |

Takeaways: res10 adds ~20 MB and ~180 ms per check on the CPU. It runs inside the camera loop,
so checking every 0.5 s lowers FPS from 14.5 to 9.4 (and observations from ~10/s to ~7/s).

## 2026-10-10 04:03 — face detector comparison (edge/tools/face_probe.py), person close up

Same live frames, 24 frames with a head in view, face ~356 px wide (close to the camera),
head-area brightness 106/255 (not backlit).

| Detector | ms per check | Found at >= 0.3 | Found at >= 0.5 | Found at >= 0.7 |
|---|---|---|---|---|
| facenet (jetson-inference, GPU) | 68 | 0% | 0% | 0% |
| res10 (OpenCV DNN, CPU) | 179 | 100% | 100% | 100% |
| haar (OpenCV cascade, CPU) | 51 | 0% (no score) | | |

Not yet measured: detection at the demo distance (2-3 m), where faces are much smaller.

## 2026-10-10 03:54 — facenet (previous default), nobody in view

Valid for speed and memory only: no one was in front of the camera. The comparison above later
showed facenet missing a clearly visible face in every frame, which is why res10 replaced it.

Jetson Nano 2GB (JetPack 4.6.1), /dev/video0 640x480 MJPEG, resnet18-body + facenet, 8 s per
mode after a 30-frame warm-up.

Memory in use (system, MB): before models 374, after poseNet 1321, after facenet 1417.

| Mode | FPS | Pose ms (mean / p95) | Face ms (mean) | Frame ms p95 | Peak memory MB | Frames with person | Face checks / faces found |
|---|---|---|---|---|---|---|---|
| pose only | 14.6 | 56.0 / 56.2 | - | 68.3 | 1426 | 0 of 117 | - |
| pose + face every 0.5 s | 13.1 | 56.0 / 56.3 | 48.4 | 118.3 | 1426 | 0 of 106 | 16 / 0 |
| pose + face every frame | 8.4 | 56.0 / 56.3 | 48.4 | 118.8 | 1427 | 7 of 68 | 68 / 0 |

Takeaways: facenet adds ~96 MB and ~48 ms per check. Checking every 0.5 s (the default) costs
about 1.5 FPS; checking every frame would cost about 6 FPS.

# Edge benchmarks (measured)

Measured numbers only, produced by `edge/tools/bench.py` on the Jetson. To refresh, run on the
Jetson with someone in view and paste the output below:

    cd ~/hearth-watch/edge && python3 tools/bench.py --seconds 20

## 2026-10-10 22:45 — our posture model (default) on the Nano

`posture_mlp.py` (numpy, 51 -> 64 -> 64 -> 3): **0.56 ms** per person per frame (500 calls, Jetson
CPU). Dry run with face detection and the posture model on, nobody in view: **13.3-13.4 FPS**
(`run_edge.py --dry-run`, the same as without the model). A rotated poseNet pass for someone the
edge isn't tracking costs one rotated-pass time (~75 ms, see the rotation entry below) at most
every 0.5 s while someone else looks lying, every 1 s otherwise; FPS with two people in view has
not been measured yet.

## 2026-10-10 13:42 — Hugging Face fall model (YOLO11n) built with TensorRT 8.2 on the Nano

`melihuzunoglu/human-fall-detection` @ `97261b0` exported to ONNX (edge/pc/03_export_onnx.py:
opset 12, static, no NMS; 15 operation types, all supported by TensorRT 8.2; PyTorch vs ONNX
decisions agree at every position), then on the Jetson:
`trtexec --onnx=best_416.onnx --fp16 --workspace=512`.

| Measure | 416 x 416, FP16 | 640 x 640, FP16 |
|---|---|---|
| Build | PASSED (~9 min on the Nano) | PASSED (~13 min) |
| GPU compute per image | 24.7 ms median, 27.0 ms max | 49.7 ms median, 256 ms max (one spike) |
| Throughput (model alone) | 40 images/s | 18 images/s |
| Engine size | 10 MB | 9.7 MB |

416 is the default (`FALL_MODEL_SIZE` in config.py): half the GPU time, and the edge only asks
for ~3 checks per second. From Python (fall_model.py: letterbox + engine + decode + rules) one
416 check took ~47 ms on bus.jpg, giving the same two "standing" boxes as PyTorch and ONNX Runtime.

### All three together: poseNet + res10 face + fall model (`run_edge.py --fall-model --dry-run`)

45 s dry run, nobody in view (speed and memory only), engine from `edge/tools/get_fall_model.sh`
(Hugging Face ONNX, sha256 checked, 24.1 ms median in trtexec):

| Measure | Without the fall model (above) | With `--fall-model` |
|---|---|---|
| FPS | 13.0-13.1 | 12.3-12.9 |
| Fall model checks | - | 3 per second (every 0.3 s), 131-143 ms each beside poseNet + res10 (46 ms alone) |
| Face checks | ~210 ms each | 209-219 ms each (unchanged) |
| System memory in use | ~1340 MB | 1633 MB peak, ~250 MB still available (+5 GB swap) |
| Observations with `posture_prediction` | - | 281 of 300 (all after the first result, ~1 s in) |

The first attempt crashed on its first check: the buffers used CUDA managed memory, which the
Nano's GPU forbids the CPU from touching while other GPU work runs (concurrentManagedAccess = 0).
fall_model.py now uses cudaMalloc + cudaMemcpy.

Not yet measured: accuracy on our camera (standing / sitting / lying across / lying toward), and
FPS in the lying-down pattern with rotated poseNet passes.

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

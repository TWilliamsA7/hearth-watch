# Edge benchmarks (measured)

Measured numbers only, produced by `edge/tools/bench.py` on the Jetson. To refresh, run on the
Jetson with someone in view and paste the output below:

    cd ~/hearth-watch/edge && python3 tools/bench.py --seconds 20

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

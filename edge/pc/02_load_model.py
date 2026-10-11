"""Step 2: the Hugging Face fall model inside PyTorch, on your GPU. Run on the Windows PC:

    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\02_load_model.py

Reuses Person 2's code (backend/hearth/perception/fall_model.py) for the parts they already
built: the download pinned to an exact Hugging Face revision, the SHA-256 check and the label
check. Same file location as their runner (models/human-fall-detection/best.pt, git-ignored).

What it shows:
  1. Download + verify the pinned checkpoint (Person 2's download_weights / load_model).
  2. Inside the model: it is an ordinary PyTorch nn.Module - layers and parameter count.
  3. The raw output tensor: what the network really produces before any post-processing.
     (The Jetson runner in step 5 has to decode exactly this.)
  4. Ultralytics' high-level predict() on a sample photo, and the posture_prediction that
     Person 2's rules turn it into - the message the backend would receive.
  5. GPU vs CPU speed for the whole model.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hearth.perception.fall_model import (ModelSettings, download_weights, load_model,
                                          observation_from_result)

WEIGHTS = Path("models/human-fall-detection/best.pt")   # same place as Person 2's runner
RUNTIME = Path(".hearth-runtime")                         # Ultralytics settings/cache (git-ignored)

print("=" * 70)
print("1. Download and verify the pinned model (Person 2's code)")
settings = ModelSettings.load()                           # reads backend/hearth/perception/model.json
print("   repo:", settings.repo_id, "@", settings.revision[:12])
path = download_weights(WEIGHTS, settings)                # downloads once, then only re-verifies SHA-256
print("   weights: %s (%.1f MB), SHA-256 matches the pinned value" % (path, path.stat().st_size / 1e6))
yolo = load_model(path, settings, RUNTIME)                # also checks labels are standing/sitting/fallen
print("   classes:", yolo.names)

print("=" * 70)
print("2. Inside: an ordinary PyTorch nn.Module")
net = yolo.model                                          # the torch.nn.Module Ultralytics wraps
print("   type:", type(net).__name__, "| is nn.Module:", isinstance(net, torch.nn.Module))
n_params = sum(p.numel() for p in net.parameters())
print("   parameters: %s (%.1f million numbers learned in training)" % (f"{n_params:,}", n_params / 1e6))
layers = list(net.model.children()) if hasattr(net, "model") else list(net.children())
print("   top-level layers: %d. First and last few:" % len(layers))
for i, layer in enumerate(layers):
    if i < 3 or i >= len(layers) - 2:
        print("     [%2d] %s" % (i, type(layer).__name__))
    elif i == 3:
        print("     ...")

print("=" * 70)
print("3. The raw output tensor")
net = net.to("cuda").eval()                               # eval(): inference mode (no dropout/batch-norm updates)
x = torch.zeros(1, 3, settings.image_size, settings.image_size, device="cuda")  # 1 image, RGB, 640x640
with torch.no_grad():                                     # no gradients needed: we are not training
    out = net(x)
raw = out[0] if isinstance(out, (tuple, list)) else out
print("   input shape:  %s  (batch, colour channels, height, width)" % (tuple(x.shape),))
print("   output shape: %s" % (tuple(raw.shape),))
if raw.dim() == 3 and raw.shape[1] == 4 + len(yolo.names):
    print("   -> for each of %d candidate positions: 4 box numbers (centre x, centre y, width,"
          " height)" % raw.shape[2])
    print("      + %d class scores (%s)." % (len(yolo.names), ", ".join(yolo.names.values())))
    print("      %d = 80x80 + 40x40 + 20x20 grid cells at three scales." % raw.shape[2])
    print("      Turning this into boxes needs: best class per position, a score cut-off, and")
    print("      non-maximum suppression (dropping overlapping duplicates) - step 5 does that.")
else:
    print("   (layout differs from the usual YOLO11 [1, 4+classes, 8400]; step 5 will adapt to it)")

print("=" * 70)
print("4. High-level: predict() on a sample photo, then Person 2's rules")
from ultralytics.utils import ASSETS  # noqa: E402  (sample images that ship with Ultralytics)
image = ASSETS / "bus.jpg"
result = yolo.predict(str(image), device=0, imgsz=settings.image_size,
                      conf=settings.detection_confidence_min, verbose=False)[0]
print("   image:", image.name, "| size %dx%d" % (result.orig_shape[1], result.orig_shape[0]))
for xyxy, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
    print("     %-9s %.2f  box (%d, %d) - (%d, %d)" % (yolo.names[int(cls)], conf, *map(int, xyxy)))
annotated = Path("models/step2_bus_annotated.jpg")
import cv2  # noqa: E402
cv2.imwrite(str(annotated), result.plot())
print("   annotated copy saved to", annotated, "(open it to see the boxes)")
obs = observation_from_result(result, timestamp=datetime.now(timezone.utc), source="pc-step2",
                              source_type="replay", room_id="room-01", run_id="pc-step2",
                              settings=settings)
p = obs.posture_prediction
print("   posture_prediction the backend would get: label=%s, detections=%d, reason=%s"
      % (p.label, p.detection_count, p.uncertainty_reason))
print("   (Several people -> 'unknown' on purpose: Person 2's rule never picks one person as 'safe'.)")

print("=" * 70)
print("5. Speed of the whole model: GPU vs CPU")
for device, runs in ((0, 20), ("cpu", 3)):
    yolo.predict(str(image), device=device, imgsz=settings.image_size, verbose=False)  # warm-up
    start = time.perf_counter()
    for _ in range(runs):
        yolo.predict(str(image), device=device, imgsz=settings.image_size, verbose=False)
    ms = (time.perf_counter() - start) / runs * 1000
    print("   %-4s %6.1f ms per image (%.0f FPS)" % ("GPU" if device == 0 else "CPU", ms, 1000 / ms))
print("=" * 70)
print("Step 2 done. Next: step 3 exports this network to ONNX for the Jetson (edge/pc/README.md).")

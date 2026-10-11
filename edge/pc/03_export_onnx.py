"""Step 3: export the fall model to ONNX for the Jetson, and prove the export is faithful.

    .\\.venv-pc\\Scripts\\python.exe -m pip install onnx onnxslim onnxruntime     (once)
    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\03_export_onnx.py

The Jetson can't run PyTorch, so the network is saved as ONNX: a portable file describing every
operation and weight, which the Jetson's TensorRT 8.2 rebuilds into a GPU engine (step 5).

Choices, all for TensorRT 8.2 (JetPack 4.6):
  * opset 12   - the ONNX operator-set version; TensorRT 8.2 officially supports up to 13
  * static     - fixed input size (no dynamic shapes)
  * nms=False  - no built-in duplicate removal; the Jetson code does it (TRT 8.2 lacks the op)
  * two sizes  - 640 (as trained) and 416 (about 2.4x fewer pixels, faster on the Nano)

Then the learning part: the same real photo goes through PyTorch and through ONNX Runtime and
the outputs are compared number by number. Files go to models/human-fall-detection/ (git-ignored):
best_640.onnx, best_416.onnx and export_info.json (class order, sizes, checksums).
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from hearth.perception.fall_model import ModelSettings, download_weights, load_model

WEIGHTS = Path("models/human-fall-detection/best.pt")
RUNTIME = Path(".hearth-runtime")
SIZES = (640, 416)
OPSET = 12
# Operations TensorRT 8.2's ONNX parser handles (the ones YOLO-style exports normally use).
TRT82_OK = {"Add", "Concat", "Constant", "Conv", "Div", "Gather", "MatMul", "MaxPool", "Mul",
            "Reshape", "Resize", "Shape", "Sigmoid", "Slice", "Softmax", "Split", "Sub",
            "Transpose", "Unsqueeze", "Squeeze", "Cast", "Pow", "Sqrt", "ReduceMean",
            "Expand", "Equal", "Where", "Range", "ConstantOfShape", "Identity", "Flatten",
            "Relu", "LeakyRelu", "Tanh", "Exp", "Log", "Neg", "Floor", "Pad", "BatchNormalization",
            "GlobalAveragePool", "AveragePool", "ArgMax", "TopK", "Clip", "HardSigmoid", "Erf"}


def letterbox(image_bgr, size):
    """Resize keeping the aspect ratio and pad to size x size with grey - what YOLO expects."""
    import cv2
    h, w = image_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0      # BGR -> RGB, 0..1
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]   # HWC -> 1 x C x H x W


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


settings = ModelSettings.load()
path = download_weights(WEIGHTS, settings)
info = {"source": {"repo_id": settings.repo_id, "revision": settings.revision,
                   "filename": settings.filename, "sha256": settings.sha256},
        "license": "AGPL-3.0 (declared by the model author); this export keeps it",
        "opset": OPSET, "exports": []}

import cv2  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from ultralytics.utils import ASSETS  # noqa: E402

photo = cv2.imread(str(ASSETS / "bus.jpg"))
for size in SIZES:
    print("=" * 70)
    print("Export at %dx%d" % (size, size))
    yolo = load_model(path, settings, RUNTIME)           # fresh copy per export
    info["classes"] = {int(k): v for k, v in yolo.names.items()}
    exported = Path(yolo.export(format="onnx", imgsz=size, opset=OPSET, simplify=True,
                                dynamic=False, nms=False, device="cpu", verbose=False))
    target = WEIGHTS.with_name("best_%d.onnx" % size)
    shutil.move(str(exported), target)
    model = onnx.load(str(target))
    onnx.checker.check_model(model)                      # structural validity
    inp, out = model.graph.input[0], model.graph.output[0]
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    ops = sorted({n.op_type for n in model.graph.node})
    print("   file: %s (%.1f MB), opset %d, %d operations"
          % (target, target.stat().st_size / 1e6, model.opset_import[0].version, len(model.graph.node)))
    print("   input  %-8s %s" % (inp.name, dims(inp)))
    print("   output %-8s %s" % (out.name, dims(out)))
    unknown = [op for op in ops if op not in TRT82_OK]
    print("   operation types:", ", ".join(ops))
    print("   not on the TensorRT 8.2 list:", ", ".join(unknown) if unknown else "none - good")

    # --- Is ONNX faithful? Same real photo through PyTorch and ONNX Runtime -------------
    x = letterbox(photo, size)
    net = yolo.model.float().cpu().eval()
    with torch.no_grad():
        ref = net(torch.from_numpy(x))
    ref = (ref[0] if isinstance(ref, (tuple, list)) else ref).numpy()
    sess = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    got = sess.run(None, {inp.name: x})[0]
    diff = np.abs(ref - got)
    best_ref = ref[0, 4:].max(axis=0)                     # best class score per position
    agree = int(((best_ref > 0.25) == (got[0, 4:].max(axis=0) > 0.25)).sum())
    print("   PyTorch vs ONNX on bus.jpg: max difference %.2e, mean %.2e" % (diff.max(), diff.mean()))
    print("   positions above the 0.25 score cut-off: PyTorch %d, ONNX %d; agreement %d/%d"
          % ((best_ref > 0.25).sum(), (got[0, 4:].max(axis=0) > 0.25).sum(), agree, best_ref.size))
    print("   (box numbers are in pixels up to %d, so differences around 1e-3 are float rounding;"
          " what must match is the decisions)" % size)
    info["exports"].append({"file": target.name, "imgsz": size, "sha256": sha256(target),
                            "input": [inp.name, dims(inp)], "output": [out.name, dims(out)],
                            "ops": ops, "max_abs_diff_vs_pytorch": float(diff.max())})

Path(WEIGHTS.with_name("export_info.json")).write_text(json.dumps(info, indent=2))
print("=" * 70)
print("Saved %s. Next: step 4 publishes the ONNX files to your Hugging Face account so the"
      % WEIGHTS.with_name("export_info.json"))
print("Jetson can download them (edge/pc/README.md).")

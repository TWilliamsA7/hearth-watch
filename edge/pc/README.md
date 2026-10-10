# PC (RTX 3060 Ti): PyTorch + the Hugging Face fall model

Goal: learn PyTorch while preparing Person 2's Hugging Face fall model
([melihuzunoglu/human-fall-detection](https://huggingface.co/melihuzunoglu/human-fall-detection),
YOLO11, classes Standing / Sitting / Fallen) to run on the Jetson. Everything here runs on the
Windows PC (Python 3.12) in PowerShell, from the repo folder.

**Builds on what Person 2 already did (on main), does not redo it:**
`backend/hearth/perception/fall_model.py` + `model.json` already download the model pinned to a
revision, verify its SHA-256, check its labels, and map detections to `posture_prediction`
(0.25 minimum, several people -> `unknown`); `edge/run_fall_model.py` runs it on a laptop CPU. The
steps below reuse that code. What nobody has done yet: ONNX export, TensorRT, running it on the
Jetson. The hand-gesture model and the contrast retry stay Person 2's laptop features.

| Step | Script | You learn | Project result |
|---|---|---|---|
| 1 | `01_check_gpu.py` | tensors, devices, GPU speed, autograd | PyTorch + CUDA working |
| 2 | `02_load_model.py` (next) | a Hugging Face model in PyTorch: layers, parameters, raw output tensor; GPU vs CPU | the pinned model via Person 2's `download_weights` / `load_model`, on the GPU |
| 3 | `03_export_onnx.py` (next) | exporting, checking PyTorch vs ONNX numerically | an ONNX file the Jetson's TensorRT can build |
| 4 | (next) | publishing a model on the Hugging Face Hub | the Jetson downloads it from your HF account |
| 5 | on the Jetson | TensorRT engine + runner | `posture_prediction` in Jetson observations, same rules as Person 2's `prediction_from_detections` (tested against it), off by default until measured |
| 6 | (optional) | datasets, training loop, evaluation | a fine-tuned model if step 5 shows weak spots |

## One-time setup

1. **Check the NVIDIA driver.** In PowerShell:
   ```powershell
   nvidia-smi
   ```
   The top right says `CUDA Version: 12.x` (or 13.x). That is the newest CUDA the driver supports.

2. **Get the repo** (public, no login needed):
   ```powershell
   cd $HOME
   git clone https://github.com/TWilliamsA7/hearth-watch.git
   cd hearth-watch
   git switch feat/pytorch
   ```
   Later, to get new steps: `git pull` in that folder.

3. **A separate Python environment for the PC** (named `.venv-pc` so it never mixes with the
   laptop's `.venv`). Calling its `python.exe` directly avoids PowerShell's script-activation
   restrictions:
   ```powershell
   py -3.12 -m venv .venv-pc
   .\.venv-pc\Scripts\python.exe -m pip install --upgrade pip
   ```

4. **PyTorch with CUDA.** The CPU-only build is the default on Windows, so use PyTorch's CUDA
   package index. `cu126` works with any driver reporting CUDA 12.6 or newer:
   ```powershell
   .\.venv-pc\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
   (If `nvidia-smi` shows an older CUDA than 12.6, use the matching index from
   pytorch.org > Get Started, e.g. `cu118`.) It is a large download (~2.5 GB).

5. **Person 2's perception code + Ultralytics** (needed from step 2 on; install it after the CUDA
   torch so pip keeps the GPU build):
   ```powershell
   .\.venv-pc\Scripts\python.exe -m pip install -e .\shared\python -e ".\backend[vision]"
   ```

## Step 1

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\01_check_gpu.py
```

Expected: `CUDA available: True`, `NVIDIA GeForce RTX 3060 Ti`, and the GPU matrix multiply
tens to hundreds of times faster than the CPU. Paste the output back for the next step.

## Licenses

The fall model is AGPL-3.0 (declared by its author); anything derived from it (the ONNX export,
a fine-tuned model) keeps that license and attribution. PyTorch is BSD-style.

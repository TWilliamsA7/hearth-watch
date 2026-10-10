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
| 2 | `02_load_model.py` | a Hugging Face model in PyTorch: layers, parameters, raw output tensor; GPU vs CPU | the pinned model via Person 2's `download_weights` / `load_model`, on the GPU |
| 3 | `03_export_onnx.py` | exporting, checking PyTorch vs ONNX numerically | an ONNX file the Jetson's TensorRT can build |
| 4 | `04_push_to_hub.py` | Hub repos, commits and pinning, model cards, licenses | the Jetson downloads it from your HF account, pinned to a commit |
| 5 | on the Jetson | TensorRT engine + runner | `posture_prediction` in Jetson observations (`run_edge.py --fall-model`), same rules as Person 2's `prediction_from_detections` (tested against it); measured in edge/BENCHMARKS.md |
| 6 | public datasets, then `05_...` onward | datasets, labels, train/test splits, a training loop written by hand, evaluation | our own posture classifier from keypoints, trained on public fall videos, tested on our own recordings |

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
   python -m venv .venv-pc
   .\.venv-pc\Scripts\python.exe -m pip install --upgrade pip
   ```
   (`python -m venv`, not `py -3.12`: the Microsoft Store Python usually has no `py` launcher.)

4. **PyTorch with CUDA.** Plain `pip install torch` on Windows gives the CPU-only build
   (`torch.__version__` ends in `+cpu` and the GPU is invisible), so use PyTorch's CUDA index.
   `cu126` worked on this PC (torch 2.14.1+cu126, driver showing CUDA 12.6+ in `nvidia-smi`):
   ```powershell
   .\.venv-pc\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   .\.venv-pc\Scripts\python.exe -m pip install numpy
   .\.venv-pc\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   The last line must print `...+cu126 True`. It is a large download (~2.5 GB).

5. **Person 2's perception code + Ultralytics** (needed from step 2 on; install it after the CUDA
   torch so pip keeps the GPU build):
   ```powershell
   .\.venv-pc\Scripts\python.exe -m pip install -e .\shared\python -e ".\backend[vision]"
   ```

## Step 1

Run it from a terminal (VS Code: Terminal > New Terminal) with the environment's python:

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\01_check_gpu.py
```

Not with VS Code's Run button: that button (Code Runner extension) uses the global Python,
which has no CUDA PyTorch.

Expected: `CUDA available: True`, `NVIDIA GeForce RTX 3060 Ti`, and the GPU matrix multiply
tens to hundreds of times faster than the CPU. Paste the output back for the next step.

## Step 2

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\02_load_model.py
```

Downloads the pinned model once (~5.5 MB, with Person 2's verified downloader), then shows its
layers and parameter count, the raw output tensor shape, detections on a sample photo (an
annotated copy is saved to `models/step2_bus_annotated.jpg`), the `posture_prediction` Person 2's
rules make of it, and GPU vs CPU speed. Paste the output back for step 3.

## Step 3

```powershell
.\.venv-pc\Scripts\python.exe -m pip install onnx onnxslim onnxruntime
.\.venv-pc\Scripts\python.exe edge\pc\03_export_onnx.py
```

Exports the model to ONNX at 640 and 416 (opset 12, fixed size, no built-in NMS: what the
Jetson's TensorRT 8.2 can build), lists the operation types against the TensorRT 8.2 list, and
runs the same photo through PyTorch and ONNX Runtime to compare the outputs number by number.
Writes `models/human-fall-detection/best_640.onnx`, `best_416.onnx`, `export_info.json`.
Paste the output back for step 4.

## Step 4

Needs a free Hugging Face account and a token with **write** access
(https://huggingface.co/settings/tokens):

```powershell
.\.venv-pc\Scripts\python.exe -m pip install huggingface_hub
.\.venv-pc\Scripts\hf.exe auth login
.\.venv-pc\Scripts\python.exe edge\pc\04_push_to_hub.py
```

Creates `<your-username>/hearth-fall-yolo11n-onnx` (public; `--private` to keep it private),
uploads both ONNX files, `export_info.json` and a model card (base model + exact revision,
AGPL-3.0, export settings, class order, the PyTorch-vs-ONNX check, intended use and limits) as
one commit, and prints the commit id. Paste it back: the Jetson pins its download to it.

## Step 6: train our own posture model on public data

The goal: a small network that reads the keypoints the Jetson already sends (no video) and says
**upright / sitting / on the floor**. Public datasets give us the training data; our own
recordings (`recordings/*.jsonl`, ~1,450 frames) are the **test set**, so the score says whether
it works on our camera in our room.

No public dataset ships 2D keypoints in our format (the skeleton ones are 3D Kinect data). So we
take labelled **videos**, run a pose model on the 3060 Ti to get COCO keypoints (the same body
points poseNet reports; poseNet's extra `neck` is the midpoint of the shoulders), and train on
those.

### The data

Labels come from [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall), which
re-labelled 8 public fall datasets with timed segments ("video X, 2.1-5.3 s: sitting"). Classes:
0 walk, 1 fall, 2 fallen, 3 sit_down, 4 sitting, 5 lie_down, 6 lying, 7 stand_up, 8 standing,
9 other. We use walk + standing -> upright, sitting -> sitting, fallen + lying -> on the floor,
and skip the transitions (fall, sit_down, lie_down, stand_up) and other. The videos come from the
original datasets:

| Dataset | What it is | Labelled time (upright / sitting / floor) | License |
|---|---|---|---|
| [GMDCSA24](https://doi.org/10.5281/zenodo.13354453) | 4 actors, 3 real homes, day and night, cheap 0.92 MP webcam (closest to our setup), 160 videos | ~3 / ~4 / ~4 min | CC BY 4.0 |
| [Le2i](https://search-data.ubfc.fr/FR-13002091000019-2024-04-09) | 191 videos, one wall-mounted camera, home / office / coffee room / lecture room, 320x240 | ~17 / ~6 / ~5.5 min | CC BY-NC-SA |

Not used for now: OmniFall's synthetic set (12,000 AI-generated clips, ~9.1 GB, `pip install
omnifall`), useful later if we need more data.

### Download (PowerShell on the PC)

Everything goes in `C:\hearth-data`, **outside the repo**, so no video can be committed by accident.

1. Labels (a few seconds):

   ```powershell
   mkdir C:\hearth-data\labels
   cd C:\hearth-data
   curl.exe -L -o labels\GMDCSA24.csv https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/labels/GMDCSA24.csv
   curl.exe -L -o labels\le2i.csv https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/labels/le2i.csv
   ```

2. GMDCSA24 (1.1 GB):

   ```powershell
   curl.exe -L -o gmdcsa24.zip "https://zenodo.org/records/13354453/files/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1.zip?download=1"
   (Get-FileHash gmdcsa24.zip -Algorithm MD5).Hash
   mkdir gmdcsa24; tar -xf gmdcsa24.zip -C gmdcsa24
   ```

   The hash must print `3D36F2C5C1A666B99639E4E9FD843EFB`; if not, the download was cut short,
   run the `curl.exe` line again.

3. Le2i: its site blocks scripts (bot check), so use a browser on the PC: open
   https://search-data.ubfc.fr/FR-13002091000019-2024-04-09, download the dataset, and extract
   everything into `C:\hearth-data\le2i` (room folders such as `Coffee_room_01`, `Home_01`).
   If it landed under another name, rename the folder:
   `Rename-Item C:\hearth-data\<folder> le2i`.

4. Check what landed:

   ```powershell
   cd C:\hearth-data
   tree /F gmdcsa24 | Select-Object -First 25
   tree /F le2i | Select-Object -First 25
   (Get-ChildItem -Recurse -Include *.mp4,*.avi | Measure-Object).Count
   ```

   Expect **350** videos (160 GMDCSA24 + 190 Le2i). Also list Le2i's folders and a few file
   names, so the next script can match them:

   ```powershell
   Get-ChildItem C:\hearth-data\le2i -Directory -Recurse | Select-Object -ExpandProperty FullName
   Get-ChildItem C:\hearth-data\le2i -Recurse -Include *.avi | Select-Object -First 5 -ExpandProperty Name
   ```

What we got (2026-10-10): 350 videos, GMDCSA24 MD5 matched. The real paths differ from the
names in OmniFall's labels, so the next script has to translate them:

| Dataset | OmniFall label path | Real file |
|---|---|---|
| GMDCSA24 | `Subject_1/ADL/01` | `gmdcsa24\ekramalam-GMDCSA24-...-5abac76\Subject 1\ADL\01.mp4` (space, not underscore; extra top folder) |
| Le2i | `Coffee_room_02/video_14` | `le2i\FallDataset\Coffee_room_02\Coffee_room_02\Videos\...` (two extra folder levels; file names to confirm, often `video (14).avi`) |

### Step 6a: `05_extract_keypoints.py`

```powershell
cd C:\Users\Gaston\hearth-watch     # the repo folder (the download steps left you in C:\hearth-data)
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py --check      # match labels to videos only (seconds)
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py --limit 5    # quick trial on 5 videos
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py              # everything
```

`--check` should end with "every label has its video" and about 335 matched videos (upright
~1,200 s, sitting ~610 s, floor ~575 s of labelled time). The first real run downloads the pose
weights (`yolo11m-pose.pt`, ~40 MB) into `C:\hearth-data`, never into the repo. Other folders:
`--data`, `--le2i`, `--gmdcsa`.

Runs on the 3060 Ti and turns labelled video into a small file of numbers:

1. Reads both label CSVs and keeps only the segments we use: walk / standing -> **upright**,
   sitting -> **sitting**, fallen / lying -> **floor** (transitions and "other" are skipped).
2. Finds each segment's video using the table above. Any label it cannot match is **listed
   loudly**, never silently skipped: a label paired with the wrong video would ruin training
   with no error.
3. Opens the video, jumps to the segment's start time, and runs a pose model on every 2nd frame
   until its end time.
4. For each frame keeps the 17 COCO keypoints (x, y, confidence, normalized to 0-1 like the
   Jetson's), the label, and which video and person (subject) it came from, so train/test can
   later be split by person instead of by frame.
5. Writes everything to `C:\hearth-data\keypoints.npz` (numbers only, a few MB): the input of
   the training script.

Pose model: **YOLO11-pose** (Ultralytics, already in `.venv-pc` because Person 2's code uses it).
It finds the same 17 body points as the Jetson's poseNet; poseNet's extra `neck` is the midpoint
of the shoulders, computed the same way on both sides. Expected to get through all 350 videos in
a few minutes on the GPU.

The catch: YOLO11-pose is a better pose model than poseNet and finds joints poseNet misses.
Training randomly hides keypoints to make up for it, and the test set is our own recordings
(real poseNet output), so the test score shows how much the difference hurts. Plan B if it
hurts a lot: run the extraction on the Jetson with poseNet itself (identical keypoints to live,
about an hour for all videos).

### Step 6b onward

- A hand-written PyTorch `Dataset`, network (`nn.Module`) and training loop on CUDA; test
  accuracy and a confusion table on our recordings.
- Export to ONNX; a network this small runs on the Jetson with plain numpy in under 1 ms.

Both datasets are research data: credit them, don't build a product on them (Le2i is
non-commercial).

## Licenses

The fall model is AGPL-3.0 (declared by its author); anything derived from it (the ONNX export,
a fine-tuned model) keeps that license and attribution. PyTorch is BSD-style.
Step 6 data: OmniFall labels CC BY-NC-SA 4.0; GMDCSA24 CC BY 4.0 (Alam et al., Data in Brief
2024); Le2i CC BY-NC-SA (Charfi et al., Laboratoire Le2i / UBFC). The model we train from them
is ours, but must credit the datasets and stays non-commercial because of Le2i and the OmniFall
labels. Ultralytics YOLO11-pose (AGPL-3.0) is used only as a tool to produce the training
keypoints; the classifier we train contains none of its code or weights.

# PC (RTX 3060 Ti): PyTorch, the Hugging Face fall model, and our own posture model

Everything here runs on the Windows PC (Python 3.12, PowerShell, from the repo folder
`C:\Users\Gaston\hearth-watch`, data in `D:\hearth-data`). Two parts:

1. **Steps 1-5:** Person 2's Hugging Face fall model (YOLO11, standing / sitting / fallen) made to
   run on the Jetson: load it in PyTorch, export to ONNX, publish it, TensorRT on the Nano.
2. **Steps 6-7: our own posture model**, trained here and now the model the system uses by
   default. How it works on the Jetson and with the backend: [`../POSTURE_MODEL.md`](../POSTURE_MODEL.md).
   Results (locked test, field test, false positive rates): [`RESULTS.md`](RESULTS.md).

## Our posture model: the training pipeline

```
5 public fall datasets (videos) + OmniFall's timed labels
  -> download_datasets.py / manual (GMDCSA24, Le2i)       videos in D:\hearth-data
  -> 05_extract_keypoints.py  YOLO11m-pose, every 2nd frame   keypoints.npz (numbers only)
  -> 06_train.py              locked 60/20/20 split by person, MLP 51-64-64-3, live chart
                              --final-test: the locked 20%, once            model_3class.pt
  -> 08_export_weights.py     numpy copy, checked frame-for-frame against PyTorch
                                                         edge/hearth_edge/posture_mlp.npz
  -> commit -> Jetson: run_edge.py uses it by default (../POSTURE_MODEL.md)
```

| Script | What it does | Output |
|---|---|---|
| `download_datasets.py` | OCCU (10 videos out of a 10.8 GB zip, by range requests), CAUCAFall (the 100 `.avi` files via Mendeley's API), OF-Synthetic (streams the 9.7 GB tar, keeps 2,203 care-room clips); checksums; re-runnable | videos + `labels\*.csv` in `D:\hearth-data` |
| `05_extract_keypoints.py` | matches every OmniFall label to its video (stops loudly on a mismatch), runs YOLO11m-pose on labelled stretches, keeps the most confident person per frame with its original label, person and frame size | `keypoints.npz` |
| `06_train.py` | features (51 numbers per frame), locked split (`split.json`), augmentation (hidden keypoints, mirroring, jitter), class-weighted training with live progress, best epoch by balanced validation accuracy, cut-off table, field test on our recordings; `--final-test` for the locked test (logged) | `model_3class.pt`, `..._curves.png`, `test_log.txt` |
| `07_finetune_closeups.py` | experiment: teaches "unsure" for close-ups by fine-tuning (kept as a record; not used, see "Decision" below) | `model_closeup*.pt` |
| `08_export_weights.py` | writes the weights + settings for the Jetson, refuses unless numpy and the Jetson module give the same decisions as PyTorch | `edge/hearth_edge/posture_mlp.npz` |

Re-run everything (after the one-time setup and the GMDCSA24 / Le2i downloads in Step 6):

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\download_datasets.py
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --epochs 15 --classes 3 --out D:\hearth-data\model_3class.pt
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --test-only --out D:\hearth-data\model_3class.pt --min-conf 0.5
.\.venv-pc\Scripts\python.exe edge\pc\08_export_weights.py
```

Then commit `edge/hearth_edge/posture_mlp.npz`. `--final-test` is for a finished model only (it
logs every run). The sections below are the development log, in the order things were done:
what was tried, measured and decided, with the commands used at each step.

## Development log

Builds on what Person 2 already did (on main), does not redo it:
`backend/hearth/perception/fall_model.py` + `model.json` already download the HF model pinned to a
revision, verify its SHA-256, check its labels, and map detections to `posture_prediction`
(0.25 minimum, several people -> `unknown`); `edge/run_fall_model.py` runs it on a laptop CPU.

| Step | Script | You learn | Project result |
|---|---|---|---|
| 1 | `01_check_gpu.py` | tensors, devices, GPU speed, autograd | PyTorch + CUDA working |
| 2 | `02_load_model.py` | a Hugging Face model in PyTorch: layers, parameters, raw output tensor; GPU vs CPU | the pinned model via Person 2's `download_weights` / `load_model`, on the GPU |
| 3 | `03_export_onnx.py` | exporting, checking PyTorch vs ONNX numerically | an ONNX file the Jetson's TensorRT can build |
| 4 | `04_push_to_hub.py` | Hub repos, commits and pinning, model cards, licenses | the Jetson downloads it from your HF account, pinned to a commit |
| 5 | on the Jetson | TensorRT engine + runner | HF `posture_prediction` from the Jetson (`run_edge.py --fall-model`); measured in edge/BENCHMARKS.md |
| 6 | datasets, `05`, `06`, `07` | datasets, labels, splits by person, a training loop written by hand, evaluation, fine-tuning | our own posture model: locked test 97.7% right, floor caught 96.1% |
| 7 | `08`, `edge/hearth_edge/posture_mlp.py` | exporting without PyTorch, checking an exact copy | the model on the Jetson, used by the backend by default |

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

### More fall data: OCCU, CAUCAFall, OF-Synthetic (on D:)

Floor (fallen / lying) footage was the thinnest class (~575 s). These add ~3,200 s more, all with
OmniFall labels:

| Dataset | Adds (upright / sitting / floor) | What it brings | Download size | License |
|---|---|---|---|---|
| [OCCU](https://doi.org/10.5281/zenodo.15494102) | 289 / 91 / 496 s, 5 people, 2 views | falls whose end is hidden behind a bed | 16 MB (10 videos out of a 10.8 GB zip) | CC BY 4.0 |
| [CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky/4) | 225 / 65 / 263 s, 10 people | a real home, varied lighting | 135 MB (100 videos; the full 10 GB is still images) | CC BY 4.0 |
| [OF-Synthetic](https://huggingface.co/datasets/simplexsigil2/omnifall), care-room filter | 2,918 / 1,691 / 2,436 s, 2,203 clips (747 elderly) | AI-generated people: indoor, adults and elderly, camera at eye height / high / top | streams 9.7 GB once, keeps ~1.2 GB | CC BY-NC 4.0 |

Everything lives in `D:\hearth-data` from now on. Move the existing folder first (once), then
download (re-runnable: finished files are skipped; checksums are verified):

```powershell
Remove-Item C:\hearth-data\gmdcsa24.zip        # optional: the extracted copy is all we need
Move-Item C:\hearth-data D:\hearth-data
cd C:\Users\Gaston\hearth-watch
.\.venv-pc\Scripts\python.exe edge\pc\download_datasets.py
```

`download_datasets.py` fetches only the videos: OCCU's 10 colour videos are read straight out of
the remote zip (its table of contents is at the end of the file, so small range requests find
them), CAUCAFall's 100 `.avi` files come through Mendeley's API, and the OF-Synthetic archive is
streamed once, writing only the clips that pass the filter (`--syn-all` keeps all 12,000;
`--only occu` etc. for one dataset). If the 9.7 GB stream breaks, run it again: clips already
saved are skipped, but the stream restarts from the beginning. OF-Synthetic is AV1 video; the
`opencv-python` in `.venv-pc` decodes it (checked).

### Step 6a: `05_extract_keypoints.py`

```powershell
cd C:\Users\Gaston\hearth-watch     # the repo folder (the download steps left you in C:\hearth-data)
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py --check      # match labels to videos only (seconds)
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py --limit 5    # quick trial on 5 videos
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py              # everything
```

`--check` should end with "every label has its video". With only GMDCSA24 + Le2i: about 335
matched videos (upright ~1,200 s, sitting ~610 s, floor ~575 s of labelled time); with all five
datasets it also lists occu (10 videos), caucafall (100) and of-syn (the downloaded clips).
Data now defaults to `D:\hearth-data` (`--data` to change). OCCU caveat: its falls end hidden
behind a bed on purpose, so most of its "fallen" frames have no visible person and are dropped
(in a test video only ~15% kept a person); the ones left are people half-visible on the floor.
A keypoint model can't judge a fully hidden person; the backend's "person not visible" covers that. The first real run downloads the pose
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

### Step 6b: `06_train.py`

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --show-data   # part 1: what the network is fed
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py               # part 2+3: train (30 epochs, live chart), then test
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --test-only   # part 3 only: field-test the saved model again
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --final-test  # the locked 20% test: once, at the end
```

**Four classes: upright / sitting / floor / other.** "Other" means none of the three: falling
(mid-fall), sitting down, lying down, getting up, kneeling, squatting, crawling, jumping and
miscellaneous activity (bending, picking things up, hopping): ~7,100 s of OmniFall-labelled
footage. `05` stores every frame's original OmniFall label and `06` maps it (`FROM_OMNIFALL`), so
changing the mapping needs no new extraction. Mid-fall is "other" on purpose: the model should
say floor once the person is down, and the backend waits for 3 s of floor anyway. Because a
missed fall is the worst error, reports show **floor caught** (of all floor frames, how many
were called floor) on its own.

First 4-class run (2026-10-10, 15 epochs): worse than 3 classes. "Other" was recognised only 45%
of the time and floor caught fell to 85% (from ~93%): OmniFall's "other" describes activities,
not postures (standing while handling things, the first half of getting up still on the floor),
so single frames labelled "other" look exactly like upright / sitting / floor and contradict
them. So both designs are options, compared on validation:

- `--classes 3` (default): upright / sitting / floor; "other" frames sit out of training; the
  negative ("unknown") comes from low confidence.
- `--classes 4`: "other" is a learned 4th answer, reported as unknown, plus low confidence.
- `--torso-rule` (off by default): also unknown when no shoulder or no hip is visible. It threw
  away right answers on our recordings (lying clips with head and shoulders only), so it's off.

**"Unknown"**: the system declines to answer when its top probability (softmax) is under the
cut-off (and, with the options above, when the network says "other" or the torso is hidden) (`--min-conf`, default 0.70, saved in
the model file). After training, and with `--test-only`, a table on **validation** shows each
cut-off's trade-off: how often it decides vs how often it's right then. Pick the cut-off there,
never on the test. Field-test timelines show `?` for unknown; reports give "decided on X%,
right on Y% of those".

`--final-test` prints accuracy and a confusion table for test real people and test synthetic
clips, and appends a line to `D:\hearth-data\test_log.txt`; it says which run this is. Report
that number; don't tune on it (every look makes it less honest).

- Part 1 (data): each frame -> 51 numbers (17 keypoints x dx, dy, found; aspect ratio undone,
  centred on the body, scaled by its size), a **locked split by person: 60% train / 20%
  validation / 20% test**, a `Dataset` that augments training frames (hidden keypoints,
  mirroring, jitter) and a `DataLoader` serving shuffled batches of 256.
  - Per dataset, people are shuffled with a fixed seed; test takes people until it holds 20% of
    that dataset's frames, then validation 20%, the rest train. Anyone with over 30% of a
    dataset (Le2i's subject 0, many unnamed actors) always trains. OF-Synthetic has no person
    ids, so each clip is its own "person". The result is saved in `D:\hearth-data\split.json`
    and reused forever; people added later to an already-split dataset go to training, so the
    test never changes.
  - Validation picks the best epoch (balanced accuracy on **real** people; synthetic clips are
    reported separately). The test is touched only by `--final-test`.
- Part 2 (training): `PostureNet` (51 -> 64 -> 64 -> 3, 7,683 weights), class-weighted
  cross-entropy (training is ~52% upright), AdamW. Each epoch: forward, loss, `backward()`,
  optimizer step for every batch, then a measurement on the unseen people. Shows a progress line,
  a summary per epoch and a live chart; keeps the epoch with the best **balanced** accuracy in
  `C:\hearth-data\posture_mlp.pt`, curves in `training_curves.png`, then prints a confusion table.

- Part 3 (the real test): the saved model on our own Jetson recordings (`recordings/*.jsonl`,
  committed in the repo, so `git pull` brings them): clip name = truth (stand / walk -> upright,
  sit -> sitting, lie_down -> floor, minus its first 5 s of getting down). Prints % right per
  clip, a timeline of predictions per clip (U / S / F, `.` = not scored) and an overall
  confusion table. The validation score only shows it learned the public datasets (same rooms
  and cameras, clean YOLO keypoints); this one shows whether it works with our camera and poseNet.

First run (2026-10-10): validation 99-100% from epoch 4 on (best epoch 26; confusion upright
2028/4/2, sitting 0/1829/10, floor 0/8/1517).

### Step 6d: `07_finetune_closeups.py` - "unsure" for close-ups

Chosen model so far: `model_3class.pt`, cut-off 0.50 (validation: answers 99.5%, right 94.4%,
floor caught 93.1%; our recordings: answers 99%, right 89%, floor caught 96%). Its main error:
close-ups. With only head and shoulders in view (stand-...053107) it confidently says floor.

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\07_finetune_closeups.py
```

- Close-up examples are cropped from the training frames (hips, knees, ankles hidden, sometimes
  arms too). Upright / sitting crops -> "other"; floor crops stay "floor", because lying with
  the legs hidden is still lying (lie_down-...061342). The network has to use the orientation of
  head and shoulders. Our recordings stay out of training.
- Fine-tune: start from `model_3class.pt`, add a 4th output (old outputs keep their weights; the
  new one starts almost never firing), 10 epochs at a very low learning rate (1e-4).
- "Unsure" about a real posture is punished hard: extra loss = `--unsure-penalty` (3.0) x the
  probability given to "other" on frames that really are upright, sitting or floor.
- Safety: an epoch is kept only if floor caught on real validation frames stays at or above the
  starting model's. Otherwise it says so and `model_3class.pt` stays the choice.
- Per epoch it prints real-frame numbers and validation close-ups (upright/sitting -> unknown,
  lying -> floor); then the field test. Output `D:\hearth-data\model_closeup.pt`, readable by
  `06_train.py --test-only / --final-test --out ...`.

First run (2026-10-10, defaults): close-ups -> unknown rose from 4% to 95% by epoch 6, lying
close-ups stayed floor (87%), real accuracy went 94.4 -> 94.8%, but floor caught dipped
93.1 -> 92.2% (about 20 of 2,062 floor frames: genuine floor frames that look like close-ups), so
the zero-tolerance rule kept nothing. Options:

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\07_finetune_closeups.py --floor-penalty 5        # punish "unsure" 6x on real floor frames
.\.venv-pc\Scripts\python.exe edge\pc\07_finetune_closeups.py --floor-tolerance 1      # or accept up to 1 point less floor caught
```

Run 2 (`--floor-penalty 5`, upper-body crops only): kept epoch 6, `model_closeup.pt`: validation
floor caught 93.2%, right 94.8%, close-ups -> unknown 85%, lying close-ups -> floor 93%; our
recordings floor caught 97%. But head-only (stand-...053107) still said floor and legs-only
(walk-...034202) newly said floor, because the crops only ever hid the legs. Now three crop
kinds: upper body (lying stays floor), head only (always "other"), legs only (lying stays floor).

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\07_finetune_closeups.py --floor-penalty 5 --out D:\hearth-data\model_closeup2.pt
```

Then: pick `model_3class.pt`, `model_closeup.pt` or `model_closeup2.pt`, freeze it, and run
`06_train.py --final-test --out <that file> --min-conf 0.5` on it once.

### Step 6c onward
- Export to ONNX; a network this small runs on the Jetson with plain numpy in under 1 ms.

Both datasets are research data: credit them, don't build a product on them (Le2i is
non-commercial).

## Licenses

The fall model is AGPL-3.0 (declared by its author); anything derived from it (the ONNX export,
a fine-tuned model) keeps that license and attribution. PyTorch is BSD-style.
Step 6 data: OmniFall labels CC BY-NC-SA 4.0; GMDCSA24 CC BY 4.0 (Alam et al., Data in Brief
2024); Le2i CC BY-NC-SA (Charfi et al., Laboratoire Le2i / UBFC); OCCU CC BY 4.0 (Zenodo
15494102); CAUCAFall CC BY 4.0 (Eraso et al., Mendeley Data 7w7fccy7ky v4); OmniFall
OF-Synthetic CC BY-NC 4.0. The model we train from them
is ours, but must credit the datasets and stays non-commercial because of Le2i and the OmniFall
labels. Ultralytics YOLO11-pose (AGPL-3.0) is used only as a tool to produce the training
keypoints; the classifier we train contains none of its code or weights.

## Quick run: all five datasets (after `download_datasets.py`)

From the repo folder (`cd C:\Users\Gaston\hearth-watch`):

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py --check
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --show-data
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --epochs 15 --classes 3 --out D:\hearth-data\model_3class.pt
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --epochs 15 --classes 4 --out D:\hearth-data\model_4class.pt
```

Compare the two runs' validation cut-off tables (floor caught first), keep the better one.

1. `--check`: all five datasets should be listed and matched.
2. Extraction: about 2,600 videos, and every labelled segment now (including "other"), so maybe
   30-45 minutes, mostly the 2,203 synthetic clips at 1280x720.
3. `--show-data`: shows which people landed in validation and test, and the frames per part.
4. Training: this retrains from scratch on the new split. The old `posture_mlp.pt` gets replaced.

Leave `06_train.py --final-test` until you've finished changing things.

## Close-up fine-tune, run 3 and next (2026-10-10)

Run 3 (three crop kinds, `--floor-penalty 5`, out `model_closeup2.pt`) kept nothing:

| epoch | floor caught (start 93.1%) | should-be-unsure close-ups -> unknown | lying close-ups -> floor |
|---|---|---|---|
| 1 | 91.8% | 33% | 94% |
| 4 | 92.9% | 49% | 96% |
| 10 | 91.8% | 80% | 93% |

- The head-only and legs-only crops are harder, so the close-up score climbs more slowly (80%
  after 10 epochs, against 85% before). Floor caught hovers at 91.7-92.9%, just under the bar.
- The bar is now the problem, not the model: epoch 4 missed by 0.2 points, about 4 of 2,062
  floor frames, which is noise (retraining with different random batches moves floor caught
  more than that). A drop of up to half a point (~10 frames) counts as "the same".
- Honesty note: `stand-...053107` and `walk-...034202` were used to design the crops, so our
  recordings are no longer a fully independent field test; they are part of development. The
  locked test set is the only unbiased number left.

Next run: stronger floor protection, a fair tolerance, more epochs for the slower close-ups:

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\07_finetune_closeups.py --floor-penalty 10 --floor-tolerance 0.5 --epochs 15 --out D:\hearth-data\model_closeup3.pt
```

- `--floor-penalty 10`: calling a real floor frame "unsure" costs 11x more.
- `--floor-tolerance 0.5`: an epoch is safe if floor caught is at least 92.6%.
- Among safe epochs it keeps the one that says "unknown" to the most close-ups.

Check in the field test: `stand-...053107` -> `?` instead of `F`; `walk-...034202` -> not `F`;
lie-downs still `F`. If it still keeps nothing, `model_closeup.pt` (run 2) is the fallback: it
passed the strict rule and improved floor caught; its gaps are head-only and legs-only framing.

## Close-up fine-tune, run 4 and next (2026-10-10)

Run 4 (`--floor-penalty 10 --floor-tolerance 0.5 --epochs 15`, out `model_closeup3.pt`) kept
nothing either:

| | penalty 5 (run 3) | penalty 10 (run 4) |
|---|---|---|
| floor caught after epoch 1 | 91.8% | 91.8% |
| best floor caught | 92.9% (epoch 4) | 92.2% (epoch 5) |
| close-ups -> unknown at the end | 80% (10 epochs) | 76% (15 epochs) |
| lying close-ups -> floor | 93-96% | 93-97% |

Doubling the floor penalty changed nothing: floor caught drops to ~91.8% in the first epoch and
stays there. So the lost floor frames are probably not being called "other" (the only thing the
penalty punishes). Likely cause: with a 4th answer the probability is split four ways instead
of three, so some borderline floor frames now have a top probability just under the 0.50
cut-off and come out "unknown" for low confidence, which the penalty can't touch.

Next run, testing that guess and possibly fixing it (the starting point is re-measured at the
same cut-off, so the comparison stays fair):

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\07_finetune_closeups.py --floor-penalty 5 --min-conf 0.4 --floor-tolerance 0.5 --out D:\hearth-data\model_closeup3.pt
```

- If an epoch is kept, the guess was right; the field test prints right after.
- If nothing is kept, the dip is real and it's a trade-off decided by the field test:

  ```powershell
  .\.venv-pc\Scripts\python.exe edge\pc\07_finetune_closeups.py --floor-penalty 5 --floor-tolerance 1.5 --out D:\hearth-data\model_closeup3.pt
  ```

  This keeps the epoch with the most close-ups caught (~80%) at ~1.3 points less floor caught.
  Worth it only if the field test shows `stand-...053107` and `walk-...034202` no longer called
  floor while the lie-downs stay `F`.

Time check: after this, stop iterating and decide. Still ahead: the locked test, the Jetson, the
demo. If neither run gives a clean win, `model_3class.pt` is a solid choice; its one known
weakness (close-up framing) is handled by placing the camera so it sees the whole body.

## Decision: `model_3class.pt`, cut-off 0.50 (2026-10-10)

Run 5 (`--floor-penalty 5 --min-conf 0.4 --floor-tolerance 0.5`) kept nothing: floor caught fell to
91.8% in epoch 1 again and never recovered (close-ups -> unknown reached 77%). The dip is identical
whatever the penalty or cut-off, so it isn't low confidence. Run 2 (upper-body crops only)
recovered to 93.2%; every run with head-only crops stayed at 91.8%. Head-only crops teach "just a
face -> unsure", but some real falls look exactly like that (OCCU: only the head visible past the
bed). One frame can't tell a face behind furniture from a close-up: a limit of single-frame
models, not of tuning. A model that watches motion over several frames would be the real fix.

The two models that passed the safety rule, on our real recordings:

| | `model_3class.pt` | `model_closeup.pt` (run 2) |
|---|---|---|
| validation: right / floor caught | 94.4% / 93.1% | 94.8% / 93.2% |
| field: false "floor" on standing clips | 86 frames (head-only close-up) | 104 (head-only close-up + legs-only walk) |
| field: floor caught | 96% | 97% |

Chosen: `model_3class.pt` at 0.50. Simpler, fewer false alarms in our room; close-up framing is
handled by placing the camera so it sees the whole body. Next: freeze it and run the locked test
once:

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --final-test --out D:\hearth-data\model_3class.pt --min-conf 0.5
```

## Locked test result: `model_3class.pt`, cut-off 0.50 (run 1, 2026-10-10 21:11)

People the model never saw and that never influenced a decision (20% of each real dataset's
people, plus 20% of the synthetic clips), scored once:

| | real people | synthetic clips |
|---|---|---|
| network alone (always answering) | 98% right, balanced 97% | 91% right, balanced 91% |
| answers (cut-off 0.50) | 99.7% | 96.3% |
| right when answering | **97.7%** (upright 99%, sitting 96%, floor 96%) | 92.1% |
| floor caught | **96.1%** | 88.6% |

Real people, frame counts:

| actually / predicted | upright | sitting | floor | unknown |
|---|---|---|---|---|
| upright | 5,568 | 51 | 13 | 15 |
| sitting | 40 | 2,072 | 39 | 15 |
| floor | 52 | 47 | 2,421 | 0 |

- False "floor" on upright people: 13 of 5,647 frames (0.2%).
- Test (97.7%) beat validation (94.4%): normal variation between groups of people (some are
  easier to read), not a mistake; which is why it's reported once and not tuned on. Honest
  summary: about 94-98% on unseen people.
- Our own camera (Jetson poseNet recordings): floor caught 96%, so the model transfers.

Full write-up of the results (test, validation, field test, false positive rates, what was tried): [RESULTS.md](RESULTS.md).

## Step 7a: `08_export_weights.py` - the model for the Jetson

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\08_export_weights.py
```

The Jetson (Python 3.6, numpy 1.13) can't run PyTorch and doesn't need to: the network is three
matrix multiplications. This writes `edge\hearth_edge\posture_mlp.npz` (~37 KB, committed with
the code; `edge/models/` would be git-ignored): the three layers' weights (W1 64x51, W2 64x64,
W3 3x64 and biases, 7,683 numbers), the classes, the cut-off (0.50, chosen on validation), the
feature recipe, the source model's SHA-256 and the locked test result. Before writing, it runs the
same inputs through PyTorch and a plain numpy copy of the network (validation frames and our
recordings) and refuses to export unless scores, classes and decisions match (tested: largest
difference ~4e-6, 100% identical). The Jetson's numpy 1.13 loads the file (checked).

## Step 7b: `edge/hearth_edge/posture_mlp.py` - the model running on the Jetson

`PostureMLP().predict(keypoints, width, height)` takes poseNet's keypoints as `run_edge.py` has
them (`{name: (x, y)}` in pixels) and returns `{"label", "confidence", "probabilities"}`: label
"unknown" under the 0.50 cut-off, `None` with fewer than 4 keypoints. numpy only, Python 3.6;
0.56 ms per frame on the Nano. Inputs are built exactly as in training (poseNet's extra `neck`
ignored). `08_export_weights.py` runs every frame of our recordings through this module after
exporting and requires 100% the same decisions as the PyTorch pipeline (tested: 1,382 of 1,382);
`edge/tests/test_posture_mlp.py` checks its rules on the laptop (position and size don't change
the inputs, frame shape is undone, unknown under the cut-off, too few keypoints).

## Step 7c: live on the Jetson (on by default)

```sh
edge/tools/view_camera.sh                                  # laptop: camera window, nothing sent
edge/tools/live_test.sh --real                             # with the team backend
```

(Update: the model is now on by default everywhere and the team backend's config trusts it;
`--no-posture-mlp` turns it off; the panel's aiming lines were removed. See
[`../POSTURE_MODEL.md`](../POSTURE_MODEL.md).)

Every frame with a pose (0.56 ms) goes through our model on poseNet's raw keypoints. The panel
shows a green line: `our model: floor 0.97`, `our model: unknown (0.43)` or `our model: - (too few
keypoints)`; the terminal status line (every 5 s) adds its label counts for the interval. Not
added to the observations: the shared contract rejects unknown fields and the backend only
accepts the Hugging Face model's predictions. If the weights file is missing it logs that and
runs without it.

Several people in view: our model runs on every pose poseNet finds (the tracked person uses
their rotated-pass pose when there is one; their plain-pass copy is dropped), left to right:
`our model, 2 people (left to right): upright 0.98 | FLOOR 0.95  <- someone on the floor`, in
amber when anyone is on the floor. The status line counts `SOMEONE ON FLOOR (frames)`. The
tracked person still drives the observations sent to the backend.

Second person lying down: they now get rotated poseNet poses too (`pose_rotate.OthersRotation`):
every rotated pass sees everyone, so passes made for the tracked person are reused for free; when
none ran, one extra pass runs at most every 0.5 s while someone else looks lying or incomplete
(every 1 s otherwise, to find a lying person the plain pass missed). A rotated pose replaces a
person's plain one only when it clearly beats it (more keypoints, torso visible, same place),
and is remembered for 1 s. Long panel lines now wrap to the frame width.

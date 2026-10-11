# Hearth posture model: results

Our own PyTorch model that reads body keypoints and says **upright / sitting / floor**, trained on
public fall datasets and tested on people it never saw. Written and trained on an RTX 3060 Ti
(`edge/pc/`, see README "Step 6"). All numbers below are measured; dates are 2026-10-10.

## Summary

On a locked test set of people the model never saw (scored once):

| | real people |
|---|---|
| right when answering | **97.7%** |
| floor caught (recall) | **96.1%** |
| precision for "floor" | **97.9%** |
| false positive rate (non-floor frames called floor) | **0.67%** |

On our own camera (Jetson poseNet recordings): floor caught **96%**. Known weak spot: a close-up
where only the head is visible gets called floor (fixed by camera placement: whole body in view).

## How it works

1. **Data:** 5 public datasets with timed labels from [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall):
   GMDCSA24, Le2i, OCCU, CAUCAFall (real people) and OF-Synthetic (AI-generated; filtered to indoor,
   adults and elderly, camera at eye height / high / top). Labels folded into our classes:
   walk + standing -> upright, sitting -> sitting, fallen + lying -> floor. Transitions and other
   activities are left out.
2. **Keypoints:** YOLO11m-pose (pretrained, used as a tool) on every 2nd frame -> 17 COCO
   keypoints per frame (the same body points the Jetson's poseNet reports). Numbers only, no video.
3. **Inputs:** per frame 51 numbers (17 keypoints x dx, dy, found): frame aspect ratio undone,
   centred on the body, scaled by its size, so position and distance in the room don't matter,
   only body shape and orientation. A keypoint counts as found at confidence >= 0.5 (like
   poseNet, which only reports what it found); frames with fewer than 4 found are dropped.
4. **Model:** multilayer perceptron 51 -> 64 -> 64 -> 3 (ReLU, dropout 0.2), **7,683 weights**.
   Class-weighted cross-entropy, AdamW (lr 1e-3, weight decay 1e-4), batches of 256, 15 epochs
   (~2.4 s each on the 3060 Ti). Training-only augmentation: each keypoint hidden with
   probability 0.2 (poseNet misses joints YOLO finds), left/right mirroring, small jitter.
5. **Decision:** the class with the highest probability; "unknown" if that probability is under
   **0.50** (in practice it almost never triggers: 0.3% of test frames).

## Locked split (60 / 20 / 20 by person)

Every dataset's people were shuffled with a fixed seed and split before training: test takes
people until it holds 20% of that dataset's frames, then validation 20%, the rest train (anyone
with over 30% of a dataset always trains). Synthetic clips have no person ids, so each clip is
its own "person". The split is saved in `split.json` and never redrawn.

- **Train:** 53,563 frames. **Validation:** 9,791 real frames (picked the best epoch, the cut-off,
  and every design choice). **Test:** 10,333 real frames + 11,361 synthetic, opened once.
- Validation and test only contain people the model never trained on.

## Locked test results (run 1, 21:11, `model_3class.pt`, cut-off 0.50)

### Real people

| actually / said | upright | sitting | floor | unknown | right |
|---|---|---|---|---|---|
| upright (5,647) | **5,568** | 51 | 13 | 15 | 98.6% |
| sitting (2,166) | 40 | **2,072** | 39 | 15 | 95.7% |
| floor (2,520) | 52 | 47 | **2,421** | 0 | 96.1% |

| Measure | Value | Counted from |
|---|---|---|
| right when answering | **97.7%** | |
| answers | 99.7% | |
| network alone (always answering) | 98% right, balanced 97% | |
| floor caught (recall, true positive rate) | **96.1%** | 2,421 of 2,520 floor frames |
| false positive rate | **0.67%** | 52 of 7,813 upright/sitting frames called floor |
| ...of upright frames | 0.23% | 13 of 5,647 |
| ...of sitting frames | 1.8% | 39 of 2,166 |
| precision for "floor" | **97.9%** | 2,421 of 2,473 "floor" calls |
| missed floor (false negative rate) | 3.9% | 99 frames: 52 called upright, 47 sitting |

Reading the rows:
- **Upright:** almost all right; the 51 called sitting are likely bent knees, leaning or slight
  crouching. Only 13 called floor (the false-alarm kind of error).
- **Sitting:** the hardest class. Sitting straight seen from the front looks like standing (40);
  low or slumped sitting looks like floor (39).
- **Floor:** 96.1% caught; the misses are likely people propped on an elbow or curled up, and
  frames at the edge of a hand-marked label.
- **Unknown:** 30 frames, none on floor. The model doesn't go silent during a fall.

### Synthetic clips (reported separately)

| actually / said | upright | sitting | floor | unknown |
|---|---|---|---|---|
| upright (5,136) | 4,487 | 330 | 58 | 261 |
| sitting (2,505) | 38 | 2,294 | 113 | 60 |
| floor (3,720) | 194 | 131 | 3,297 | 98 |

Right when answering 92.1%, answers 96.3%, floor caught 88.6%, false positive rate 2.2%
(171 of 7,641), precision 95.1%. Harder than real people: AI bodies are sometimes distorted, and
from directly overhead (top cameras are included) lying and standing can look alike.

### How to read the test number

- **Test (97.7%) beat validation (94.4%).** Different people: with only a handful of people per
  group (20% of each dataset's people, roughly 6-8 in total), who lands where moves the score by
  several points. Honest claim: **about 94-98% on unseen people**. Thousands of frames make the
  numbers look precise, but frames from one person are near-copies; a different handful of
  people could easily move it +-2-3 points.
- These are **per-frame** rates. The backend raises a possible fall only after **3 s of
  continuous floor**, so scattered single errors (0.67%) rarely line up into an alert. The
  alert-level false alarm rate has not been measured yet (it needs clips replayed through the
  backend).

## Validation (real people, picked the model)

| actually / said | upright | sitting | floor |
|---|---|---|---|
| upright | 5,395 | 112 | 39 |
| sitting | 238 | 1,902 | 43 |
| floor | 99 | 44 | 1,919 |

Best epoch 13: 94% right (balanced 92%) always answering; at cut-off 0.50: answers 99.5%, right
94.4% (upright 98%, sitting 88%, floor 93%), floor caught 93.1%. Synthetic validation clips: 90%.

Cut-off trade-off (validation): raising the cut-off buys accuracy but loses caught falls
(0.50: floor caught 93.1%, 0.70: 91.8%, 0.95: 89.1%), so the lowest useful cut-off (0.50) was chosen.

## Field test: our Jetson recordings (our camera, poseNet)

16 clips of one of us standing, walking, sitting and lying on a mat, recorded on the Jetson (real
poseNet output, never trained on; the first 5 s of lie-down clips, still getting down, aren't
scored). At cut-off 0.50:

| actually / said | upright | sitting | floor | unknown |
|---|---|---|---|---|
| upright (553) | 466 | 0 | 86 | 1 |
| sitting (199) | 17 | 176 | 0 | 6 |
| floor (299) | 10 | 0 | 287 | 2 |

- Answers 99%, right 89% (upright 84%, sitting 91%, floor 97%), **floor caught 96%**; all 8
  lie-down clips 84-100% right.
- False positive rate 11.4% (86 of 752), almost all from **one clip**: a close-up where poseNet
  saw only the head (`stand-...053107`, ~78 of the 86 frames). Without it, about 1.2% (approximate).
  A head-only close-up is called floor continuously, so unlike scattered errors it would add up
  to 3 s: camera placement (whole body in view) matters.
- These clips were also used to diagnose problems during development, so they are not a fully
  independent test; the locked test is.

## What we tried and rejected

| Idea | Result | Decision |
|---|---|---|
| Torso rule ("unknown" without a shoulder and a hip) | Threw away right answers: lie-downs where poseNet saw only head and shoulders | off |
| Learned 4th class "other" from OmniFall activity labels | "Other" frames look like the 3 postures; floor caught fell to 85% (74% at cut-off 0.70), field floor caught 67% | rejected |
| Close-up fine-tune (5 runs): cropped close-ups as "other", very low learning rate, heavy penalty on "unsure" for real postures | Upper-body crops worked (close-ups -> unknown 85%, floor caught 93.2%), but head-only crops conflict with real falls seen head-only behind furniture (floor caught stuck at 91.8%); on our recordings the passing model made more false floors | kept the 3-class model |

Lesson: a single frame can't tell a face peeking past a bed from a close-up. A model that watches
motion over several frames would be the real fix.

## Limits

- Real datasets are mostly young actors staging falls on mats; elderly people fall differently
  (slower, sliding, partial). OF-Synthetic adds elderly bodies, but AI-generated.
- One person at a time.
- A fully hidden person (OCCU's falls behind a bed) gives no keypoints at all; the backend's
  "person not visible" covers that, not this model.
- Not yet connected to the backend: Person 2's backend only accepts the Hugging Face model's id.

## Reproduce (PowerShell on the PC, from the repo)

```powershell
.\.venv-pc\Scripts\python.exe edge\pc\download_datasets.py
.\.venv-pc\Scripts\python.exe edge\pc\05_extract_keypoints.py
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --epochs 15 --classes 3 --out D:\hearth-data\model_3class.pt
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --test-only --out D:\hearth-data\model_3class.pt --min-conf 0.5
.\.venv-pc\Scripts\python.exe edge\pc\06_train.py --final-test --out D:\hearth-data\model_3class.pt --min-conf 0.5
```

GMDCSA24 and Le2i are downloaded by hand first (README "Step 6"). Augmentation is random, so a
retrain gives slightly different numbers; the locked split is fixed.

## Credits

Labels: OmniFall (CC BY-NC-SA 4.0). Videos: GMDCSA24 (CC BY 4.0, Alam et al., Data in Brief 2024),
Le2i (CC BY-NC-SA, Charfi et al., Laboratoire Le2i / UBFC), OCCU (CC BY 4.0, Zenodo 15494102),
CAUCAFall (CC BY 4.0, Eraso et al., Mendeley Data 7w7fccy7ky v4), OF-Synthetic (CC BY-NC 4.0).
Keypoints from Ultralytics YOLO11-pose (AGPL-3.0), used only as a tool; the trained model
contains none of its code or weights. Non-commercial because of Le2i and the OmniFall labels.
Research use only; not a clinical device.

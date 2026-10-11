# Our posture model: how it works and how to use it

Hearth's own posture model, written and trained in PyTorch on public fall datasets
(`edge/pc/`), running on the Jetson Nano in plain numpy. It reads the body keypoints poseNet
already finds and says **upright / sitting / floor** for every person in view, and it is the
model the team backend uses for possible-fall alerts. **It is on by default everywhere**:
`run_edge.py` and every laptop tool run it, and the team backend's config trusts it.

On a locked test set of people it never saw: **right 97.7%, floor caught 96.1%, precision 97.9%,
false positive rate 0.67%** per frame; on our own camera, floor caught 96%. Full results:
[`pc/RESULTS.md`](pc/RESULTS.md).

## Quick start (laptop, from the repo root)

```sh
edge/tools/live_test.sh --real                       # team backend + Jetson + video window
```

Then open http://127.0.0.1:8000/nurses (room-01, Possible falls) and
http://127.0.0.1:8000/sources. Ctrl+C once to stop everything.

Just the camera window, nothing sent: `edge/tools/view_camera.sh`. Turn the model off with
`--no-posture-mlp` (after `--` for `live_test.sh`).

## How it works, end to end

```
camera frame (640x480, ~13 FPS)
  -> poseNet (TensorRT): keypoints for every person in the frame          [pose.py]
  -> rotated poseNet passes for anyone who looks lying / incomplete       [pose_rotate.py]
  -> tracker picks the person the edge follows (one per room)             [features.py PersonTrack]
  -> OUR MODEL on every person: keypoints -> upright / sitting / floor    [posture_mlp.py]
  -> one PosturePrediction for the backend (tracked person, or whoever is on the floor)
  -> Observation (keypoints, posture numbers, face, posture_prediction), ~10/s, over USB
  -> team backend FallDetector: 3 s of "fallen" -> possible_fall; 1 s "standing" -> recovered
  -> nurse view (/nurses), source view (/sources)
```

### 1. Keypoints

poseNet (`resnet18-body`, jetson-inference) finds people and their joints in each frame. A person
lying on the floor looks unusual to it, so when the followed person looks lying or incomplete the
frame is also rotated 90 degrees and run again (`RotationPolicy`); the rotated pose is used when it
clearly finds more of the body. Everyone else in view gets the same treatment from
`OthersRotation`: every rotated pass sees everyone, so passes made for the followed person are
reused, plus one extra pass at most every 0.5 s while someone else looks lying (every 1 s
otherwise, to find someone the plain pass missed). Each rotated pass costs ~75 ms.

### 2. The model's inputs (`PostureMLP.features`)

Each person becomes 51 numbers: for each of the 17 COCO keypoints (nose, eyes, ears, shoulders,
elbows, wrists, hips, knees, ankles), `dx, dy, found`:

1. each keypoint in 0-1 of the frame, x stretched back to true proportions (times width / height);
2. minus the centre of the found keypoints (where the person stands in the room doesn't matter);
3. divided by the larger side of their box (near or far doesn't matter);
4. missing keypoints are 0 with `found` 0, so the network knows they're missing.

What's left is the body's shape and orientation, which is what separates upright, sitting and
floor. poseNet's extra `neck` isn't used. Fewer than 4 keypoints: no answer (`None`).

### 3. The network and the decision (`PostureMLP.predict`)

Three layers, 51 -> 64 -> 64 -> 3 (ReLU between), 7,683 numbers in total, trained in PyTorch on
~100,000 labelled frames from five public fall datasets (`pc/`). The three output scores go
through softmax (probabilities that add up to 1); the highest wins. If it is under the cut-off
**0.50** (chosen on validation people) the answer is `unknown`. 0.56 ms per person on the Nano.

### 4. What the backend is told (`PostureMLP.backend_prediction`)

One `posture_prediction` per observation (the shared contract allows one), describing one person:

- the person the edge is following, **unless someone else is on the floor** with probability
  >= 0.50 (then that person: someone down matters most); with nobody followed, the largest person;
- label mapping: upright -> `standing`, sitting -> `sitting`, floor -> `fallen`; plus the
  probability as `confidence`, the person's box (0-1), `detection_count` 1;
- under the cut-off or too few keypoints: `unknown` (detection_count 1);
- nobody in view: `unknown`, `no_detection`, detection_count 0. It is still sent: the backend
  restarts its floor timer whenever the evidence source changes, so a prediction must be on
  **every** observation;
- `model_id` `hearth/posture-mlp`, `model_revision` = first 12 characters of the trained model's
  SHA-256 (changes whenever the model is retrained).

### 5. How the backend turns it into alerts (Person 2's `FallDetector`)

With a prediction present the backend uses it, not its pose rules. A prediction counts
("reliable") only if `model_id` equals `fall_model_id`, `detection_count` is 1 and `confidence` is
at least `model_confidence_min` (0.5) in `backend/hearth/monitor/config.json`. Then:

| Rule (config.json) | Effect |
|---|---|
| reliable `fallen` for `floor_hold_seconds` (3 s) in a row | **possible_fall** alert |
| anything else in between (`standing`, `sitting`, `unknown`, unreliable) | the 3 s timer restarts |
| reliable `standing` for `model_standing_hold_seconds` (1 s) after an alert | **recovered** |
| a gap over `max_observation_gap_seconds` (1 s) between observations | timer restarts |
| no person detected | posture unknown, timer restarts |
| no observations for `source_timeout_seconds` (3 s) | **source_offline** (silence is not safety) |

So one wrong frame can't raise an alert (it takes ~30 "fallen" frames in a row), but every frame
counts toward breaking a hold, which is why the cut-off was chosen to keep "floor caught" high.

## What you see

| Where | What happens |
|---|---|
| Jetson, every frame | Every person poseNet finds goes through the model (0.56 ms each). People lying down get rotated poseNet passes too (also someone the edge isn't tracking). |
| Video panel | `our model: floor 0.97`; with several people `our model, 2 people (left to right): upright 0.98 \| FLOOR 0.95  <- someone on the floor`, in amber when anyone is on the floor. |
| Terminal (every 5 s) | Label counts for the interval, plus `SOMEONE ON FLOOR (frames)`. |
| Backend | Every observation carries a `posture_prediction` from our model (`model_id` `hearth/posture-mlp`, `model_revision` = the trained model's fingerprint). The backend raises **possible_fall** after 3 s of "fallen" and **recovered** after 1 s of "standing". |

Which person the backend hears about: the one the edge tracks, unless someone else is on the
floor (confidence >= 0.50), because someone down matters most. Nobody in view is sent as
`unknown` / `no_detection`, never left out: the backend restarts its 3 s floor timer whenever the
evidence source changes, so the prediction is on every observation.

With `--fall-model` as well, the Hugging Face fall model is shown on the panel only; ours goes
to the backend. With `--no-posture-mlp`, the backend gets no model prediction (or the HF one with
`--fall-model`) and uses its pose rules.

## The backend side

Person 2's backend uses a model's prediction only when its id equals `fall_model_id` in
`backend/hearth/monitor/config.json` and its confidence is at least `model_confidence_min`. That
config now says `fall_model_id = "hearth/posture-mlp"` and `model_confidence_min = 0.5` (our
cut-off, chosen on validation; any frame under the threshold restarts the 3 s floor timer), so any
normal backend start trusts our model:

```sh
.venv/bin/python -m uvicorn hearth.api.app:app --app-dir backend --host 0.0.0.0 --port 8000
```

To use the Hugging Face model with the backend instead (Person 2's laptop runner, or the Jetson
with `--no-posture-mlp --fall-model`), start it with a config whose `fall_model_id` is
`melihuzunoglu/human-fall-detection` (and `model_confidence_min` 0.65), via `HEARTH_CONFIG`
(`live_test.sh --real --config <file>`).

## No floor / hip lines

The backend decides floor and upright from the model's label alone when a prediction is present,
and with our model there's one on every observation; its head/hip heights only matter for the
pose rules. Our model also doesn't care where in the frame a person is (its inputs are centred and
scaled to the body), so the panel has no aiming lines any more: just keep the whole body in view.

## Camera setup

The model needs to **see the whole body**: head to feet when standing, side to side when lying.
Height and distance matter much less than for the pose rules. Avoid:

- **close-ups** where only a head (or only legs) is in view: a head-only close-up is called
  "floor" (the model's one known false alarm, see RESULTS.md);
- a bright window behind people (silhouettes lose joints).

A good default: 1.5-2 m high, tilted down, 3-3.5 m from where someone could fall.

## Files

| File | What |
|---|---|
| `hearth_edge/posture_mlp.py` | The model on the Jetson: inputs, network, `predict()`, `backend_prediction()` |
| `hearth_edge/posture_mlp.npz` | The trained weights + settings (37 KB, from `pc/08_export_weights.py`) |
| `hearth_edge/pose_rotate.py` (`OthersRotation`) | Rotated poses for people the edge isn't tracking |
| `pc/` | Training pipeline: datasets, keypoints, training, tests, export ([`pc/README.md`](pc/README.md)) |
| `tests/test_posture_mlp.py` | Rules, the contract, and Person 2's real detector alerting on our model |

## Updating the model

Retrain on the PC (`pc/README.md`, steps 6-7), then `pc/08_export_weights.py`: it checks that the
Jetson module gives the same decisions as PyTorch on every recorded frame before writing
`posture_mlp.npz`. Commit the new file; `model_revision` changes with it, so the backend treats
it as a new evidence source.

## Limits

- Per-frame accuracy; alerts additionally need 3 s of "fallen" in a row (the backend).
- The backend gets one person per observation (the panel shows everyone).
- A fully hidden person (behind a bed) has no keypoints; the backend's "person not visible"
  covers that, not the model.
- Trained on young actors staging falls plus synthetic people; not a clinical device.

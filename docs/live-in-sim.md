# The live Jetson feed inside the simulation

Real people seen by the Jetson camera appear in the Sim Studio's 3D room, one avatar each. They can be there alone ("Live only") or alongside simulated residents ("Link into this room"). Each shows what the Jetson's own posture model says about them.

## How the parts connect

```
Jetson (run_edge.py, poseNet + posture model) ──POST /ingest/observations──▶ MonitorService (real rules, alerts, nurse desk)
   one observation per person: person_id p1, p2…                  │  latest observation + posture + events, per person
                                                                  ▼
                                    LiveLink (hearth/sim/live/link.py), every live-engine tick, for each person
                                      lift keypoints into the room (hearth/sim/live/lift.py)
                                      World.set_external(...)  ──▶  live engine (residents, camera, incidents)
                                                                  │
                                                     WS /sim/live/ws frames ──▶ Sim Studio (LiveView3D)
```

**What each piece does:**
- **People.** The edge gives everyone in view an anonymous id (`p1`, `p2`, …) and sends one observation per person, each with its own keypoints and its own posture-model call (`posture_prediction`). In the room each is `live-jetson-01-p1`, `live-jetson-01-p2`, …; a source without ids (older recordings, `--single-person`) gives one `live-<source>`. Someone whose frames stop while others are still seen is taken out after about 5 s.
- **Monitor.** Every decision comes from the real `MonitorService`, separately for each person: posture (standing, sitting, on the floor), the possible-fall alert and recovery. With the posture model on, that posture is the model's label held over time. The simulator never decides anything about the real people.
- **The Jetson's model.** Its latest call on each person (label and confidence, e.g. *sitting 93%*) is shown in the roster and the inspector, and is in `GET /sim/live/link` under `people[].model` and in each live person's frame (`model`).
- **Placement.** The person's position comes from **one camera and the floor**. Feet, or else knees, hips or neck at a typical height, are found where their ray from the lens meets the floor plane. Upright bodies stand on a vertical plane there, and bodies on the floor lie on the floor.
- **Inside the engine, the live person is a full participant:**
  - residents walk round them;
  - they block the simulated camera's view of whoever is behind them;
  - when the monitor raises a possible fall for one of them, the nearest free resident (getting up from a chair if needed) walks over and kneels beside that person until the monitor sees them up.
- **What the engine never does with them:** they get no ground truth, no behaviour and no incidents. The simulated camera never reports them, because their own camera does.
- **Pausing.** The live person keeps moving while the simulation is paused.

## Calibration, step by step

Calibration makes the simulated camera match the real one. The avatar is placed by casting
rays from the simulated camera, so any mismatch shows up as the avatar standing in the wrong place.

### What you need

| Item | Why | How exact |
|---|---|---|
| The camera mounted **2.2–2.5 m up**, above a front corner of the space, aimed at its middle and tilted **30–35° down** | Keeps your feet in view across the area. From head height, the floor only appears about 4 m out | Roughly; calibration measures the real height and tilt |
| The area's **width and depth**, and **which corner** (left or right) the camera is above | Sets the simulated room and where "outside the area" begins | ±30 cm is fine; measure, pace or estimate |
| A **marker at the centre** of the area (a coin, a sticky note, a tile joint) | Step 2 works out which way the camera points | Within about 10–20 cm of the true centre |
| **Your height** | Every measurement is scaled by it | ±3 cm; a wrong height scales the camera height by the same proportion |
| Tape at the edges (optional) | Shows you the boundary; makes the final check easy | — |

Before you start:
- Jetson connected: `ping 192.168.55.1` replies, with the network fixes in place (fixed IP on the Jetson adapter, its NCM adapter disabled).
- `run_edge.py` running on the Jetson.
- Backend and Studio running.
- Good, even lighting; **no other people in view**. Calibration skips frames with more than one person and says so if that made it fail.
- Plain clothes help poseNet find your joints.

### Step 0 · Set up the room in the Studio (1 minute)
1. Open the Studio. In **Live camera**, set **Source** `jetson-01`, **Their height**, **Area width / depth**, and **Camera at** (left or right corner).
2. Press **Live: tracked area**. The simulated room becomes the area, seen from that corner. The link turns **live** when you're in view.
3. Turn on the **Camera FOV** overlay. The shaded floor is roughly what the camera sees; it gets accurate after calibration.
4. Quick check: stand in the middle. The skeleton should show your **head and at least your hips or knees** (switch the bottom panel's **Real camera** chip on to see the Jetson's 2D skeleton). If your head is cut off, tilt the camera less. If only your head shows, move further away or tilt more.

### Step 1 · Walk (12 s): finds height and tilt

**What the software needs from this walk.** It watches for 12 seconds and keeps every frame that shows your head (eyes, nose, ears or neck) and a hip or knee. For each candidate camera height and tilt, it checks whether your head appears exactly where a person of your height, standing upright, would appear. Your **distance from the camera** varying is what separates "a high camera tilted a little" from "a lower camera tilted more". Side-to-side movement helps less, but is also checked. So the walk has to:
- cover **near and far**: the most important part;
- cover **left and right** a little;
- keep you **upright**, and your **head plus hips or knees in view**, the whole time.

**The route.** Labels below are seen from the camera, for a 5 × 4 m area with the camera at the **left** front corner. Mirror left and right if it's at the right corner. Stay about **1 m inside the edges**: right at the edges, your head or feet leave the frame.

```
            back edge (far from the camera)
   +---------------------------------------+
   |        B ·  ·  ·  ·  ·  ·  · C         |
   |        ·                     ·         |
   |        ·                     ·         |
   |        A ·  ·  ·  ·  ·  ·  · D         |
   +---------------------------------------+
  [cam]     front edge (near the camera)
```

| Time | Do | Why |
|---|---|---|
| Before pressing | Stand at **A**: about 1 m in from the front edge, a third of the way across from the camera's side. Face the camera, arms relaxed. Check the Real camera view shows your head and hips | A is the nearest spot where you're still fully in frame |
| 0–4 s | Press **1 · Calibrate: walk**, then walk straight away from the camera from **A to B** (about 1 m from the back edge). Slow, normal steps | The near-to-far stretch: this is what pins down height and tilt |
| 4–5 s | Pause at **B** for about one second, standing straight | A clean upright frame at the far end |
| 5–8 s | Walk sideways across the back from **B to C**, about two-thirds of the way across | Side-to-side coverage, at a distance |
| 8–12 s | Walk towards the camera from **C to D**, and stop at **D** until it finishes | A second near-to-far stretch, on the other side |

That's about 6–7 m in 12 s, a slow walk.

**While walking:**
- **Face the camera or turn sideways; don't walk with your back to it.** With your back to it, poseNet finds no eyes or nose, only ears, which are less reliable.
- **Stay upright.** Don't bend, crouch, lean, sit, or look at the floor. The maths assumes your head is at full standing height.
- **Arms:** relaxed or swinging naturally; they're not used.
- **Pace:** slow and steady. Fast walking blurs the joints, and a long stop wastes frames.
- **Alone:** nobody else may walk through the view.

**What success looks like:** *Camera set: 2.31 m up, tilted 31.4° (fit ±5 px, 64 frames). Now step 2.*
- **Frames:** usually 50–110 (about 10 per second while you're fully seen). Under 30 means you were often partly out of view, so redo it closer to the middle.
- **Fit:** under about 8 px is good, 8–12 px is usable, and above 12 px is rejected.
- **Sanity check:** the height should be close to how high you mounted the camera (±10 cm), and the tilt close to how far you tilted it. If the height is clearly wrong, check **Their height** first: an error there scales the camera height by the same proportion.

What it checks, and what to do when it fails:

| Message | Meaning | Fix |
|---|---|---|
| *only N usable frames (need 10)* | Fewer than 10 frames showed your head **and** a hip or knee | Stay in view and upright. Step back if you're too close. Check lighting |
| *you stayed in one place* | Your position in the image didn't change enough (less than 15% of the width or 5% of the height) | Walk further: corner to corner, and front to back |
| *the fit is poor* | The frames don't agree on one camera (fit worse than 12 px) | Stay upright the whole time, check **Their height**, and make sure nobody else is in view |
| *no camera height and tilt fits* | Nothing between 1.0–3.0 m and 15° up to 60° down explains what it saw | The camera is outside that range, or the lens isn't the C270 (43°) |

### Step 2 · Stand on the centre marker (3 s): finds which way the camera points
1. Stand **on the centre marker**, upright and still, facing the camera.
2. Press **2 · Calibrate: stand on X**. It shows: *Direction set: turned 4.2° (you stood 9 cm from where the X should be).*
3. If "you stood … from the X" is more than about 30 cm, either the area size or camera corner in Step 0 doesn't match reality, or Step 1's height is off. Fix that and redo both steps.

| Message | Fix |
|---|---|
| *only N usable frames* | Head and hips or knees in view; hold still for the full 3 s |
| *couldn't place you on the floor* | Run Step 1 first |
| *the camera seems turned more than 60°* | Aim the camera at the centre and redo it |

You can skip Step 2 if the camera is aimed carefully at the centre; the software then assumes it points dead centre.

### Step 3 · Check (2 minutes)
1. Walk to each **corner of the area** and to the **centre**. Each time, check the avatar is at the same spot in the simulated room, within about **20 cm**.
2. Stand still for 5 s: the avatar should stay put, not drift.
3. Step **outside** the area: after about 1 s the panel says *Outside the tracked area* and the avatar leaves.
4. Raise an arm, then sit on a chair in the area: the arm and the seated posture should show on the avatar.

If a corner is badly off, redo Step 1 and walk more evenly across the whole area, and to that corner in particular.

### Redo calibration when…
- the camera moves or is bumped, or its angle changes;
- you change the area size or corner;
- someone of a different height uses it: set **Their height** first.

It isn't saved yet. After a backend restart, run Steps 0–2 again; it takes about a minute.

## Run it

```powershell
.\scripts\start.ps1 -Lan                # server, dashboard and Studio; then on the Jetson: python3 run_edge.py --backend http://<laptop>:8000
.\scripts\start.ps1 -Replay             # no Jetson: Gaston's recorded stand / lie down / stand clips, looped, as replay-01
.\scripts\start.ps1 -Replay -People     # the same clips as two people side by side (p2 lies down), with the posture model's calls
```

In the Studio (`http://127.0.0.1:8000/studio/`, or the developer menu in the dashboard header):
1. **Live camera → Live only (empty room).** This makes a room on the Jetson camera preset and links it. To add residents, generate a room instead, then press **Link into this room**.
2. **Pick the source:** `jetson-01` for the live Jetson, `replay-01` for recordings.
3. **Calibrate.** Choose camera **Jetson mount**, then set height, tilt and field of view to match the real camera (edge/DEMO.md: 1.5–2 m up, 10–15° down) and regenerate. Adjust until the avatar stands where the person stands.
4. **Watch the alert.** Lie down in view for 3 s. The monitor's alert turns the avatar red, shows in the event log and in the dashboard's alerts, and a resident comes over. Get up, and they stand down.

The **Real camera** chip on the bottom panel switches the 2D view to the Jetson's own skeleton.

The dashboard and Studio both have a **Maximum live people** control (1–10, default **3**). Avatars follow real walking at a controlled pace: movement is limited to **0.6 m/s**, small position jitter within **6 cm** is ignored, and large changes over **0.8 m** must form a consistent sequence for **0.75 seconds** before following. Consecutive distant estimates more than **0.35 m** apart restart that check; a smoothly moving target can still be followed. A camera stall, posture change, or change of visible body parts cannot bypass the movement limit. Turning is limited to **60 degrees/second**. This deliberately adds lag during fast real movement.

A new track needs **1.5 seconds** and at least **five distinct observations** with a detected person and keypoints before an avatar appears. Gaps over **0.5 seconds** or missing-person evidence reset this confirmation. Jetson provides no person-detection confidence score; posture confidence is not used to decide whether someone exists.

Confirmed people retain their slots through brief dropouts and expire **four seconds after their last valid observation arrives**. New candidates wait for space, with established tracks taking priority. Lowering the limit removes the most recently admitted excess tracks immediately. Unlinking, switching source/run, or replacing the room requires fresh confirmation. Monitoring and alerts still receive every observation, including people waiting for admission.

API, if you want it without the Studio: `POST /sim/live/link {"enabled": true, "source": "jetson-01", "height": 1.7, "max_people": 3}`, and `GET /sim/live/link` for the effective `max_people`, state (`waiting`, `live` or `stale`) and confirmed `people`: for each person their room id, track id, posture, the model's call and whether an alert is open for them. Omit `max_people` to preserve its current value; it starts at three when the backend starts. The top-level `posture` is the first live person's; `alert_open` is true if anyone has one. This update needs a backend restart and refreshed frontend assets, with no Jetson redeployment.

## Limits
- **Depth is inferred, not measured.** A wrong camera height or tilt puts the person in the wrong place; bodies are clamped inside the room. On the floor, from a low camera, expect up to ~0.3 m error.
- **Ids are tracks, not identities.** They follow the nearest box from frame to frame: two people crossing closely can swap ids, and someone who leaves for over a second comes back with a new one (a new avatar). **Their height** is shared by everyone, so people of very different heights are placed less accurately.
- **The simulated room is not the real room.** The furniture is generated, so the real person can appear to stand inside a simulated table. Keep the furniture density low, or use "Live only".
- **The fall model (`fall-model-01`) can't be placed.** It sends no keypoints. Its alerts still reach the monitor and the dashboard.
- **Simulated and live alerts share `room-01`.** In the dashboard they're labelled by source (Simulation or Live camera). The contract's `Alert` has no `person_id`; the monitor puts it in `metadata.person_id`.

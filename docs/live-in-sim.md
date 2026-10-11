# The live Jetson feed inside the simulation

A real person seen by the Jetson camera appears in the Sim Studio's 3D room. They can be there alone ("Live only") or alongside simulated residents ("Link into this room").

## How the parts connect

```
Jetson (run_edge.py, poseNet) ──POST /ingest/observations──▶ MonitorService (real rules, alerts, nurse desk)
                                                                  │  latest observation + posture + events
                                                                  ▼
                                    LiveLink (hearth/sim/live/link.py), every live-engine tick
                                      lift keypoints into the room (hearth/sim/live/lift.py)
                                      World.set_external(...)  ──▶  live engine (residents, camera, incidents)
                                                                  │
                                                     WS /sim/live/ws frames ──▶ Sim Studio (LiveView3D)
```

**What each piece does:**
- **Monitor.** Every decision comes from the real `MonitorService`: posture (standing, sitting, on the floor), the possible-fall alert and recovery. The simulator never decides anything about the real person.
- **Placement.** The person's position comes from **one camera and the floor**. Feet, or else knees, hips or neck at a typical height, are found where their ray from the lens meets the floor plane. Upright bodies stand on a vertical plane there, and bodies on the floor lie on the floor.
- **Inside the engine, the live person is a full participant:**
  - residents walk round them;
  - they block the simulated camera's view of whoever is behind them;
  - when the monitor raises a possible fall for them, the nearest free resident (getting up from a chair if needed) walks over and kneels beside them until the monitor sees them up.
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
- Good, even lighting; no other people in view.
- Plain clothes help poseNet find your joints.

### Step 0 · Set up the room in the Studio (1 minute)
1. Open the Studio. In **Live camera**, set **Source** `jetson-01`, **Their height**, **Area width / depth**, and **Camera at** (left or right corner).
2. Press **Live: tracked area**. The simulated room becomes the area, seen from that corner. The link turns **live** when you're in view.
3. Turn on the **Camera FOV** overlay. The shaded floor is roughly what the camera sees; it gets accurate after calibration.
4. Quick check: stand in the middle. The skeleton should show your **head and at least your hips or knees** (switch the bottom panel's **Real camera** chip on to see the Jetson's 2D skeleton). If your head is cut off, tilt the camera less. If only your head shows, move further away or tilt more.

### Step 1 · Walk (12 s): finds height and tilt
1. Stand inside the area, upright, facing roughly towards the camera.
2. Press **1 · Calibrate: walk**, then for 12 seconds **walk slowly all over the area**: near and far, left and right. Keep upright: no bending, crouching or sitting.
3. On success it shows: *Camera set: 2.31 m up, tilted 31.4° (fit ±5 px, 64 frames). Now step 2.*
   - A fit under about 8 px is good, and 8–12 px is usable. The result is applied straight away.

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
.\scripts\demo.ps1              # backend + Studio; then on the Jetson: python3 run_edge.py --backend http://<laptop>:8000
.\scripts\demo.ps1 -Replay      # no Jetson: Gaston's recorded stand / lie down / stand clips, looped, as replay-01
```

In the Studio (`http://localhost:5174/studio.html`):
1. **Live camera → Live only (empty room).** This makes a room on the Jetson camera preset and links it. To add residents, generate a room instead, then press **Link into this room**.
2. **Pick the source:** `jetson-01` for the live Jetson, `replay-01` for recordings.
3. **Calibrate.** Choose camera **Jetson mount**, then set height, tilt and field of view to match the real camera (edge/DEMO.md: 1.5–2 m up, 10–15° down) and regenerate. Adjust until the avatar stands where the person stands.
4. **Watch the alert.** Lie down in view for 3 s. The monitor's alert turns the avatar red, shows in the event log and on `/nurses`, and a resident comes over. Get up, and they stand down.

The **Real camera** chip on the bottom panel switches the 2D view to the Jetson's own skeleton.

API, if you want it without the Studio: `POST /sim/live/link {"enabled": true, "source": "jetson-01", "height": 1.7}`, and `GET /sim/live/link` for the state (`waiting`, `live` or `stale`), posture and whether an alert is open.

## Limits
- **Depth is inferred, not measured.** A wrong camera height or tilt puts the person in the wrong place; bodies are clamped inside the room. On the floor, from a low camera, expect up to ~0.3 m error.
- **One live person.** The Jetson tracks one person and sends no `person_id`.
- **The simulated room is not the real room.** The furniture is generated, so the real person can appear to stand inside a simulated table. Keep the furniture density low, or use "Live only".
- **The fall model (`fall-model-01`) can't be placed.** It sends no keypoints. Its alerts still reach the monitor and `/nurses`.
- **Simulated and live alerts share `room-01`.** On `/nurses` they're labelled by source (SIMULATED or LIVE).

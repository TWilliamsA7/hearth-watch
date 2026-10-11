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

## Set up the tracked area (do this once)

The real room is larger than one camera can track well. Track part of it:

1. **Tape a rectangle on the floor**, for example 5 m wide and 4 m deep, and put an **X at its centre**.
2. **Mount the camera high above one front corner**, 2.2–2.5 m up, aimed at the X and tilted 30–35° down.
   - Why: from a camera at head height, looking level, the floor only comes into view about 4 m away. Feet and knees drop out of frame, and depth has to be guessed from body proportions. From a high corner, your feet stay in view across the area, and position errors roughly halve.
3. In the Studio's **Live camera** panel:
   - set the area's width and depth, and which corner the camera is in;
   - press **Live: tracked area**. The simulated room becomes the taped area, empty, seen from that corner.
4. **Calibrate:**
   - **1 · Walk:** for 12 s, walk slowly all over the area (near, far, left, right), upright. This solves the camera's height and tilt.
   - **2 · Stand on X:** for 3 s, stand still on the centre mark. This solves which way the camera points.
5. **Check coverage.** Turn on the **Camera FOV** overlay. The shaded floor is what the camera sees. Then walk to each corner of the tape and to the X: the avatar should be within about 20 cm of the same spot.

When you step outside the tape, the avatar leaves the room, and the link reports `outside`.

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

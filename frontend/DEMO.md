# Demo rehearsal

Start Hearth from the repository root: `.\scripts\start.ps1` (add `-Lan` for the Jetson, `-Camera 0` for the laptop webcam, `-Replay` for recorded clips). Open http://127.0.0.1:8000/ for the caregiver view, and `/room` in a second tab for the patient. In the caregiver view, start a nurse session covering room-01 with falls and help requests. Choose **Simulation**, **Live camera** or **Recorded replay** above the main feed; each is labelled.

For a scripted simulation, open **Scenario controls and evaluation** (play icon), choose a scenario, then **Load** and **Play**. **Pause** and **Reset** call the simulator. **Run again** resets and reruns the same seed; it is still a simulation. When the run ends, the evaluation report appears automatically. Earlier runs stay listed; **Live room report** scores the live 3D room so far.

1. **0–15 seconds:** "Hearth helps people review possible safety events in one room." Show the simulated room and point out its simulation label.
2. **15–35 seconds:** Switch to **Live camera** with a consenting teammate. Show the skeleton and the posture in source health.
3. **35–55 seconds:** The teammate safely lowers onto a mat. Show the possible-fall alert and accept it. Explain that accepting means a nurse took it, not that it was resolved.
4. **55–65 seconds:** Show an ordinary sit or bend without an alert, only if that behavior has actually been validated.
5. **65–90 seconds:** Stage a fall in the simulated room (**Simulate fall**), or run a scripted scenario. Keep the source label accurate. Show the measured delay and the after-action results.

Before presenting, rehearse three times from a cold start, check offline behavior (stop the camera: the source goes offline and the room becomes unknown), and prepare a recorded backup (`-Replay`). Label replay as replay. Do not ask judges to fall, or claim measured results, live face detection or clinical validation that the implementation does not support.

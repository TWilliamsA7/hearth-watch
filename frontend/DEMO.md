# Demo rehearsal

Use this script after the backend integration blockers in `COMPATIBILITY.md` are
resolved. Start the dashboard on port 5173 and the unified backend on port 8000.
Click **Connect to backend**, then choose **Simulation**, **Live camera**, or
**Recorded replay** as appropriate. Preview mode is example data only.

For simulation: choose a scenario, click **Load**, then **Play**. **Pause** and
**Reset** call the simulator. **Run again** resets and reruns the same seed; it is
still a simulation. At run end the evaluation report appears automatically.

1. **0–15 seconds:** “Hearth helps people review possible safety events in one room.” Start a normal simulated scenario and point out its simulation label.
2. **15–35 seconds:** Switch to live observations from a consenting teammate. Show the skeleton and face-presence indicator if detection works.
3. **35–55 seconds:** A teammate safely lowers onto a mat. Show the backend's possible-fall alert and acknowledge it. Explain that acknowledgement means the event was seen, not that it was resolved.
4. **55–65 seconds:** Show an ordinary sit or bend without an alert only if that behavior has actually been validated.
5. **65–90 seconds:** Run a simulated incident again or use an actual recorded replay feed. Keep the source label accurate. Show measured delay and the after-action results; explain how simulation lets the team inspect monitoring behavior.

Before presenting, rehearse three times from a cold start, check offline behavior,
and prepare a recorded backup. Label replay as replay. Do not ask judges to fall
or claim measured results, live face detection, or clinical validation that the
implementation does not support.

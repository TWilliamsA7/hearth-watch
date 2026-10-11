# Hearth dashboard

The single Hearth interface: React + TypeScript, built with Vite and served by the Hearth server itself, next to the API, on one port. Start it from the repository root:

```powershell
.\scripts\start.ps1          # then open http://127.0.0.1:8000/
.\scripts\start.ps1 -Dev     # also a hot-reload dev server on http://127.0.0.1:5173/
```

`start.ps1` installs packages and rebuilds `dist/` when sources changed. Working on the dashboard directly:

```powershell
npm.cmd install
npm.cmd test          # node --test over src/*.test.ts
npm.cmd run build     # generate types, typecheck, build dist/
npm.cmd run dev       # http://127.0.0.1:5173/, forwarding to HEARTH_BACKEND (default http://127.0.0.1:8000)
```

## Views

**Caregiver** (`/`, also `/nurses`)

- **Main feed:** the room in view from one source type at a time, kept separate: **Simulation** (the live 3D room), **Live camera** (the webcam's annotated frames) or **Recorded replay** (skeletons from a replay).
- **Simulation controls:** play/pause, speed, room settings and generation. The header stages a fall, inactivity, pacing or exit loitering for the resident clicked in the 3D room (or someone suitable); a selected resident can be reset.
- **Nurse session:** coverage (rooms and situations), on duty, alert sound. The session belongs to the tab and survives a reload.
- **Incident alerts:** every alert in the nurse's coverage, with reason, identity and recovery wording, accept/resolve, and counts with **Show resolved**. Before a session starts, the room's alerts are shown read-only.
- **Source health** and the **event log**; skeleton feed; posture measurements; caregiver response; today's appointments, with scheduling and cancelling.
- **Scenario controls and evaluation** (play icon): scripted scenarios, the completed run's report, earlier runs, the live room's report so far, and the batch summary.
- **Developer tools** (signal icon): the Sim Studio, the recorded-run player and the API reference.

**Patient** (`/room`)

- The SOS help button, with request and acknowledgement status, the help-gesture instruction and camera status.
- Appointments, with check-in and scheduling.
- **Face recognition check-in** (`/room#recognition`): opt-in enrollment from three camera views, linking appointments, three-match check-in, and withdrawal. It is disabled with an explanation unless the server enables recognition (see the root README).

A **room picker** in the header switches both views between the configured rooms.

## Files

| File | Purpose |
| --- | --- |
| `src/App.tsx` | Layout, views, room and source selection, simulation and developer controls. |
| `src/useRoomFeed.ts`, `src/feedState.ts` | Room WebSocket, reconnects, observation freshness and filtering by room and source type. |
| `src/useNurseSession.ts`, `src/NurseControls.tsx` | Nurse session, its WebSocket and source health, coverage form. |
| `src/IncidentList.tsx`, `src/CareResponsePanel.tsx`, `src/alertText.ts` | Alert lists and the shared wording for alerts and posture. |
| `src/SourceHealth.tsx` | Per-source online/stale state and reported posture. |
| `src/ResidentPanel.tsx`, `src/RecognitionPanel.tsx` | Patient view: help, appointments, recognition check-in. |
| `src/AppointmentPanel.tsx`, `src/useAppointments.ts` | Schedule, check-in, cancel. |
| `src/useLiveSimulation.ts`, `src/liveSimulation.ts`, `src/LiveSimulationView.tsx`, `src/RoomSettingsPanel.tsx` | Live 3D room via `@hearth/room-view`. |
| `src/SimulatorPanel.tsx` | Scripted scenarios and evaluation reports. |
| `src/LiveCameraView.tsx` | Annotated webcam frames from `/camera-preview`. |
| `src/RoomView.tsx`, `src/FeatureSpark.tsx` | Skeleton drawing and feature sparklines. |
| `src/api.ts`, `src/protocol.ts` | Endpoint calls; validation of server messages against the contract schemas. |
| `src/useRooms.ts` | Configured rooms and the gesture hold time. |
| `contracts/`, `src/generated/` | Schema snapshots and fixtures; types generated from them (`npm run generate:types`, do not edit). |
| `room-view/` | The renderer package, plus the Sim Studio and run player in `room-view/dev/` (built to `room-view/dist/`, served at `/studio/`). |
| `DESIGN.md`, `DEMO.md` | Visual design notes; demo rehearsal. |

No observation means unknown, even if heartbeats keep a source online. Observations expire three seconds after arrival; a stale flag from the server also hides an old reconnect snapshot. Simulation clocks can differ from wall time, so freshness never uses observation timestamps. Face `null`, `false` and `true` stay distinct. WebGL 2 is required for the 3D room.

## Removed or disabled, and why

Consolidating the dashboards removed these:

- **The server's standalone pages** (`/nurses`, `/room` and `/face-check-in` from `backend/hearth/api/static/`). Their features now live here: alert wording, overdue appointments, identity lines, resolved alerts, source health, sessions that survive a reload, every room, the gesture instruction, and enrollment. `/face-check-in` redirects to `/room#recognition`.
- **Preview mode and its example data.** That covers "Return to preview", the example schedule, "Hello, Margaret" and the "Acknowledge example" button. They showed invented data next to real controls. The dashboard is always served by a Hearth server now.
- **The "Connect to monitoring" address box.** It wrote an address that was never read back. The page talks to the server that served it.
- **`scripts/dashboard.ps1`, `scripts/demo.ps1`** (replaced by `start.ps1`).
- **`frontend/scripts/demo_backend.py`, `verify_demo.py`, `verify-branches.py`, `sync-branch-contracts.mjs`.** These were a stand-in backend and tools for comparing the separate team branches, all now merged. The fixtures they produced stay as test data.
- **The separate Sim Studio dev server on :5174.** It still works for Studio development (`npm run dev` in `room-view`); the built Studio is served at `/studio/`.

Kept on purpose:

- **The Recorded replay tab.** It shows any `replay` source; `start.ps1 -Replay` provides one, and the tab explains this when nothing is playing.
- **The alert-acknowledge API call in `api.ts`.** Unassigned alerts still support `/ack` on the server; the UI uses accept/resolve.

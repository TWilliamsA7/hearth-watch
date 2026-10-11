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

The header has a tab per view (each with its own address) and a Developer menu (Sim Studio, recorded-run player, API reference). The dashboard is for staff and watches the server's configured room.

**Monitor** (`/`)

- **Main view:** one source type at a time, kept separate: **Simulation** (the live 3D room), **Live camera** (the webcam's annotated frames) or **Recorded replay** (skeletons from a replay). It fills the window's remaining height.
- **Under the simulated room:** play/pause, speed, staging a fall, inactivity, pacing or exit loitering (for the selected resident, or someone suitable), resetting a resident or the room, **Live people** and **Room settings**.
- **Live people:** links the people a real camera (the Jetson, or a replay) sees into the simulated room, with the two-step camera calibration. Same server API as the Sim Studio's panel.
- **Person inspector:** click a resident in the 3D room (or a linked live person), or pick someone a camera or replay reports. It shows what they're doing and what the monitor sees for them: posture, torso angle, head and hip height, face, and their open alert.
- **Alerts:** every alert in the room from any source, whichever view is showing. Acknowledge, then resolve; show resolved; optional sound.
- **Sources** (each source's latest run: online, stale, posture) and the **event log**.

**Scenarios** (`/scenarios`): scripted scenarios and evaluation: the completed run's report, earlier runs, the live room's report so far, and the batch summary.

**Recognition** (`/recognition`): staff-run, opt-in resident enrollment from three camera views, a recognition test, and withdrawal. If recognition can't run, it says why.

Popovers (Room settings, Live people, Developer) close on a click elsewhere or Escape.

## Files

| File | Purpose |
| --- | --- |
| `src/App.tsx` | Header, tabs and their addresses. |
| `src/MonitorView.tsx` | The Monitor tab: main view, simulation controls, side column. |
| `src/useRoomFeed.ts`, `src/feedState.ts` | Room WebSocket, reconnects, observation freshness, per-run source health. |
| `src/AlertList.tsx`, `src/alertText.ts` | Alert list with acknowledge/resolve, and the shared alert and posture wording. |
| `src/PersonInspector.tsx`, `src/people.ts` | The selected person, and the people a camera or replay reports. |
| `src/LiveLinkPanel.tsx`, `src/useLiveLink.ts` | The live camera link and calibration. |
| `src/SourceHealth.tsx` | Per-source online/stale state and reported posture. |
| `src/RecognitionView.tsx` | The Recognition tab. |
| `src/Popover.tsx` | Panels that close on an outside click or Escape. |
| `src/useLiveSimulation.ts`, `src/liveSimulation.ts`, `src/LiveSimulationView.tsx`, `src/RoomSettingsPanel.tsx` | Live 3D room via `@hearth/room-view`. |
| `src/SimulatorPanel.tsx` | Scripted scenarios and evaluation reports. |
| `src/LiveCameraView.tsx` | Annotated webcam frames from `/camera-preview`. |
| `src/RoomView.tsx`, `src/FeatureSpark.tsx` | Skeleton drawing and feature sparklines. |
| `src/api.ts`, `src/protocol.ts` | Endpoint calls; validation of server messages against the contract schemas. |
| `src/useRooms.ts` | The server's configured room. |
| `contracts/`, `src/generated/` | Schema snapshots and fixtures; types generated from them (`npm run generate:types`, do not edit). |
| `room-view/` | The renderer package, plus the Sim Studio and run player in `room-view/dev/` (built to `room-view/dist/`, served at `/studio/`). |
| `DESIGN.md`, `DEMO.md` | Visual design notes; demo rehearsal. |

No observation means unknown, even if heartbeats keep a source online. Observations expire three seconds after arrival; a stale flag from the server also hides an old reconnect snapshot. Simulation clocks can differ from wall time, so freshness never uses observation timestamps. Face `null`, `false` and `true` stay distinct. WebGL 2 is required for the 3D room.

## Removed, and why

- **Patient-facing UI** (the patient view, SOS button and appointment panels). The dashboard is for staff. Appointments and `POST /help-requests` remain server APIs; their alerts still appear in the alert list.
- **The nurse-session layer** (shifts, on duty, display names, room coverage, the room picker, exclusive acceptance in the UI). Without authentication or persistence it added setup but no real capability. The server's nurse API and its tests remain. Alerts are acknowledged and resolved directly.
- **Torso-angle and hip-position cards, the skeleton card.** One number for a room with several people was meaningless; the person inspector shows them per person.
- **The server's standalone pages** (`/nurses`, `/room`, `/face-check-in`). They redirect to `/` and `/recognition`.
- **Preview mode and its example data, the unused address box.**
- **`scripts/dashboard.ps1`, `scripts/demo.ps1`** (replaced by `start.ps1`), and the frontend's branch-sync and stand-in backend scripts (the branches are merged; their fixtures stay as test data).

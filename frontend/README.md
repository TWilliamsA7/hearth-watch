# Hearth dashboard

React + TypeScript dashboard for Room 01. Starts in Simulation mode connected to http://127.0.0.1:8000, using the existing live 3D room from `room-view`.
Use Connection settings to choose a different server or return to the static UI preview. Choose **Live
camera**, **Simulation**, or **Recorded replay** to keep source types separate.

## Start

With Node.js 22 installed, open a PowerShell terminal in `frontend`:

```powershell
npm.cmd install
npm.cmd run dev
```

Open the address Vite prints, normally http://127.0.0.1:5173. Keep the server on
port 5173: the monitoring branch permits this origin. The default backend address
is http://127.0.0.1:8000. Enter the laptop's backend address if it runs elsewhere.

```powershell
npm.cmd test
npm.cmd run build
```

The dashboard uses local fonts/assets and works without an external service once
dependencies are installed. Generated files and the npm lockfile are checked in.

## Files and responsibilities

| File | Purpose |
| --- | --- |
| `src/main.tsx` | Starts React and loads CSS. |
| `src/App.tsx` | Dashboard layout, source selection, preview states, alert acknowledgement. |
| `src/RoomView.tsx` | Draws observed skeletons or model bounding boxes in camera coordinates. |
| `src/ResidentPanel.tsx` | Resident help requests with confirmed server receipt and explicit preview states. |
| `src/useRoomFeed.ts` | Opens the monitoring WebSocket, reconnects, and cleans up connections. |
| `src/feedState.ts` | Keeps observations, health, alerts, and events separate; handles freshness and filtering. |
| `src/protocol.ts` | Validates incoming messages against the teammates' JSON Schemas. |
| `src/api.ts` | Calls the exact monitoring, simulator, and evaluation endpoints. |
| `src/SimulatorPanel.tsx` | Scenario controls and measured evaluation/after-action results. |
| `src/demoData.ts` | Example data used only in preview mode. |
| `src/styles.css` | Layout, colors, mobile behavior. |
| `contracts/` | Schema snapshots and synthetic integration fixtures with reviewed branch revisions. |
| `src/generated/` | TypeScript generated from schema snapshots; do not edit by hand. |
| `src/feedState.test.ts` | Contract, isolation, freshness, alert, and endpoint regression checks. |
| `COMPATIBILITY.md` | Verified integrations and remaining backend merge blockers. |
| `DEMO.md` | Demo preparation and rehearsal. |
| `DESIGN.md` | Visual design, palette, and bringing Adobe exports into the app. |
| `src/Icon.tsx` | Small local SVG interface icons. |

## Backend interfaces

- `WS /ws/room/room-01`: initial state and ongoing observations, source health,
  events, and authoritative alert updates. No raw video is requested.
- `POST /alerts/{alert_id}/ack`: acknowledgement. The alert changes only when the
  backend broadcasts the updated status; no optimistic acknowledgement or resolution.
- `POST /help-requests`: resident help request with `request_id` and `room_id`.
  Manual alerts are displayed alongside the selected observation source type.
  The UI confirms receipt only; it does not promise a caregiver has been dispatched.
- `GET /sim/scenarios`, `GET /sim/status`: available scenarios and controller state.
- `POST /sim/load`: JSON `{ "scenario_id": "possible-fall-01" }`.
- `POST /sim/play`, `/sim/pause`, `/sim/reset`: controller actions.
- **Run again** calls reset then play, preserving the simulation seed. This is
  labeled simulation, not recorded replay.
- `GET /eval/runs/{run_id}`: completed-run report, fetched automatically at run end.
- `GET /eval/summary`: optional saved batch summary, fetched by its button.

No current observation means unknown, even if heartbeats keep the source online.
Observations expire after three seconds of arrival-time silence. A backend stale
flag also hides an old reconnect snapshot. Simulation clocks can differ from wall
time, so the frontend does not calculate freshness from observation timestamps.
Face `null`, `false`, and `true` remain different. Low-confidence keypoints are
hidden; unknown confidence is shown with dashed connections.

## Updating contracts

After fetching reviewed teammate changes:

```powershell
npm.cmd run sync:contracts
npm.cmd run generate:types
npm.cmd test
npm.cmd run build
```

`sync:contracts` reads the monitoring and simulation branches without switching
branches. After the team has merged a unified shared contract and simulator API
schema into this checkout, use `npm.cmd run sync:contracts -- --local` instead.
Commit schema snapshots and generated types together. This dashboard accepts each
current branch's output schema; it does not make their Python schemas interchangeable.

To repeat the real branch API checks, install `fastapi`, `httpx`, `pydantic>=2.7,<3`,
and `pyyaml` in a Python environment, then run
`python scripts/verify-branches.py` from `frontend`. The script reads branch files
into temporary directories and regenerates synthetic API fixtures. It does not
test physical camera inference or merge branches.

## Live 3D simulation

Start the standard monitoring backend from the repository root in a configured Python environment:

```powershell
python -m pip install -e shared/python -e backend
python -m uvicorn hearth.api.app:app --host 127.0.0.1 --port 8000
```

In another terminal, run `npm.cmd install` then `npm.cmd run dev -- --strictPort` from `frontend`. Open http://127.0.0.1:5173. The dashboard automatically connects to the live simulation. When no room exists, it generates six residents in a furnished room and starts playback. An existing run is reused without changing its paused/playing state.

Click a resident to select them, drag to orbit, and scroll to zoom. **Simulate fall** triggers a fall for the selected resident, or lets the engine choose; recovery occurs after 30 seconds on the floor. **Reset** resets the same room and resumes playback. **Play/Pause** controls the live engine.

The main feed shows simulated world positions. The skeleton, measurements, events, and incident alerts still come from the monitoring feed and may differ from ground truth. Live camera and Recorded replay retain their observation views. Scripted scenario controls remain separate: playing a scripted scenario pauses the live room, and its Play button resumes it.

The backend must remain running. If it is unavailable, the main feed shows a disconnected state and reconnects automatically. Use **Retry connection** after a request failure. WebGL 2 and browser hardware acceleration are required for the 3D view. The mannequin asset and its license are bundled locally in `room-view/assets/models`.
### Wrong-server error (404)

A 404 on `/sim/live/status` means the connected server does not expose the live simulation API. Connect to the standard `hearth.api.app:app` backend rather than an older running demo server. If port 8000 is already occupied, run the standard backend with `--port 8001`, then set **Connection settings → Return to preview → Monitoring server address** to `http://127.0.0.1:8001` and choose **Connect to monitoring**. The dashboard remembers the last connected address across reloads.
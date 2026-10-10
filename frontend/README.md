# Hearth dashboard

React + TypeScript dashboard for Room 01. Starts in a clearly labeled UI preview.
Use **Connect to backend** to receive real monitoring messages. Choose **Live
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

## Demo backend in this checkout

The normal backend app still lacks simulator routes. The separate demo entry point
connects the team's existing services without editing their owned folders.

For this laptop, open a terminal in the repository root and run:

```powershell
$env:PYTHONPATH="$PWD\frontend\.compat-python"
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" frontend/scripts/demo_backend.py
```

This uses the existing local Python dependencies (plus the installed server
packages). For a teammate with a configured Python environment instead:

```powershell
python -m pip install -e shared/python -e backend
python frontend/scripts/demo_backend.py
```

In a second terminal, run the frontend:

```powershell
cd frontend
$env:PATH="C:\Program Files\nodejs;" + $env:PATH
npm.cmd run dev -- --strictPort
```

Node 22 is installed here but was missing from this session's PATH; the line above
makes it available in that terminal. Dependencies already exist, so another install
is unnecessary unless packages change. Open http://127.0.0.1:5173, use **Connection
settings → Connect to monitoring**, then select **Simulation**. Choose **Fall while
walking, stays on the floor**, **Load**, and **Play**. The report appears at the end.
Use **Patient → SOS**, then **Caregiver** to review and acknowledge it.
These actions create local demo records, not messages to an outside care service.

Keep both terminals open. Ctrl+C stops each server. Data and completed reports
are held in memory; restarting the backend clears them. Port conflicts usually
mean an earlier server is still running. See COMPATIBILITY.md for verified behavior
and the remaining team-owned integration work.

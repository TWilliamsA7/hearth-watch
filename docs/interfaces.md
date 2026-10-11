# Component interfaces

## 2.2 Monitor ingestion

`POST /ingest/observations` takes an `ObservationBatch`; `POST /ingest/heartbeat`
takes a `Heartbeat`. Both use the Pydantic contract in `shared/python/hearth_schema`.
The API injects an asynchronous `ingest(items) -> events` function into the simulator.
It routes individual heartbeats and observations through the same `MonitorService`
used by the camera. It never supplies ground truth, labels, world poses, or expected
alerts to the monitor.

Resident observations sharing one timestamp keep independent state by anonymous
`person_id`. This is a session track, not facial recognition. Source health is shared
by room, source type, source ID, scenario ID and run ID. Heartbeats must carry the same
scenario/run context as observations. A live reset/replay creates a new run ID.

`hearth.monitor:create_monitor` returns a single-record adapter for headless evaluation:
`ingest(Observation | Heartbeat) -> list[Event]`, and
`advance_to(datetime) -> list[Event]`. It runs the real monitoring service with a
virtual receipt clock. The batch runner additionally assigns each resident a separate
adapter. The factory loads the normal monitor config, or `HEARTH_CONFIG` if set.

## 2.4 Clocks and uncertainty

Posture and gesture holds use observation timestamps. Live health uses arrival time
on the backend; simulation playback can accelerate or pause without changing the
meaning of a three-second floor hold. Paused simulations send heartbeats except
during a scripted outage. Stale observations invalidate pending evidence; heartbeat
alone never establishes a safe posture or recovery. Headless evaluation advances
its injected clock through silence, so outage checks do not use wall time.

## 3.3 Simulation control

The FastAPI app mounts `/sim/scenarios`, `/sim/status`, `/sim/world`, and POST
`/sim/generate`, `/sim/load`, `/sim/play`, `/sim/pause`, `/sim/reset`, `/sim/speed`.
Load takes `{"scenario_id":"possible-fall-01","seed":42}`; speed takes
`{"speed":4}`. Replay a finished run by playing again, or reset and play.
`/sim/status` truth and `/sim/world` are dashboard-only actual-state data.
The monitor's observed feed is `/ws/room/{room_id}`; nurse alerts use
`/ws/nurse/{nurse_id}`. Background simulation tasks stop on backend shutdown.

## 3.4 Evaluation

`GET /eval/runs` lists completed live simulation runs. `/eval/runs/{run_id}` returns
their after-action report. `/eval/summary` reads the latest batch output (404 until
one exists). `python -m hearth.eval batch --seeds 1` writes JSON and Markdown results
under `recordings/eval` at the repository root by default. The summary API reads that
default directory. Reports describe synthetic scenarios, not clinical accuracy.

The dashboard (`frontend/`, served at `/`, `/room` and `/nurses`) drives the scripted
simulator, the live room and the evaluation reports; the Sim Studio is at `/studio/`
and the recorded-run player at `/studio/runs`, all from the same server. Development
origins on ports 5173 (`start.ps1 -Dev`) and 5174 (Studio dev server) are allowed by
the default monitor config.

## Optional patient identity integration

Recognition adds metadata to confirmed demo incidents; anonymous `person_id` remains unchanged. The optional local `/face-demo/observe` envelope now accepts same-frame poseNet body evidence as well as the laptop YOLO path. The Jetson's `--face-id` sends face templates (never frames) to the token-authenticated `/face-demo/edge-embeddings`, which names that `person_id`'s fall incidents. See [the integration decision and edge/dashboard contract](recognition-integration.md) before connecting a Jetson source. Normal ingest, simulation, and evaluation do not require recognition.

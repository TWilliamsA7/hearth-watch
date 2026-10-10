# Dashboard integration review

Reviewed remote branches on October 10, 2026:

| Role | Branch | Revision |
| --- | --- | --- |
| Edge/CV | `origin/gaston/edge` | `879771bac173fa8ae0fcdc2624c43ac04cf02b8e` |
| Monitoring | `origin/monitoring` | `44564b04476e169a8b7d6858a28b46f583b13607` |
| Simulator/evaluation | `origin/feat/simulation` | `7fac21de459c76b152e82c05368f3c633597b126` |

`feat/scenario-generation` also exists. This integration targets the simulator
API and schema on `feat/simulation`, not its separate scenario-generation changes.

## Verified

The dashboard consumes the actual monitoring `observation`, `source_status`,
`event`, and `alert` messages. Its validator supports both the monitoring and
simulator output schemas. The view supports normalized skeleton keypoints,
nullable keypoint confidence, optional posture-model bounding boxes, and simulator
anonymous track ids. It uses Room 01 and isolates live, simulation, and replay data.

The Python verification script exercised the current edge serializer against the
actual monitoring FastAPI app: observation ingestion, room WebSocket output,
possible-fall alert creation, HTTP acknowledgement, and stale-source status passed.
Synthetic pixel/pose fixtures were used; this is not a hardware or accuracy test.

The simulator routes were exercised in a separate FastAPI app: scenario listing,
load, play, pause, reset, status, and completed-run evaluation passed. That test
used an empty monitoring callback to verify API responses; it does not prove the
simulator and real monitor work together. These API outputs are saved as fixtures
and validated by the dashboard tests.

## Confirmed backend integration blockers

1. **Simulator routes are not mounted.** `backend/hearth/api/app.py` on monitoring
   does not create a `SimController` or include `/sim` and `/eval` routers. An actual
   request to `/sim/scenarios` returns 404. The dashboard shows this error and does
   not fabricate scenarios or results.
2. **The shared schemas diverged.** Simulator `Observation`, `Event`, and
   `GroundTruth` include `person_id`; the monitoring models reject unknown fields
   and lack that field. Monitoring has additional `run_id`, prediction, alert,
   and nurse fields absent from the simulator branch. An actual serialized
   simulator observation is rejected by monitoring because of `person_id`.
   Role 2 should reconcile the shared models with role 3 and regenerate schemas.
   Simply copying one branch's schema over the other would lose required features.
3. **The simulator needs an ingestion adapter.** `SimController` expects an async
   callback accepting observations AND heartbeats and returning newly generated
   monitor Events. `MonitorService.ingest()` accepts observations only and returns
   ingestion counts; `MonitorService.heartbeat()` is separate. Passing `ingest`
   directly will not meet this interface. The adapter must route each record type,
   preserve order, return actual new events, and use a consistent run id for
   reset isolation. Health-ticker events also need forwarding to `controller.observe()`
   for outage evaluation. Close the controller in the app lifespan on shutdown.

Start with a single-person scenario. Multi-person monitoring needs explicit
tracking support in the monitoring core, beyond drawing multiple simulator tracks.

## Team completion checklist

- Role 2 + role 3 unify the shared contract and create the ingestion adapter.
- Mount the real simulator/evaluation routers on the monitoring app, with lifecycle
  cleanup and health-event forwarding.
- Refresh frontend contracts/types and rerun tests after that merge.
- Run the unified backend; in the dashboard select Simulation, load
  `possible-fall-01`, play, inspect the backend alert, acknowledge it, and inspect
  the completed-run report.
- Connect the Jetson, select Live camera, and validate actual keypoints/face status.
- Stop the source: observations must become unknown within three seconds, and
  the monitor must report source health truthfully.
- Test a backend restart/reconnect, then rehearse the complete demo three times.

No teammate branch was merged or edited during this review. Frontend changes and
the `.gitignore` entries for temporary verification dependencies are local to the
dashboard checkout. Browser visual verification was not completed because the
local preview was unreachable from the sandbox and permission to run it outside
the sandbox was declined.

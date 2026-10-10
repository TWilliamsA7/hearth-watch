# Current dashboard integration — October 10, 2026

This separate checkout was initially clean on `feat/dashboard`, at `137c8af`.
Eleven uncommitted dashboard files were restored from the OneDrive checkout.
The original folder and downloaded design HTML were left untouched.

## Runnable demo

`frontend/scripts/demo_backend.py` composes the existing monitoring app,
SimController, simulator routes, and evaluation routes. No files under edge/,
shared/, backend/, or scenarios/ were edited. This is a demo entry point for the
team to review; the default `hearth.api.app` still does not mount /sim or /eval.

The adapter routes observations and heartbeats in order, stamps the controller's
run id onto both, forwards real monitor events (including health events), retires
previous-run health on reset, and closes the controller on shutdown. Only the six
single-person scenarios without resident track IDs are offered. Multiple-person
monitoring remains a teammate task. Completed reports live in memory and disappear
when the demo backend stops. No invented detection results are used.

This checkout's shared Python models already include person_id. Earlier branch
snapshots differ; the dashboard now validates against both those snapshots and
this checkout's unified schema and simulator API schema. Refresh the two current
schema snapshots together after team changes. Do not assume these represent a
new remote revision: no remote fetch or team branch merge was performed here.

## Verified

- Frontend tests and production build.
- Desktop visual review of preview, offline, Resident, caregiver alerts, and evaluation.
- Narrow-screen Resident layout.
- Browser WebSocket connection, real local help request, caregiver acknowledgement.
- Browser possible-fall-01 completion: one true positive, zero false alarms,
  zero misses, 4.07-second simulated detection delay in the reviewed run.
- Deterministic real-service checks for detection, evaluation, reset run isolation,
  paused heartbeats, help-request retry deduplication, acknowledgement, and cleanup.
- Synthetic messages captured from the composed backend pass frontend validation.

## Still requires the team

- Review and adopt the demo composition in the official backend entry point.
- Add person-level monitoring before enabling multiple-person scenarios.
- Verify Jetson hardware, real live/replay feeds, backend restart, and three rehearsals.
- No live-camera accuracy or clinical validation was established by these checks.

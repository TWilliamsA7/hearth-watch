# Recognition behavior audit — October 11, 2026

Scope: current `monitoring` working tree, local staged demo. This is an implementation audit, not a security certification or camera-accuracy evaluation.

| Intended behavior | Finding | Evidence / limit |
|---|---|---|
| Explicit optional enrollment | Implemented | `api/face_demo.py` and `monitor/face_demo.py`; false consent rejected; three enrollment views |
| Independent named-incident permission | Implemented; UI corrected in this audit | `identify_alerts` defaults false; enrollment page now has a separate unchecked choice |
| Appointment-only recognition | Corrected in this audit | Fresh private match can check in an explicitly linked appointment without exposing a name on incident alerts |
| Recognized patient's name on possible fall | Implemented | exact-observation labels in `MonitorService.ingest`; test covers face turning away while same body persists |
| Recognized patient's name on held fist | Implemented and tightened | requires named-alert permission and hand-box center inside matched body; unassociated hand still alerts anonymously |
| Anonymous alerts when not enrolled/uncertain | Implemented | body loss, ambiguity, expiry, changed track, nonconsent; ordinary ingest still works |
| No movement/sighting history with patient names | Implemented in recognition feature | name is attached to incident metadata; no persistent recognition-sighting log; anonymous observations still exist for monitoring |
| Patient-controlled withdrawal | Implemented | profile deletion removes encrypted enrollment and invalidates sessions; prior incident labels remain |
| Separate appointment permissions | Implemented | explicit links only, correct room, time window, rotated token guard; link ids now public without tokens and links can be deleted |
| Enrollment encrypted at rest | Implemented | Fernet payload includes face template, alias, consent and links; opaque id remains plaintext |
| Raw camera frames not saved | Implemented by this feature | frame processed in memory; no image/video writes in recognition path; deployment/server logging must preserve this |
| All data encrypted / anonymous | NOT implemented / not an accurate claim | appointment storage and incident feed are outside enrollment encryption; biometric templates are pseudonymous |
| Staff-only access | NOT implemented as authentication | nurse subscriptions filter workflow; ids, global endpoints, and WebSockets are not authenticated authorization |
| Access audit log | NOT implemented | incident lifecycle events are not an audit of who accessed patient data |
| Defined retention for all named incidents | NOT implemented | enrollment has 24-hour expiry checked on access; events are bounded in memory, fall alerts are in memory, appointments persist; no time-based incident purge policy |
| Jetson recognition deployment | NOT implemented end to end | monitor accepts poseNet body evidence; hardware runtime and encrypted bridge remain separate work |
| No monitoring delay from inference | Partial | API inference is off event loop and bounded; laptop capture client still waits for request/response, so camera throughput needs measurement |

New regression coverage is in `backend/tests/test_recognition_safety.py`; broader tests cover consent, matching uncertainty, appointment behavior, poseNet compatibility, camera isolation, and nurse delivery. Tests use synthetic embeddings and do not measure real-world false matches.

Use [the frontend-agent prompt](frontend-recognition-agent-prompt.md) for UI work. It explicitly identifies unsupported security/deployment claims so they are not turned into decorative settings or misleading status badges.

Validation after these changes: **84 relevant backend tests passed** across recognition, appointment, API, nurse workflow, architecture, model-adapter, simulator-adapter, and recorded Jetson contract suites. JavaScript syntax validation and `git diff --check` also passed. No live camera accuracy or Jetson hardware test was performed.

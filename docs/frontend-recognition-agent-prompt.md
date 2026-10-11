# Frontend agent prompt: optional patient recognition

Copy the text below into the frontend AI agent. This is a task specification based on the monitoring working tree audited on October 11, 2026. Backend changes must be present in the server being tested; do not assume a different branch or an old running server has these endpoints.

---

You are implementing the frontend integration for HearthWatch's optional patient recognition demo. Do the work in the existing frontend branch/worktree. Do not switch, merge, reset, commit, push, or modify the monitoring branch. Inspect the frontend's existing architecture and reuse its components, styles, API client, and tests. This demo uses consenting participants playing fictional patients; do not present it as deployed clinical software.

## Outcome

A patient can opt in to recognition, independently choose whether their demo name appears on safety alerts, and explicitly link appointments for camera check-in. A nurse can see a named **possible fall** or **help request** when the backend supplies confirmed incident identity. Declining recognition must not disable unnamed alerts. The product supports response to an incident, not a search history of a person's movements.

Implement the UI, run its tests/build, and report actual backend integration separately from mock tests. If the connected server lacks a documented capability, show unavailable instead of inventing success. Do not add backend capabilities or weaken origin checks as a shortcut.

## Backend facts you must preserve

- `GET /rooms` returns `{room_ids: string[], gesture_hold_seconds: number}`.
- YuNet detects faces; SFace compares enrolled templates. The frontend must never decide identity, posture, fall severity, or whether a gesture satisfies its hold time.
- Face recognition is optional and disabled by default. Enrollment requires three views, explicit consent, and a backend encryption key. Templates and enrollment aliases are encrypted at rest; this is not end-to-end encryption of the application.
- `identify_alerts` is a separate enrollment permission, default false. Explicit appointment links are separate from this flag.
- Recognition uses three consistent frames, rejects ambiguous matches, and may retain a confirmed name for up to 8 seconds of continuous single-body tracking when the face turns away. Track loss, another person, stale evidence, expiry, or withdrawal clears identity.
- A camera help gesture is named only if its hand box belongs to the matched body box. A manual room help button remains unnamed; do not attach the last recognized person to it.
- Raw face frames and embeddings are not part of the nurse WebSocket feed. This feature does not save camera frames. Do not add storage, analytics payloads, screenshots, video recording, or console logging of face frames/templates/tokens.
- Names are attached to incidents, not normal observation messages. `person_id` is an anonymous sensor track; it is NOT a patient profile id. Simulation names and room occupants cannot establish identity.
- Profiles expire after 24 hours; expired records are removed on the next profile access/recognition attempt, not by an exact-time deletion scheduler.
- The laptop path is implemented. Monitoring accepts optional poseNet body evidence, but the Logitech/Jetson recognition bridge and hardware performance are NOT verified.

## 1. Patient view and enrollment entry point

Add a clear **Optional recognition** card to the real Patient view (inspect `frontend/src/ResidentPanel.tsx` if it exists):

> Optional recognition helps your care team identify you when you request assistance or a possible fall is detected. You can choose whether your name appears on safety alerts and separately link appointments for check-in. Alerts still work if you decline.

Use patient language, not a separate "volunteer product." Keep a visible demo designation. Do not auto-open a camera when the dashboard loads or a patient tab is selected.

**Important origin constraint:** all `/face-demo/*` routes are loopback-only and same-origin. A React development server on port 5173 must NOT fetch them cross-origin from port 8002 or proxy them into a generally accessible remote enrollment service. For the current integration, provide a working **Manage enrollment** link that opens the configured local backend's `/face-check-in` page. This existing page provides enrollment, separate unchecked consent choices, profile status, appointment linking, recognition testing, and withdrawal. Do not present a failed cross-origin request as "not enrolled." If the backend is remote, explain that enrollment must be opened locally on its host; disable unsupported controls.

If the React build is already served from the exact same local backend origin, native controls may use the APIs below. Otherwise the real backend page link is the required working integration; do not fake inline controls. Explain this tradeoff in your final report.

## 2. Same-origin enrollment API contract

All mutating `/face-demo` requests require `X-Hearth-Face-Demo: 1`; JSON requests also use `Content-Type: application/json`. This header is an origin guard, NOT staff authentication. Do not set or spoof an Origin header in browser code.

| Method/path | Request | Success |
|---|---|---|
| GET `/face-demo/status` | none | `{enabled, message, body_detectors:["yolo","posenet"], transport:"loopback_only", identity_ttl_seconds:8}` |
| GET `/face-demo/profiles` | none | Array of public profile records |
| POST `/face-demo/enroll` | `{alias, consent:true, identify_alerts:boolean, frames:[a,b,c], appointment_id:null or id}` | Public profile |
| DELETE `/face-demo/profiles/{profile_id}` | none | `{deleted:true}`; removes enrollment and invalidates recognition sessions |
| POST `/face-demo/profiles/{profile_id}/appointments/{appointment_id}` | none | `{linked:true}` |
| DELETE `/face-demo/profiles/{profile_id}/appointments/{appointment_id}` | none | `{unlinked:true}`; does not cancel the appointment or erase an earlier check-in |
| POST `/face-demo/sessions` | `{room_id}` | `{session_id}` |
| DELETE `/face-demo/sessions/{session_id}` | none | `{stopped:true}` |
| POST `/face-demo/recognize` | `{session_id, frame}` | `status` plus `matches`, or confirmed `profile`/`room_id`; successful arrival adds `checked_in` and `checked_in_appointments` |

Example public profile shape (illustrative values):

```ts
type FaceProfile = {
  profile_id: string;
  alias: string;
  consent_version: string; // currently demo-patient-v2
  consented_at: number;   // Unix seconds, multiply by 1000 for JS Date
  expires_at: number;     // Unix seconds
  identify_alerts?: boolean; // absent/false means named alerts off
  appointment_id: string | null; // legacy first link
  appointment_ids: string[];     // all explicit links, no tokens
};
```

Treat model output as data and render names through normal escaped React text. Do not store profiles or recognition results as a movement history. Never assume missing profile fields imply permission.

For native enrollment controls, use TWO initially unchecked choices:
1. "I opt in to face recognition for this demo. Only appointments I explicitly link may be checked in." (`consent`)
2. "Also show my demo name on possible-fall and camera help-signal alerts." (`identify_alerts`, optional)

Do not force choice 2 to enroll or check in. The current API does not provide an in-place named-alert preference update: to change that enrollment permission, explain withdrawal and reenrollment; do not create a toggle that changes only frontend state. Link/unlink appointments without recapturing the face.

An alias has 1–40 characters. Enrollment requires exactly three separate base64 JPEG/PNG frames without a data-URL prefix. Each decoded image must be at most 800,000 bytes, between 160x120 and 1280x960; 640x480 JPEG is suitable. Start the camera only on an explicit button click, show its active state, stop tracks on stop/unmount/pagehide, and discard frames after the request. Stop the enrollment camera before running the fall-model camera process so two clients do not compete for the webcam.

Recognition states include `confirming`, `recognized`, `unknown`, `new_frame_required`; errors may indicate no face, poor lighting, expired session, or disabled setup. `/observe`, used by the camera pipeline rather than this dashboard, additionally returns `tracked`, `face_not_visible`, or `identity_unconfirmed`. Do not use the diagnostic response as a live patient-location roster.

Handle 403 as local/same-origin restriction, 422 as validation/recognition feedback, and 503 as disabled/unavailable or busy (read the message). Do not queue a backlog of camera frames or retry a mutation as successful. A failed deletion must not show "deleted." Refresh state after successful operations. No face key or model weights belong in frontend code.

## 3. Nurse safety alerts

Use the existing nurse workflow, not a new recognition-specific alert rail:

- POST `/nurse-sessions` with `{display_name, room_ids, alert_kinds, on_duty}`. Kinds include `possible_fall`, `help_requested`, `appointment_overdue`, and `source_offline`.
- Subscribe to `/ws/nurse/{nurse_id}`; apply the initial `nurse_snapshot` and subsequent alert/event/source messages. Reconnect using GET `/nurse-sessions/{nurse_id}/snapshot`; a 404 means the session expired.
- POST `/alerts/{alert_id}/accept` with `{nurse_id}` for nurse acceptance. This is preferable to the generic acknowledgment endpoint because it records assignment.
- POST `/alerts/{alert_id}/resolve` with `{nurse_id}`. Only the assigned nurse can resolve their accepted alert. Handle conflicts from the server; do not resolve optimistically on face recognition, posture recovery, or arrival.

Alert identity is in `alert.metadata`:

```json
{
  "resident_label": "Alex",
  "patient_profile_id": "opaque-enrollment-id",
  "identity_method": "recent_face_body_track",
  "identity_demo": true
}
```

Valid `identity_method` values are `face_match` and `recent_face_body_track`. A safe display helper should require an edge `possible_fall`/`help_requested` alert, `identity_demo === true`, a nonempty string name/profile id, and a recognized method. Otherwise show **Identity unconfirmed**. Do not derive a name from the selected patient, room, previous alert, `source`, or anonymous `person_id`.

Display examples:

- **Possible fall — Alex**, Room 01, event time, **Face match confirmed** or **Recent face match associated with body track**, and **Accept request**.
- **Help requested — Alex**, with the actual gesture/request description.
- **Possible fall — Identity unconfirmed** when no authorized match is attached. Preserve prominence and nurse actions; anonymity must not downgrade the incident.

Always keep "possible" for a fall. Do not label match scores as percent certainty. The name belongs to the incident at its recorded time; an 8-second matching timeout does not retroactively erase that incident's label. Withdrawal stops new matching but does not erase previous alerts.

Do not label manual SOS/button requests, equipment-offline alerts, or simulated residents using a recent camera match. Deduplicate updates by alert_id. Use the supplied accepted_by/accepted_by_name/status/recovered_at fields. Recovery is information for the nurse, not automatic resolution.

Do not create live identity maps, movement timelines, a searchable sighting archive, "last seen everywhere" screens, a watchlist, or a gallery of unknown people. An incident timeline is appropriate.

## 4. Appointment UI

Connect the real `AppointmentPanel` to the backend instead of leaving example data looking live:

- GET `/appointments`: list real backend records. Refresh after actions and use bounded polling if necessary; there is no general appointment WebSocket message type.
- POST `/appointments`: `{resident_label, room_id, scheduled_at:<ISO with timezone>, grace_seconds:900, demo:true}`. Convert local form input to an explicit timestamp; show times in the user's timezone.
- PUT `/appointments/{id}` with the same fields reschedules. It rotates the check-in token; unlink/relink recognition to the new schedule explicitly.
- POST `/appointments/{id}/cancel` cancels.
- POST `/appointments/{id}/check-in` with `{method:"manual"}` is the demo manual alternative. Label it as manual confirmation; never call it automatically because a face was detected or the user clicked a patient card.

Appointment status is `scheduled | overdue | arrived | cancelled`. Confirmed camera check-in has `check_in_method:"face_recognition_demo"`; do not fabricate this value or send it as a user-selected method. Recognition checks only explicitly linked open demo appointments in the session room, beginning 15 minutes BEFORE scheduled time. The default missed-arrival reminder is 15 minutes AFTER scheduled time (`grace_seconds:900`). Those are different windows. The backend's quick demo button uses 30 seconds ahead and a 15-second grace period; label accelerated demo timing clearly.

`appointment_overdue` means **arrival not confirmed**, not that the patient is missing or has fallen. Arrival may set recovered_at on the reminder, while nurse resolution remains separate. Tokens returned by appointment APIs must not be placed in URLs, analytics, logs, or local storage. Linking a profile uses appointment id, not an exposed check-in token.

## 5. Honest data and security copy

Use this wording or a plain-language equivalent:

> Face enrollment is optional. Enrollment templates are encrypted on the local backend. This demo does not save camera frames. Named incident alerts and appointments use the existing monitoring storage and are not covered by enrollment encryption. Withdrawing deletes enrollment and stops future matches; existing incident records remain.

Do not claim anonymity, clinical accuracy, liveness verification, HIPAA compliance, authenticated staff access, end-to-end encryption, automatic deletion of all data, or fully deployed Jetson recognition. Nurse room subscriptions and exclusive acceptance are demo workflow controls, NOT authorization. Authentication, access auditing, secure remote deployment, and timed incident retention remain backend work; do not simulate security with hidden UI controls.

## 6. Required validation and deliverable

Add focused frontend tests for:

- Named fall, named held-fist help request, recent-body-track wording, anonymous fall, missing/malformed identity metadata, and escaping a name containing markup.
- No fallback name from room, selected profile, anonymous person_id, manual request, or simulation.
- Recognition disabled/unreachable/403/busy, rejected consent, expired enrollment, failed deletion, and camera stop on unmount if native camera UI is used.
- Two independent consent choices; named alerts default off. Appointment-only opt-in does not enable incident naming.
- Linking/unlinking uses the documented endpoints and confirms server success. A rescheduled appointment does not pretend its old link is valid.
- Acceptance conflicts, recovery without automatic resolution, arrival without automatic resolution, and reconnect replacing the snapshot without duplicate alerts.
- Live appointments versus clearly marked preview fixtures; timezone conversion and the two separate 15-minute windows.

Run the frontend tests and production build. Perform a real-backend smoke test against the monitoring server if available; otherwise state that it remains unverified. Do not claim synthetic embeddings or mock responses prove face accuracy. Finish with changed files, test results, a short demo procedure, and any remaining backend/Jetson dependency. Do not commit or push unless the human explicitly authorizes it.

---

End of copy-paste prompt.

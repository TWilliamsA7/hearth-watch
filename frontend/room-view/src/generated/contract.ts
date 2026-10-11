/* Generated from shared/schema/hearth.schema.json by scripts/gen-types.mjs. Do not edit. */

/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "ServerMessage".
 */
export type ServerMessage =
  ObservationMessage | EventMessage | SourceStatusMessage | AlertMessage | NurseSnapshotMessage;

export interface HearthContract {
  Alert?: Alert;
  BoundingBox?: BoundingBox;
  Event?: Event;
  GroundTruth?: GroundTruth;
  HandGesturePrediction?: HandGesturePrediction;
  Heartbeat?: Heartbeat;
  HelpRequest?: HelpRequest;
  Keypoint?: Keypoint;
  NurseAction?: NurseAction;
  NursePreferences?: NursePreferences;
  NurseProfile?: NurseProfile;
  NurseSnapshot?: NurseSnapshot;
  Observation?: Observation;
  ObservationBatch?: ObservationBatch;
  PostureFeatures?: PostureFeatures;
  PosturePrediction?: PosturePrediction;
  RoomSnapshot?: RoomSnapshot;
  SourceStatus?: SourceStatus;
  AlertMessage?: AlertMessage;
  EventMessage?: EventMessage;
  NurseSnapshotMessage?: NurseSnapshotMessage;
  ObservationMessage?: ObservationMessage;
  SourceStatusMessage?: SourceStatusMessage;
  ServerMessage?: ServerMessage;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "Alert".
 */
export interface Alert {
  alert_id: string;
  event_id: string;
  kind: "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue";
  source: string;
  source_type: "edge" | "simulation" | "replay" | "manual";
  room_id: string;
  scenario_id: string | null;
  run_id: string | null;
  status: "open" | "acknowledged" | "resolved";
  severity: "warning" | "info";
  opened_at: string;
  acknowledged_at: string | null;
  resolved_at: string | null;
  recovered_at: string | null;
  accepted_by: string | null;
  accepted_by_name: string | null;
  metadata: {
    [k: string]: unknown;
  };
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "BoundingBox".
 */
export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "Event".
 */
export interface Event {
  schema_version: "1";
  event_id: string;
  timestamp: string;
  source: string;
  source_type: "edge" | "simulation" | "replay" | "manual";
  room_id: string;
  scenario_id: string | null;
  run_id: string | null;
  /**
   * Anonymous per-session track id of the person the event concerns (e.g. 'p1'). Not an identity. None = single-person source.
   */
  person_id: string | null;
  event_type:
    | "possible_fall"
    | "recovered"
    | "person_visible"
    | "person_not_visible"
    | "face_present"
    | "face_absent"
    | "source_offline"
    | "source_online"
    | "alert_acknowledged"
    | "alert_resolved"
    | "help_requested"
    | "appointment_overdue";
  confidence: number | null;
  metadata: {
    [k: string]: unknown;
  };
}
/**
 * One labelled segment of a simulated scenario. Times are seconds from scenario start.
 *
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "GroundTruth".
 */
export interface GroundTruth {
  schema_version: "1";
  scenario_id: string;
  seed: number;
  segment_id: string;
  /**
   * Which simulated resident this segment describes. None = single-resident scenario.
   */
  person_id: string | null;
  t_start: number;
  t_end: number;
  truth_label:
    | "walking"
    | "standing"
    | "sitting"
    | "bending"
    | "lying_intentional"
    | "fall"
    | "occluded"
    | "camera_outage"
    | "absent";
  expects_alert: boolean;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "HandGesturePrediction".
 */
export interface HandGesturePrediction {
  model_id: string;
  model_revision: string;
  label:
    | "like"
    | "palm"
    | "fist"
    | "peace"
    | "ok"
    | "one"
    | "three"
    | "four"
    | "call"
    | "point"
    | "grip"
    | "middle_finger"
    | "unknown";
  confidence: number | null;
  bbox: BoundingBox | null;
  detection_count: number;
  uncertainty_reason: ("no_detection" | "multiple_hands" | "unsupported_label") | null;
}
/**
 * Sent by every source about once per second; also carries edge performance metrics.
 *
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "Heartbeat".
 */
export interface Heartbeat {
  schema_version: "1";
  timestamp: string;
  source: string;
  source_type: "edge" | "simulation" | "replay" | "manual";
  room_id: string;
  scenario_id: string | null;
  run_id: string | null;
  fps: number | null;
  inference_ms: number | null;
  mem_used_mb: number | null;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "HelpRequest".
 */
export interface HelpRequest {
  request_id: string;
  room_id: string;
}
/**
 * Coordinates are normalized to the frame: (0, 0) top-left, (1, 1) bottom-right.
 *
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "Keypoint".
 */
export interface Keypoint {
  name:
    | "nose"
    | "left_eye"
    | "right_eye"
    | "left_ear"
    | "right_ear"
    | "left_shoulder"
    | "right_shoulder"
    | "left_elbow"
    | "right_elbow"
    | "left_wrist"
    | "right_wrist"
    | "left_hip"
    | "right_hip"
    | "left_knee"
    | "right_knee"
    | "left_ankle"
    | "right_ankle"
    | "neck";
  x: number;
  y: number;
  /**
   * Measured keypoint confidence, when available; null is unknown, not zero or certainty.
   */
  confidence: number | null;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "NurseAction".
 */
export interface NurseAction {
  nurse_id: string;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "NursePreferences".
 */
export interface NursePreferences {
  display_name: string;
  /**
   * @minItems 1
   * @maxItems 100
   */
  room_ids: [string, ...string[]];
  /**
   * @minItems 1
   * @maxItems 4
   */
  alert_kinds:
    | ["possible_fall" | "help_requested" | "source_offline" | "appointment_overdue"]
    | [
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue"
      ]
    | [
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue"
      ]
    | [
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue"
      ];
  on_duty: boolean;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "NurseProfile".
 */
export interface NurseProfile {
  display_name: string;
  /**
   * @minItems 1
   * @maxItems 100
   */
  room_ids: [string, ...string[]];
  /**
   * @minItems 1
   * @maxItems 4
   */
  alert_kinds:
    | ["possible_fall" | "help_requested" | "source_offline" | "appointment_overdue"]
    | [
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue"
      ]
    | [
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue"
      ]
    | [
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue",
        "possible_fall" | "help_requested" | "source_offline" | "appointment_overdue"
      ];
  on_duty: boolean;
  nurse_id: string;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "NurseSnapshot".
 */
export interface NurseSnapshot {
  nurse: NurseProfile;
  alerts: Alert[];
  events: Event[];
  sources: SourceStatus[];
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "SourceStatus".
 */
export interface SourceStatus {
  source: string;
  source_type: "edge" | "simulation" | "replay" | "manual";
  room_id: string;
  scenario_id: string | null;
  run_id: string | null;
  online: boolean;
  last_seen: string | null;
  last_observation_at: string | null;
  observation_stale: boolean;
  posture: "standing" | "sitting" | "fallen" | "unknown";
  posture_confidence: number | null;
  posture_model: string | null;
  fps: number | null;
  inference_ms: number | null;
  mem_used_mb: number | null;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "Observation".
 */
export interface Observation {
  schema_version: "1";
  obs_id: string;
  timestamp: string;
  /**
   * Device or generator id, e.g. 'jetson-01' or 'sim'.
   */
  source: string;
  source_type: "edge" | "simulation" | "replay" | "manual";
  room_id: string;
  scenario_id: string | null;
  /**
   * Unique replay/run id; change on reset.
   */
  run_id: string | null;
  /**
   * Anonymous per-session track id when a source reports several people (e.g. 'p1'). Not an identity. None = single-person source.
   */
  person_id: string | null;
  person_detected: boolean;
  keypoints: Keypoint[] | null;
  features: PostureFeatures | null;
  posture_prediction: PosturePrediction | null;
  hand_gesture_prediction: HandGesturePrediction | null;
  /**
   * None = face detection did not run on this frame.
   */
  face_present: boolean | null;
  face_confidence: number | null;
}
/**
 * Per-frame posture features. Every field is optional: missing = not measurable this frame.
 *
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "PostureFeatures".
 */
export interface PostureFeatures {
  /**
   * Neck-to-hip angle from vertical. 0 = upright, 90 = horizontal.
   */
  torso_angle_deg: number | null;
  /**
   * Normalized head height (0 = top of frame).
   */
  head_y: number | null;
  /**
   * Normalized mid-hip height (0 = top of frame).
   */
  hip_y: number | null;
  /**
   * Person bounding box width / height.
   */
  bbox_aspect: number | null;
  /**
   * Change in hip_y per second; positive = moving down.
   */
  vertical_velocity: number | null;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "PosturePrediction".
 */
export interface PosturePrediction {
  model_id: string;
  model_revision: string;
  input_processing: "original" | "clahe";
  label: "standing" | "sitting" | "fallen" | "unknown";
  confidence: number | null;
  bbox: BoundingBox | null;
  detection_count: number;
  uncertainty_reason: ("no_detection" | "multiple_people" | "unsupported_label") | null;
}
/**
 * Body of POST /ingest/observations.
 *
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "ObservationBatch".
 */
export interface ObservationBatch {
  /**
   * @minItems 1
   * @maxItems 500
   */
  observations: [Observation, ...Observation[]];
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "RoomSnapshot".
 */
export interface RoomSnapshot {
  room_id: string;
  observations: Observation[];
  events: Event[];
  alerts: Alert[];
  sources: SourceStatus[];
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "AlertMessage".
 */
export interface AlertMessage {
  type: "alert";
  data: Alert;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "EventMessage".
 */
export interface EventMessage {
  type: "event";
  data: Event;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "NurseSnapshotMessage".
 */
export interface NurseSnapshotMessage {
  type: "nurse_snapshot";
  data: NurseSnapshot;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "ObservationMessage".
 */
export interface ObservationMessage {
  type: "observation";
  data: Observation;
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "SourceStatusMessage".
 */
export interface SourceStatusMessage {
  type: "source_status";
  data: SourceStatus;
}

/* Generated from shared/schema/hearth.schema.json by scripts/gen-types.mjs. Do not edit. */

/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "ServerMessage".
 */
export type ServerMessage = ObservationMessage | EventMessage | SourceStatusMessage;

export interface HearthContract {
  Event?: Event;
  GroundTruth?: GroundTruth;
  Heartbeat?: Heartbeat;
  Keypoint?: Keypoint;
  Observation?: Observation;
  ObservationBatch?: ObservationBatch;
  PostureFeatures?: PostureFeatures;
  SourceStatus?: SourceStatus;
  EventMessage?: EventMessage;
  ObservationMessage?: ObservationMessage;
  SourceStatusMessage?: SourceStatusMessage;
  ServerMessage?: ServerMessage;
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
  source_type: "edge" | "simulation" | "replay";
  room_id: string;
  scenario_id: string | null;
  event_type:
    | "possible_fall"
    | "recovered"
    | "person_visible"
    | "person_not_visible"
    | "face_present"
    | "face_absent"
    | "source_offline"
    | "source_online"
    | "alert_acknowledged";
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
 * Sent by every source about once per second; also carries edge performance metrics.
 *
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "Heartbeat".
 */
export interface Heartbeat {
  schema_version: "1";
  timestamp: string;
  source: string;
  source_type: "edge" | "simulation" | "replay";
  room_id: string;
  fps: number | null;
  inference_ms: number | null;
  mem_used_mb: number | null;
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
  confidence: number;
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
  source_type: "edge" | "simulation" | "replay";
  room_id: string;
  scenario_id: string | null;
  person_detected: boolean;
  keypoints: Keypoint[] | null;
  features: PostureFeatures | null;
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
 * Body of POST /ingest/observations.
 *
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "ObservationBatch".
 */
export interface ObservationBatch {
  observations: Observation[];
}
/**
 * This interface was referenced by `HearthContract`'s JSON-Schema
 * via the `definition` "SourceStatus".
 */
export interface SourceStatus {
  source: string;
  source_type: "edge" | "simulation" | "replay";
  room_id: string;
  online: boolean;
  last_seen: string | null;
  fps: number | null;
  inference_ms: number | null;
  mem_used_mb: number | null;
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

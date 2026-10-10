/* Generated from backend/schema/sim.schema.json by scripts/gen-types.mjs. Do not edit. */

export interface HearthSimApi {
  AlertResult?: AlertResult;
  Counts?: Counts;
  EvaluationReport?: EvaluationReport;
  EvaluationSummary?: EvaluationSummary;
  RunInfo?: RunInfo;
  ScenarioInfo?: ScenarioInfo;
  Scene?: Scene;
  SceneObject?: SceneObject;
  SegmentResult?: SegmentResult;
  SimStatus?: SimStatus;
  TimelineEvent?: TimelineEvent;
  TruthNow?: TruthNow;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "AlertResult".
 */
export interface AlertResult {
  event_id: string;
  t: number;
  outcome: "true_positive" | "duplicate" | "false_positive";
  /**
   * Resident the record concerns; None in single-person runs.
   */
  person_id: string | null;
  segment_id: string | null;
  truth_label:
    | (
        | "walking"
        | "standing"
        | "sitting"
        | "bending"
        | "lying_intentional"
        | "fall"
        | "occluded"
        | "camera_outage"
        | "absent"
      )
    | null;
  delay: number | null;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "Counts".
 */
export interface Counts {
  expected_alerts: number;
  true_positives: number;
  false_positives: number;
  missed: number;
  duplicates: number;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "EvaluationReport".
 */
export interface EvaluationReport {
  report_version: "1";
  run_id: string;
  scenario_id: string;
  seed: number;
  source: string;
  start: string;
  duration: number;
  /**
   * Resident ids; empty for single-person runs.
   */
  people: string[];
  alert_tolerance: number;
  counts: Counts;
  detection_delays: number[];
  segments: SegmentResult[];
  alerts: AlertResult[];
  timeline: TimelineEvent[];
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "SegmentResult".
 */
export interface SegmentResult {
  segment_id: string;
  /**
   * Resident the record concerns; None in single-person runs.
   */
  person_id: string | null;
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
  t_start: number;
  t_end: number;
  /**
   * What the monitor should have reported, if anything.
   */
  expected_event:
    | (
        | "possible_fall"
        | "recovered"
        | "person_visible"
        | "person_not_visible"
        | "face_present"
        | "face_absent"
        | "source_offline"
        | "source_online"
        | "alert_acknowledged"
      )
    | null;
  /**
   * None when nothing was expected.
   */
  detected: boolean | null;
  /**
   * Seconds from segment start to the first matching event.
   */
  delay: number | null;
  matched_event_id: string | null;
  /**
   * Unexpected possible_fall alerts raised during this segment.
   */
  false_alerts: number;
}
/**
 * Every monitor event of the run, placed on the scenario clock for the after-action view.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "TimelineEvent".
 */
export interface TimelineEvent {
  event_id: string;
  t: number;
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
  /**
   * Resident the record concerns; None in single-person runs.
   */
  person_id: string | null;
}
/**
 * Totals across many runs. These are simulated results, not real-world accuracy.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "EvaluationSummary".
 */
export interface EvaluationSummary {
  report_version: "1";
  runs: number;
  scenarios: number;
  counts: Counts;
  delay_median: number | null;
  delay_max: number | null;
  false_positives_by_label: {
    [k: string]: number;
  };
  missed_runs: string[];
  false_positive_runs: string[];
}
/**
 * Run metadata (<run>.run.json). The dashboard uses it to draw the room and place
 * events on the scenario clock (t = timestamp - start).
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "RunInfo".
 */
export interface RunInfo {
  run_id: string;
  scenario_id: string;
  title: string;
  seed: number;
  start: string;
  duration: number;
  fps: number;
  room_id: string;
  scene: Scene;
  residents: string[];
  observations: number;
  heartbeats: number;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "Scene".
 */
export interface Scene {
  /**
   * Frame y of the floor line under a person at the front.
   */
  floor_y: number;
  /**
   * Standing height at the front, as a fraction of frame height.
   */
  person_height: number;
  /**
   * Simulated camera width / height.
   */
  frame_aspect: number;
  /**
   * How far the floor line rises in the frame for a person at depth 1 (back wall).
   */
  depth_rise: number;
  /**
   * How much smaller a person at depth 1 appears than one at depth 0.
   */
  depth_shrink: number;
  objects: SceneObject[];
}
/**
 * A piece of furniture drawn in the room view. Rectangle in frame coordinates.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "SceneObject".
 */
export interface SceneObject {
  kind: "bed" | "chair" | "couch" | "mat" | "occluder";
  x: number;
  y: number;
  w: number;
  h: number;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "ScenarioInfo".
 */
export interface ScenarioInfo {
  scenario_id: string;
  title: string;
  description: string;
  /**
   * Nominal seconds; each seed's run may differ by about 10%.
   */
  duration: number;
  /**
   * Default seed.
   */
  seed: number;
  /**
   * Resident ids; empty for a single-person scenario.
   */
  residents: string[];
  scene: Scene;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "SimStatus".
 */
export interface SimStatus {
  state: "idle" | "loaded" | "playing" | "paused" | "ended";
  scenario_id: string | null;
  seed: number | null;
  /**
   * Key for GET /eval/runs/{run_id} once the run ends.
   */
  run_id: string | null;
  /**
   * Scenario seconds.
   */
  t: number;
  duration: number | null;
  speed: number;
  scene: Scene | null;
  truth: TruthNow[];
}
/**
 * Ground truth at this instant, for people watching. Never sent to the monitor.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "TruthNow".
 */
export interface TruthNow {
  person_id: string | null;
  segment_id: string;
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

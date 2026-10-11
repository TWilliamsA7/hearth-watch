/* Generated from backend/schema/sim.schema.json by scripts/gen-types.mjs. Do not edit. */

export interface HearthSimApi {
  ActorDetail?: ActorDetail;
  AlertResult?: AlertResult;
  CameraSpec?: CameraSpec;
  Counts?: Counts;
  EngineSpec?: EngineSpec;
  EvaluationReport?: EvaluationReport;
  EvaluationSummary?: EvaluationSummary;
  FallBreakdown?: FallBreakdown;
  FalseAlertBreakdown?: FalseAlertBreakdown;
  IncidentRecord?: IncidentRecord;
  IncidentSpec?: IncidentSpec;
  LiveActorInfo?: LiveActorInfo;
  LiveAppearance?: LiveAppearance;
  LiveConversation?: LiveConversation;
  LiveFrame?: LiveFrame;
  LivePerson?: LivePerson;
  LiveStatus?: LiveStatus;
  LiveWorld?: LiveWorld;
  Rate?: Rate;
  RoomSpec?: RoomSpec1;
  RunInfo?: RunInfo;
  ScenarioInfo?: ScenarioInfo;
  Scene?: Scene1;
  SceneObject?: SceneObject;
  SegmentResult?: SegmentResult;
  SimStatus?: SimStatus;
  StudyReport?: StudyReport;
  StudyRun?: StudyRun;
  TimelineEvent?: TimelineEvent;
  TruthNow?: TruthNow;
  WorldDoor?: WorldDoor;
  WorldFrame?: WorldFrame;
  WorldObject?: WorldObject;
  WorldPerson?: WorldPerson;
  WorldRoom?: WorldRoom;
  WorldRoute?: WorldRoute;
  WorldTrack?: WorldTrack;
  WorldWindow?: WorldWindow;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "ActorDetail".
 */
export interface ActorDetail {
  info: LiveActorInfo;
  now: LivePerson;
  /**
   * Scenario seconds the current intent began.
   */
  since: number;
  /**
   * Activities still to come.
   */
  plan: string[];
  /**
   * Recent intents and when they began.
   */
  history: [string, number][];
  cooldowns: {
    [k: string]: number;
  };
  incidents: IncidentRecord[];
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "LiveActorInfo".
 */
export interface LiveActorInfo {
  person_id: string;
  archetype: string;
  height: number;
  /**
   * Cruising pace, m/s.
   */
  speed: number;
  sociability: number;
  mobility: string;
  appearance: LiveAppearance;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "LiveAppearance".
 */
export interface LiveAppearance {
  skin: string;
  top: string;
  bottom: string;
  hair: string;
  /**
   * Width relative to average.
   */
  build: number;
}
/**
 * A WorldPerson plus what they're up to. Ground truth: for people watching only.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "LivePerson".
 */
export interface LivePerson {
  /**
   * Matches Observation.person_id; None in a single-person run.
   */
  person_id: string | null;
  /**
   * Radians about +y; 0 faces the camera, +pi/2 faces image right.
   */
  heading: number;
  /**
   * Standing height in metres. Constant per person, unlike limb lengths in motion.
   */
  height: number;
  /**
   * Metres; every joint, visible to the camera or not.
   */
  joints: {
    /**
     * @minItems 3
     * @maxItems 3
     */
    [k: string]: [number, number, number];
  };
  /**
   * Ground-truth label right now (TruthLabel).
   */
  label: string;
  /**
   * What they're doing this moment, e.g. 'sitting down', 'talking'.
   */
  activity: string;
  /**
   * What the current plan is for: sit, rest, chat, fall...
   */
  intent: string;
  posture: string;
  destination: [number, number] | null;
  /**
   * Rest of the route, floor (x, z).
   */
  path: [number, number][];
  conversation: string | null;
  incident: string | null;
  /**
   * Fraction of joints the camera could see last frame (0..1), before noise.
   */
  visible: number;
  /**
   * 'live': a real person from a live camera feed, not simulated.
   */
  kind: "simulated" | "live";
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "IncidentRecord".
 */
export interface IncidentRecord {
  incident_id: string;
  kind: "fall" | "inactivity" | "pacing" | "exit_loiter";
  source: "manual" | "scheduled" | "procedural";
  person_id: string;
  t_start: number;
  /**
   * @minItems 2
   * @maxItems 2
   */
  location: [number, number];
  posture_before: string;
  posture_after: string | null;
  motion: string | null;
  t_floor: number | null;
  t_recovered: number | null;
  t_end: number | null;
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
 * via the `definition` "CameraSpec".
 */
export interface CameraSpec {
  preset: "corner_left" | "corner_right" | "front" | "ceiling" | "jetson" | "jetson_corner";
  height: number;
  /**
   * Vertical field of view, degrees.
   */
  fov: number;
  aspect: number;
  fps: number;
  /**
   * Preset 'jetson': degrees the camera looks down (edge/DEMO.md: 10-15).
   */
  tilt: number;
  /**
   * Preset 'jetson': metres into the room where the view centre is aimed.
   */
  reach: number;
  /**
   * Preset 'jetson_corner': which front corner of the tracked area the camera is above.
   */
  corner: "left" | "right";
  /**
   * Preset 'jetson_corner': degrees the view is turned from the area centre (+ = right).
   */
  pan: number;
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
 * via the `definition` "EngineSpec".
 */
export interface EngineSpec {
  room: RoomSpec;
  actors: number;
  camera: CameraSpec1;
  lighting: "day" | "evening" | "dim";
  /**
   * Scales how often people change activity (2 = dwell half as long).
   */
  activity_frequency: number;
  /**
   * Scales how often people start conversations (0 = never).
   */
  social_frequency: number;
  /**
   * Scheduled incidents.
   */
  incidents: IncidentSpec[];
  /**
   * Procedural incidents per actor-hour, by kind. Empty = none (the default).
   */
  incident_rates: {
    [k: string]: number;
  };
  /**
   * Perception noise on the camera's keypoints.
   */
  noise: boolean;
  room_id: string;
}
export interface RoomSpec {
  /**
   * Metres, wall to wall across the camera's view. None = drawn from the seed.
   */
  width: number | null;
  /**
   * Metres, front to back.
   */
  depth: number | null;
  height: number;
  /**
   * 0 = a bare room with a chair or two, 1 = as much as fits.
   */
  furniture_density: number;
  doors: number;
  window: boolean;
}
export interface CameraSpec1 {
  preset: "corner_left" | "corner_right" | "front" | "ceiling" | "jetson" | "jetson_corner";
  height: number;
  /**
   * Vertical field of view, degrees.
   */
  fov: number;
  aspect: number;
  fps: number;
  /**
   * Preset 'jetson': degrees the camera looks down (edge/DEMO.md: 10-15).
   */
  tilt: number;
  /**
   * Preset 'jetson': metres into the room where the view centre is aimed.
   */
  reach: number;
  /**
   * Preset 'jetson_corner': which front corner of the tracked area the camera is above.
   */
  corner: "left" | "right";
  /**
   * Preset 'jetson_corner': degrees the view is turned from the area centre (+ = right).
   */
  pan: number;
}
/**
 * One incident, at a time (scheduled) or now (manual).
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "IncidentSpec".
 */
export interface IncidentSpec {
  kind: "fall" | "inactivity" | "pacing" | "exit_loiter";
  /**
   * Scenario seconds (scheduled incidents).
   */
  t: number;
  /**
   * person_id; None = someone suitable, chosen by the seed.
   */
  actor: string | null;
  /**
   * Falls: how. None = what suits the posture.
   */
  motion: ("trip" | "faint" | "collapse" | "sit_miss" | "stand_up") | null;
  /**
   * Inactivity: how long. Falls: time on the floor before getting up.
   */
  duration: number | null;
  /**
   * Falls: get up after `duration` (else stay down until reset).
   */
  recover: boolean;
  /**
   * Pacing legs or exit visits.
   */
  repeats: number;
  /**
   * Falls: stage it where the camera has a clear, side-on view (as hand-written scenarios do). False = wherever there's room, seen or not: the unbiased setting for validation.
   */
  in_view: boolean;
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
        | "alert_resolved"
        | "help_requested"
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
    | "alert_acknowledged"
    | "alert_resolved"
    | "help_requested";
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
 * How the monitor did on falls that share one factor value (e.g. fault = trip).
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "FallBreakdown".
 */
export interface FallBreakdown {
  factor: string;
  value: string;
  detected: Rate;
  delay_median: number | null;
  delay_p90: number | null;
}
/**
 * hits out of n, with a 95% Wilson interval so small samples read as uncertain.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "Rate".
 */
export interface Rate {
  n: number;
  hits: number;
  rate: number | null;
  low: number | null;
  high: number | null;
}
/**
 * False possible_fall alerts per simulated hour, for runs or activities sharing one factor value.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "FalseAlertBreakdown".
 */
export interface FalseAlertBreakdown {
  factor: string;
  value: string;
  false_alerts: number;
  /**
   * Simulated room-hours (runs) or person-hours (activities).
   */
  hours: number;
  per_hour: number | null;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "LiveConversation".
 */
export interface LiveConversation {
  conv_id: string;
  members: string[];
  /**
   * @minItems 2
   * @maxItems 2
   */
  centre: [number, number];
}
/**
 * Everyone right now. Pushed over WS /sim/live/ws as {"type": "frame", "data": ...}.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "LiveFrame".
 */
export interface LiveFrame {
  t: number;
  state: "idle" | "playing" | "paused";
  speed: number;
  people: LivePerson[];
  conversations: LiveConversation[];
  /**
   * Incidents in progress.
   */
  incidents: IncidentRecord[];
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "LiveStatus".
 */
export interface LiveStatus {
  state: "idle" | "playing" | "paused";
  run_id: string | null;
  scenario_id: string | null;
  seed: number | null;
  t: number;
  speed: number;
  actors: number;
}
/**
 * The run's fixed parts: sent once per connection and on every new room ({"type": "world"}).
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "LiveWorld".
 */
export interface LiveWorld {
  run_id: string;
  scenario_id: string;
  seed: number;
  spec: EngineSpec;
  room: WorldRoom;
  scene: Scene;
  actors: LiveActorInfo[];
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "WorldRoom".
 */
export interface WorldRoom {
  /**
   * Metres the camera sees across at depth 0.
   */
  width_front: number;
  /**
   * Metres the camera sees across at the back wall.
   */
  width_back: number;
  depth: number;
  height: number;
  /**
   * Approximate; for display only.
   *
   * @minItems 3
   * @maxItems 3
   */
  camera_position: [number, number, number];
  /**
   * @minItems 3
   * @maxItems 3
   */
  camera_target: [number, number, number];
  /**
   * Vertical field of view, degrees.
   */
  camera_fov: number;
  objects: WorldObject[];
  /**
   * Live rooms: the walls as (x0, z0, x1, z1). None = the scripted room (width_back x depth).
   */
  bounds: [number, number, number, number] | null;
  doors: WorldDoor[];
  windows: WorldWindow[];
  /**
   * Width / height of the camera image.
   */
  camera_aspect: number | null;
  /**
   * Live rooms: day, evening or dim.
   */
  lighting: string | null;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "WorldObject".
 */
export interface WorldObject {
  kind: "bed" | "chair" | "couch" | "mat" | "occluder";
  /**
   * @minItems 3
   * @maxItems 3
   */
  center: [number, number, number];
  /**
   * Width (x), height (y), depth (z) in metres.
   *
   * @minItems 3
   * @maxItems 3
   */
  size: [number, number, number];
  /**
   * Radians about +y: the heading its front faces (as WorldPerson.heading).
   */
  rotation: number;
  /**
   * The full kind for live rooms (armchair, table, bookshelf...); kind is the nearest ObjectKind.
   */
  variant: string | null;
  /**
   * Metres: top of the seat. None = not for sitting, or unknown.
   */
  seat_height: number | null;
  /**
   * Metres: top of what someone lies on.
   */
  surface_height: number | null;
  /**
   * Metres: width of each arm (armchairs, couches).
   */
  arm_width: number | null;
  /**
   * Metres: top of the arms.
   */
  arm_height: number | null;
  /**
   * Metres the backrest (or a bed's headboard) comes forward from the back of the piece.
   */
  back_depth: number | null;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "WorldDoor".
 */
export interface WorldDoor {
  door_id: string;
  wall: "back" | "left" | "right" | "front";
  /**
   * Floor point (x, z) at the middle of the opening.
   *
   * @minItems 2
   * @maxItems 2
   */
  center: [number, number];
  width: number;
  /**
   * Radians: the way into the room.
   */
  heading_in: number;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "WorldWindow".
 */
export interface WorldWindow {
  wall: "back" | "left" | "right" | "front";
  /**
   * @minItems 2
   * @maxItems 2
   */
  center: [number, number];
  width: number;
  sill: number;
  height: number;
}
/**
 * The furniture as the camera sees it, for the 2D camera view.
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
 * via the `definition` "RoomSpec".
 */
export interface RoomSpec1 {
  /**
   * Metres, wall to wall across the camera's view. None = drawn from the seed.
   */
  width: number | null;
  /**
   * Metres, front to back.
   */
  depth: number | null;
  height: number;
  /**
   * 0 = a bare room with a chair or two, 1 = as much as fits.
   */
  furniture_density: number;
  doors: number;
  window: boolean;
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
  scene: Scene1;
  residents: string[];
  observations: number;
  heartbeats: number;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "Scene".
 */
export interface Scene1 {
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
  scene: Scene1;
  /**
   * Scenario.tags, e.g. generator settings.
   */
  tags: {
    [k: string]: string | number | boolean;
  };
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
  scene: Scene1 | null;
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
/**
 * Results of many generated scenarios, broken down by what was in them. Simulated only.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "StudyReport".
 */
export interface StudyReport {
  report_version: "1";
  monitor: string;
  runs: number;
  simulated_hours: number;
  counts: Counts;
  detected: Rate;
  delay_median: number | null;
  delay_p90: number | null;
  false_alerts_per_hour: number | null;
  falls_by: FallBreakdown[];
  false_alerts_by: FalseAlertBreakdown[];
  /**
   * Runs with missed falls or the most false alerts, worst first.
   */
  worst: StudyRun[];
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "StudyRun".
 */
export interface StudyRun {
  run_id: string;
  scenario_id: string;
  seed: number;
  title: string;
  duration: number;
  tags: {
    [k: string]: string | number | boolean;
  };
  expected_alerts: number;
  missed: number;
  false_alerts: number;
  delays: number[];
  /**
   * Command that writes this run's files for the room-view dev page.
   */
  reproduce: string;
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "WorldFrame".
 */
export interface WorldFrame {
  /**
   * Scenario seconds.
   */
  t: number;
  people: WorldPerson[];
}
/**
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "WorldPerson".
 */
export interface WorldPerson {
  /**
   * Matches Observation.person_id; None in a single-person run.
   */
  person_id: string | null;
  /**
   * Radians about +y; 0 faces the camera, +pi/2 faces image right.
   */
  heading: number;
  /**
   * Standing height in metres. Constant per person, unlike limb lengths in motion.
   */
  height: number;
  /**
   * Metres; every joint, visible to the camera or not.
   */
  joints: {
    /**
     * @minItems 3
     * @maxItems 3
     */
    [k: string]: [number, number, number];
  };
}
/**
 * One continuous walk: the straight legs a resident takes from t_start to t_end.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "WorldRoute".
 */
export interface WorldRoute {
  person_id: string | null;
  t_start: number;
  t_end: number;
  /**
   * Floor points (x, z) in metres, start first.
   */
  points: [number, number][];
}
/**
 * A run's bodies in 3D (<run>.world.json, GET /sim/world). Simulator-only.
 *
 * This interface was referenced by `HearthSimApi`'s JSON-Schema
 * via the `definition` "WorldTrack".
 */
export interface WorldTrack {
  scenario_id: string;
  seed: number;
  fps: number;
  duration: number;
  room: WorldRoom;
  /**
   * One per frame slot, in time order.
   */
  frames: WorldFrame[];
  /**
   * Every walk, for drawing routes on the floor.
   */
  routes: WorldRoute[];
}

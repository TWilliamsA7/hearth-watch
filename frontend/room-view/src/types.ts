// Contract and API types used by this package. All generated: see scripts/gen-types.mjs.
export type {
  Event,
  GroundTruth,
  Heartbeat,
  Keypoint,
  Observation,
  ServerMessage,
  SourceStatus,
} from "./generated/contract";
export type {
  Appointment,
  AppointmentInput,
  CheckIn,
  EvaluationReport,
  RunInfo,
  Scene,
  SceneObject,
  WorldFrame,
  WorldObject,
  WorldPerson,
  WorldRoom,
  WorldRoute,
  WorldTrack,
} from "./generated/sim";

import type { Observation } from "./generated/contract";

export type SourceType = Observation["source_type"];
export type KeypointName = NonNullable<Observation["keypoints"]>[number]["name"];

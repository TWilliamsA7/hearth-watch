// The live camera link (/sim/live/link): real people a camera sees, played out in the simulated room.
// Shapes follow backend/hearth/sim/live/link.py.

export interface ModelCall {
  model_id: string;
  label: string;
  confidence: number | null;
}

/** One real person the linked camera tracks (person_id is their id in the room). */
export interface LivePersonStatus {
  person_id: string;
  track_id: string | null;
  state: "live" | "stale" | "outside";
  posture: string | null;
  label: string | null;
  model: ModelCall | null;
  alert_open: boolean;
}

export interface LinkStatus {
  enabled: boolean;
  source: string;
  state: "off" | "waiting" | "live" | "stale" | "outside";
  person_id: string;
  last_seen: string | null;
  posture: string | null;
  label: string | null;
  alert_open: boolean;
  camera: { preset: string; height: number; tilt: number; fov: number; corner: string; pan: number } | null;
  people: LivePersonStatus[];
}

export interface CalibrateResult {
  ok: boolean;
  step: "walk" | "centre";
  pan: number | null;
  off_by: number | null;
  frames: number;
  height: number | null;
  tilt: number | null;
  rms: number | null;
  reason: string;
}

// Sources that send keypoints (the fall model sends none, so it can't be placed in the room).
export const LIVE_SOURCES = ["jetson-01", "replay-01"];

/** True for a person the link put in the room, as opposed to a simulated resident. */
export const isLivePerson = (personId: string | null | undefined) => !!personId?.startsWith("live-");

/** "LIVE P2" for live-jetson-01-p2, "LIVE" for a one-person camera. */
export const liveName = (personId: string | null | undefined, source: string | undefined) => {
  const track = personId && source && personId.startsWith(`live-${source}-`) ? personId.slice(`live-${source}-`.length) : "";
  return track ? `LIVE ${track.toUpperCase()}` : "LIVE";
};

/** "sitting 93%": what the camera's own posture model says. */
export const modelText = (m: ModelCall | null | undefined) =>
  m ? `${m.label}${m.confidence != null ? ` ${Math.round(m.confidence * 100)}%` : ""}` : "—";

/** What the calibration did, in words. */
export function calibrationText(r: CalibrateResult): string {
  if (!r.ok) return `Calibration failed: ${r.reason}`;
  return r.step === "centre"
    ? `Direction set: turned ${r.pan?.toFixed(1)}° (you stood ${Math.round((r.off_by ?? 0) * 100)} cm from where the X should be)`
    : `Camera set: ${r.height?.toFixed(2)} m up, tilted ${r.tilt?.toFixed(1)}° (fit ±${Math.round((r.rms ?? 0) * 480)} px, ${r.frames} frames). Now step 2.`;
}

/** The link's state in a sentence. */
export function linkStateText(link: LinkStatus): string {
  return link.state === "live" ? `In the room: ${link.people.filter((q) => q.state === "live").map((q) =>
      `${link.people.length > 1 ? `${liveName(q.person_id, link.source)} ` : ""}${q.label ?? q.posture}${q.alert_open ? " (ALERT OPEN)" : ""}`).join(" · ")}`
    : link.state === "waiting" ? `Waiting for ${link.source}: start run_edge.py or edge/tools/replay.py`
    : link.state === "outside" ? "Outside the tracked area: step back inside the tape"
    : link.state === "off" ? "Not linked"
    : `No fresh frames from ${link.source}`;
}

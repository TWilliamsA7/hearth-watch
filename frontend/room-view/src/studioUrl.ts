import type { EngineSpec } from "./generated/sim";
import type { CameraMode, Overlays } from "./LiveView3D";
import type { Lighting } from "./room3d";

/**
 * The Sim Studio's settings in its URL, so a link opens the same room with the same view:
 * the room and population (seed, size, how many people...), the camera, and the overlays.
 * Only what differs from the defaults is written, with short keys.
 */
export interface StudioSettings {
  seed: number;
  actors: number;
  width: number;
  depth: number;
  density: number;
  preset: EngineSpec["camera"]["preset"];
  camHeight: number;
  camTilt: number;
  camFov: number;
  lighting: Lighting;
  activity: number;
  social: number;
  overlays: Overlays;
  cameraMode: CameraMode;
}

type NumberKey = "seed" | "actors" | "width" | "depth" | "density" | "camHeight" | "camTilt" | "camFov" | "activity" | "social";

const NUMBERS: [NumberKey, string][] = [
  ["seed", "seed"], ["actors", "people"], ["width", "w"], ["depth", "d"], ["density", "density"],
  ["camHeight", "camh"], ["camTilt", "tilt"], ["camFov", "fov"], ["activity", "act"], ["social", "social"],
];
const PRESETS: EngineSpec["camera"]["preset"][] = ["corner_left", "corner_right", "front", "ceiling", "jetson", "jetson_corner"];
const LIGHTING: Lighting[] = ["day", "evening", "dim"];
const MODES: CameraMode[] = ["orbit", "top", "camera", "follow"];

/** Settings from a query string, falling back to `defaults` for anything missing or invalid. */
export function readSettings(search: string, defaults: StudioSettings): StudioSettings {
  const q = new URLSearchParams(search);
  const out: StudioSettings = { ...defaults, overlays: { ...defaults.overlays } };
  for (const [key, name] of NUMBERS) {
    const raw = q.get(name);
    const n = raw === null || raw === "" ? NaN : Number(raw);
    if (Number.isFinite(n)) out[key] = n;
  }
  const preset = q.get("cam");
  if (preset && (PRESETS as string[]).includes(preset)) out.preset = preset as StudioSettings["preset"];
  const light = q.get("light");
  if (light && (LIGHTING as string[]).includes(light)) out.lighting = light as Lighting;
  const view = q.get("view");
  if (view && (MODES as string[]).includes(view)) out.cameraMode = view as CameraMode;
  const ov = q.get("ov");
  if (ov !== null) {
    const on = new Set(ov.split(",").filter(Boolean));
    for (const k of Object.keys(out.overlays) as (keyof Overlays)[]) out.overlays[k] = on.has(k);
  }
  return out;
}

/** A query string ("" or "?...") holding what differs from `defaults`. */
export function writeSettings(s: StudioSettings, defaults: StudioSettings): string {
  const q = new URLSearchParams();
  for (const [key, name] of NUMBERS) if (s[key] !== defaults[key]) q.set(name, String(s[key]));
  if (s.preset !== defaults.preset) q.set("cam", s.preset);
  if (s.lighting !== defaults.lighting) q.set("light", s.lighting);
  if (s.cameraMode !== defaults.cameraMode) q.set("view", s.cameraMode);
  const keys = Object.keys(s.overlays) as (keyof Overlays)[];
  if (keys.some((k) => s.overlays[k] !== defaults.overlays[k])) q.set("ov", keys.filter((k) => s.overlays[k]).join(","));
  const str = q.toString();
  return str ? `?${str}` : "";
}

/** Does the room running now differ from the one these settings would generate? */
export function roomDiffers(s: StudioSettings, running: { seed: number; spec: EngineSpec }): boolean {
  const { spec } = running;
  return running.seed !== s.seed || spec.actors !== s.actors || spec.room.width !== s.width || spec.room.depth !== s.depth
    || spec.room.furniture_density !== s.density;
}

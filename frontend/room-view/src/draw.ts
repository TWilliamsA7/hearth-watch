import { BONES, HEAD_POINTS } from "./skeleton";
import type { Keypoint, Scene, SceneObject, SourceType } from "./types";

/**
 * ok       recent data is arriving
 * waiting  nothing received yet
 * stale    data stopped arriving (the view's own check)
 * offline  the backend says the source is offline
 * Only "ok" may look calm: silence is never shown as safety.
 */
export type FeedStatus = "ok" | "waiting" | "stale" | "offline";

export interface FrameState {
  keypoints: Keypoint[] | null;
  /** null = no observation received yet. */
  personDetected: boolean | null;
  /** null = face detection did not run on this frame. */
  facePresent: boolean | null;
  sourceType: SourceType | null;
  status: FeedStatus;
  alertActive: boolean;
}

export interface Theme {
  background: string;
  floor: string;
  furniture: Record<SceneObject["kind"], string>;
  bone: string;
  joint: string;
  alert: string;
  face: string;
  text: string;
  muted: string;
  badge: Record<SourceType, string>;
}

export const DEFAULT_THEME: Theme = {
  background: "#1b1f27",
  floor: "#262b35",
  furniture: { bed: "#3a4a63", chair: "#4a3f35", mat: "#2f4a3c", occluder: "#3b3640" },
  bone: "#e8edf5",
  joint: "#7cc4ff",
  alert: "#ff5d5d",
  face: "#ffd166",
  text: "#f4f6fa",
  muted: "#9aa3b2",
  badge: { simulation: "#7b61ff", replay: "#d99a1e", edge: "#e5484d" },
};

const BADGE_TEXT: Record<SourceType, string> = { simulation: "SIMULATED", replay: "REPLAY", edge: "LIVE" };

/** Scene used when none is supplied (e.g. a live room with no configured geometry). */
export const EMPTY_SCENE: Scene = { floor_y: 0.92, person_height: 0.62, frame_aspect: 4 / 3, objects: [] };

const MIN_CONFIDENCE = 0.3;

/**
 * Draw one frame of the camera view. Pure: everything it shows comes from its arguments.
 * Width and height are in canvas pixels; callers keep width / height = scene.frame_aspect.
 */
export function drawRoom(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  frame: FrameState,
  scene: Scene = EMPTY_SCENE,
  theme: Theme = DEFAULT_THEME,
): void {
  const px = (x: number) => x * width;
  const py = (y: number) => y * height;
  const unit = Math.min(width, height);

  ctx.save();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = theme.floor;
  ctx.fillRect(0, py(scene.floor_y), width, height - py(scene.floor_y));

  const furniture = scene.objects.filter((o) => o.kind !== "occluder");
  const occluders = scene.objects.filter((o) => o.kind === "occluder");
  for (const o of furniture) drawObject(ctx, o, px, py, theme);

  const live = frame.status === "ok";
  if (frame.keypoints && frame.personDetected) {
    ctx.globalAlpha = live ? 1 : 0.3; // keep the last known pose visible, clearly faded
    drawSkeleton(ctx, frame.keypoints, px, py, unit, frame.alertActive ? theme.alert : theme.bone, theme);
    if (frame.facePresent && live) drawFaceBox(ctx, frame.keypoints, px, py, unit, theme);
    ctx.globalAlpha = 1;
  }
  for (const o of occluders) drawObject(ctx, o, px, py, theme); // in front of the person

  if (frame.alertActive) {
    ctx.strokeStyle = theme.alert;
    ctx.lineWidth = unit * 0.012;
    ctx.strokeRect(ctx.lineWidth / 2, ctx.lineWidth / 2, width - ctx.lineWidth, height - ctx.lineWidth);
  }

  drawBadge(ctx, frame.sourceType, unit, theme);
  const message = statusMessage(frame);
  if (message) drawMessage(ctx, message, width, unit, theme, frame.status !== "ok");
  ctx.restore();
}

export function statusMessage(frame: FrameState): string | null {
  switch (frame.status) {
    case "waiting":
      return "Waiting for data — room status unknown";
    case "offline":
      return "Source offline — room status unknown";
    case "stale":
      return "No recent data — room status unknown";
    default:
      return frame.personDetected === false ? "No person detected" : null;
  }
}

function drawObject(
  ctx: CanvasRenderingContext2D,
  o: SceneObject,
  px: (x: number) => number,
  py: (y: number) => number,
  theme: Theme,
): void {
  ctx.fillStyle = theme.furniture[o.kind];
  ctx.fillRect(px(o.x), py(o.y), px(o.w), py(o.h));
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  px: (x: number) => number,
  py: (y: number) => number,
  unit: number,
  color: string,
  theme: Theme,
): void {
  const pts = new Map(keypoints.filter((k) => k.confidence >= MIN_CONFIDENCE).map((k) => [k.name, k]));
  ctx.lineCap = "round";
  ctx.lineWidth = unit * 0.012;
  ctx.strokeStyle = color;
  for (const [a, b] of BONES) {
    const p = pts.get(a);
    const q = pts.get(b);
    if (!p || !q) continue;
    ctx.beginPath();
    ctx.moveTo(px(p.x), py(p.y));
    ctx.lineTo(px(q.x), py(q.y));
    ctx.stroke();
  }
  ctx.fillStyle = theme.joint;
  for (const k of pts.values()) {
    ctx.beginPath();
    ctx.arc(px(k.x), py(k.y), unit * 0.008, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawFaceBox(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  px: (x: number) => number,
  py: (y: number) => number,
  unit: number,
  theme: Theme,
): void {
  const head = keypoints.filter((k) => HEAD_POINTS.includes(k.name) && k.confidence >= MIN_CONFIDENCE);
  if (!head.length) return;
  const xs = head.map((k) => px(k.x));
  const ys = head.map((k) => py(k.y));
  const pad = unit * 0.03;
  const x0 = Math.min(...xs) - pad;
  const y0 = Math.min(...ys) - pad;
  ctx.strokeStyle = theme.face;
  ctx.lineWidth = unit * 0.005;
  ctx.strokeRect(x0, y0, Math.max(...xs) + pad - x0, Math.max(...ys) + pad - y0);
  ctx.fillStyle = theme.face;
  ctx.font = `${Math.round(unit * 0.03)}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  ctx.fillText("face detected", x0, y0 - unit * 0.005);
}

function drawBadge(ctx: CanvasRenderingContext2D, sourceType: SourceType | null, unit: number, theme: Theme): void {
  const text = sourceType ? BADGE_TEXT[sourceType] : "NO SOURCE";
  const size = Math.round(unit * 0.045);
  ctx.font = `700 ${size}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + size;
  const m = unit * 0.025;
  ctx.fillStyle = sourceType ? theme.badge[sourceType] : theme.muted;
  ctx.fillRect(m, m, w, size * 1.6);
  ctx.fillStyle = theme.text;
  ctx.textBaseline = "middle";
  ctx.fillText(text, m + size / 2, m + size * 0.8);
}

function drawMessage(
  ctx: CanvasRenderingContext2D,
  text: string,
  width: number,
  unit: number,
  theme: Theme,
  warning: boolean,
): void {
  const size = Math.round(unit * 0.045);
  ctx.font = `600 ${size}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + size * 1.5;
  const h = size * 2;
  const x = (width - w) / 2;
  const y = unit * 0.025 + size * 2.4; // below the source badge
  ctx.fillStyle = warning ? "rgba(229, 72, 77, 0.9)" : "rgba(0, 0, 0, 0.55)";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = theme.text;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, width / 2, y + h / 2);
  ctx.textAlign = "start";
}

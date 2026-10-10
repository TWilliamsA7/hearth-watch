import type { Keypoint, KeypointName } from "./types";

/** Limbs of the COCO-18 (poseNet resnet18-body) skeleton. */
export const BONES: ReadonlyArray<readonly [KeypointName, KeypointName]> = [
  ["nose", "left_eye"],
  ["nose", "right_eye"],
  ["left_eye", "left_ear"],
  ["right_eye", "right_ear"],
  ["nose", "neck"],
  ["neck", "left_shoulder"],
  ["neck", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
];

export const HEAD_POINTS: ReadonlyArray<KeypointName> = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"];

/**
 * Blend two frames of keypoints for smooth display between observations.
 * Joints only in `to` appear at their new position; joints missing from `to` are dropped,
 * so the display never shows a joint the latest frame did not see.
 */
export function interpolateKeypoints(from: Keypoint[] | null, to: Keypoint[] | null, alpha: number): Keypoint[] | null {
  if (!to) return null;
  if (!from || alpha >= 1) return to;
  const a = Math.max(0, alpha);
  const prev = new Map(from.map((k) => [k.name, k]));
  return to.map((k) => {
    const p = prev.get(k.name);
    if (!p) return k;
    return { ...k, x: p.x + (k.x - p.x) * a, y: p.y + (k.y - p.y) * a };
  });
}

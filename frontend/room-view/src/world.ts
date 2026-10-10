import type { WorldFrame, WorldPerson, WorldTrack } from "./types";

export type Vec3 = [number, number, number];

const EMPTY_FRAME: WorldFrame = { t: 0, people: [] };

/**
 * The world at scenario time t, blending each person between the two frames around t.
 * Before the first frame or after the last, the nearest frame holds. A person in only
 * one of the two frames (arriving or leaving) is shown as that frame has them.
 */
export function sampleWorld(track: WorldTrack | null, t: number): WorldFrame {
  const frames = track?.frames ?? [];
  if (!frames.length) return EMPTY_FRAME;
  if (t <= frames[0].t) return frames[0];
  if (t >= frames[frames.length - 1].t) return frames[frames.length - 1];

  // Last frame at or before t.
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t <= t) lo = mid;
    else hi = mid - 1;
  }
  const a = frames[lo];
  const b = frames[lo + 1];
  const alpha = (t - a.t) / (b.t - a.t);
  const next = new Map(b.people.map((p) => [p.person_id, p]));
  const people = a.people.map((p) => {
    const q = next.get(p.person_id);
    return q ? blendPerson(p, q, alpha) : p;
  });
  const before = new Set(a.people.map((p) => p.person_id));
  for (const q of b.people) if (!before.has(q.person_id)) people.push(q);
  return { t, people };
}

function blendPerson(a: WorldPerson, b: WorldPerson, alpha: number): WorldPerson {
  const joints: WorldPerson["joints"] = {};
  for (const [name, p] of Object.entries(a.joints)) {
    const q = b.joints[name];
    joints[name] = q ? lerp3(p, q, alpha) : p;
  }
  return { person_id: a.person_id, heading: lerpAngle(a.heading, b.heading, alpha), height: b.height, joints };
}

function lerp3(p: Vec3, q: Vec3, alpha: number): Vec3 {
  return [p[0] + (q[0] - p[0]) * alpha, p[1] + (q[1] - p[1]) * alpha, p[2] + (q[2] - p[2]) * alpha];
}

/** Turn the short way round. */
function lerpAngle(a: number, b: number, alpha: number): number {
  const turn = ((((b - a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
  return a + turn * alpha;
}

import { describe, expect, it } from "vitest";

import type { WorldPerson, WorldTrack } from "./types";
import { routesAt, sampleWorld } from "./world";

function person(id: string | null, x: number, heading = 0): WorldPerson {
  return { person_id: id, heading, height: 1.7, joints: { nose: [x, 1.5, 0], neck: [x, 1.4, 0] } };
}

const room: WorldTrack["room"] = {
  width_front: 3.6, width_back: 5.2, depth: 4, height: 2.6,
  camera_position: [0, 1.2, 3], camera_target: [0, 1.2, -2], camera_fov: 50, objects: [],
  bounds: null, doors: [], windows: [], camera_aspect: null, lighting: null,
};

function track(frames: WorldTrack["frames"]): WorldTrack {
  return { scenario_id: "test", seed: 0, fps: 2, duration: 1, room, frames, routes: [] };
}

describe("sampleWorld", () => {
  const tr = track([
    { t: 0, people: [person("p1", 0), person("p2", 1)] },
    { t: 0.5, people: [person("p1", 1), person("p3", 2)] },
  ]);

  it("blends people between the frames around t", () => {
    const f = sampleWorld(tr, 0.25);
    const p1 = f.people.find((p) => p.person_id === "p1")!;
    expect(p1.joints.nose[0]).toBeCloseTo(0.5);
    expect(p1.joints.nose[1]).toBeCloseTo(1.5);
  });

  it("holds the nearest frame outside the track", () => {
    expect(sampleWorld(tr, -1).people.map((p) => p.joints.nose[0])).toEqual([0, 1]);
    expect(sampleWorld(tr, 9).people.map((p) => p.person_id)).toEqual(["p1", "p3"]);
  });

  it("shows arriving and leaving people as their one frame has them", () => {
    const ids = sampleWorld(tr, 0.25).people.map((p) => p.person_id).sort();
    expect(ids).toEqual(["p1", "p2", "p3"]);
  });

  it("turns the short way round", () => {
    const turning = track([
      { t: 0, people: [person(null, 0, 3)] },
      { t: 1, people: [person(null, 0, -3)] },
    ]);
    const h = sampleWorld(turning, 0.5).people[0].heading;
    expect(Math.abs(Math.abs(h) - Math.PI)).toBeLessThan(0.01);
  });

  it("is empty without a track", () => {
    expect(sampleWorld(null, 1).people).toEqual([]);
  });
});

describe("routesAt", () => {
  const walk = { person_id: "p1", t_start: 4, t_end: 6, points: [[0, 0], [1, -1]] as [number, number][] };
  const tr = { ...track([]), routes: [walk] };

  it("shows a route from just before the walk until just after it", () => {
    expect(routesAt(tr, 2)).toEqual([]);
    expect(routesAt(tr, 3)).toEqual([walk]);
    expect(routesAt(tr, 6.5)).toEqual([walk]);
    expect(routesAt(tr, 7.5)).toEqual([]);
  });

  it("copes with no track and with tracks recorded before routes existed", () => {
    expect(routesAt(null, 5)).toEqual([]);
    expect(routesAt({ ...track([]), routes: undefined } as unknown as WorldTrack, 5)).toEqual([]);
  });
});

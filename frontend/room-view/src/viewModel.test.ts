import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { interpolateKeypoints } from "./skeleton";
import type { Observation, ServerMessage, SourceStatus } from "./types";
import { INITIAL_VIEW, frameAt, reduce } from "./viewModel";

// The shared contract fixture: what the Jetson sends. The view must accept it unchanged.
const EDGE: Observation = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "../../../shared/fixtures/observation.example.json"), "utf8"),
);
const T0 = Date.parse(EDGE.timestamp);
const OPTS = { staleAfterMs: 3000, alertActive: false, smoothing: false };

const observation = (data: Partial<Observation> = {}): ServerMessage => ({ type: "observation", data: { ...EDGE, ...data } });
const status = (online: boolean, source = EDGE.source): ServerMessage => ({
  type: "source_status",
  data: { source, source_type: "edge", room_id: "room-01", online, last_seen: null, fps: null, inference_ms: null,
    mem_used_mb: null, scenario_id: null, run_id: null, last_observation_at: null,
    observation_stale: true, posture: "unknown", posture_confidence: null, posture_model: null } satisfies SourceStatus,
});

describe("status: silence is never safety", () => {
  it("is waiting before anything arrives", () => {
    expect(frameAt(INITIAL_VIEW, T0, OPTS).status).toBe("waiting");
  });

  it("is ok while data is fresh, stale after the threshold", () => {
    const vm = reduce(INITIAL_VIEW, observation(), T0);
    expect(frameAt(vm, T0 + 2900, OPTS).status).toBe("ok");
    expect(frameAt(vm, T0 + 3100, OPTS).status).toBe("stale");
  });

  it("shows offline when the backend says so, until data resumes", () => {
    let vm = reduce(INITIAL_VIEW, observation(), T0);
    vm = reduce(vm, status(false), T0 + 100);
    expect(frameAt(vm, T0 + 200, OPTS).status).toBe("offline");
    vm = reduce(vm, observation(), T0 + 5000);
    expect(frameAt(vm, T0 + 5000, OPTS).status).toBe("ok");
  });

  it("ignores status for other sources", () => {
    let vm = reduce(INITIAL_VIEW, observation(), T0);
    vm = reduce(vm, status(false, "other"), T0);
    expect(frameAt(vm, T0, OPTS).status).toBe("ok");
  });
});

describe("filtering", () => {
  it("ignores other rooms and sources", () => {
    expect(reduce(INITIAL_VIEW, observation({ room_id: "room-02" }), T0, { roomId: "room-01" })).toBe(INITIAL_VIEW);
    expect(reduce(INITIAL_VIEW, observation(), T0, { source: "sim" })).toBe(INITIAL_VIEW);
  });
});

describe("frame contents", () => {
  it("carries the live source type through for the badge", () => {
    const frame = frameAt(reduce(INITIAL_VIEW, observation(), T0), T0, OPTS);
    expect(frame.sourceType).toBe("edge");
    expect(frame.people).toHaveLength(1);
    expect(frame.people[0].personId).toBeNull(); // the fixture predates person_id; absent means single-person
    expect(frame.people[0].facePresent).toBe(true);
    expect(frame.people[0].keypoints).toHaveLength(EDGE.keypoints!.length);
  });

  it("smooths between consecutive observations", () => {
    const moved = EDGE.keypoints!.map((k) => ({ ...k, y: k.y + 0.1 }));
    let vm = reduce(INITIAL_VIEW, observation(), T0);
    vm = reduce(vm, observation({ timestamp: new Date(T0 + 100).toISOString(), keypoints: moved }), T0 + 100);
    const half = frameAt(vm, T0 + 150, { ...OPTS, smoothing: true });
    expect(half.people[0].keypoints[0].y).toBeCloseTo(EDGE.keypoints![0].y + 0.05);
  });
});

describe("interpolateKeypoints", () => {
  it("drops joints the latest frame did not see", () => {
    const [a, b] = EDGE.keypoints!;
    expect(interpolateKeypoints([a, b], [b], 0.5)!.map((k) => k.name)).toEqual([b.name]);
  });
});

describe("several people", () => {
  const person = (id: string, dx: number, at: number): ServerMessage =>
    observation({
      person_id: id,
      timestamp: new Date(T0 + at).toISOString(),
      keypoints: EDGE.keypoints!.map((k) => ({ ...k, x: k.x + dx })),
    });

  it("keeps each person separately and tags them", () => {
    let vm = reduce(INITIAL_VIEW, person("p1", 0, 0), T0);
    vm = reduce(vm, person("p2", 0.2, 0), T0);
    const frame = frameAt(vm, T0, OPTS);
    expect(frame.people.map((p) => p.personId).sort()).toEqual(["p1", "p2"]);
    expect(frame.personDetected).toBe(true);
  });

  it("drops a person the camera stops reporting while others continue", () => {
    let vm = reduce(INITIAL_VIEW, person("p1", 0, 0), T0);
    vm = reduce(vm, person("p2", 0.2, 0), T0);
    for (let ms = 100; ms <= 1000; ms += 100) vm = reduce(vm, person("p1", 0, ms), T0 + ms);
    expect(frameAt(vm, T0 + 1000, OPTS).people.map((p) => p.personId)).toEqual(["p1"]);
  });

  it("keeps everyone on screen (faded by status) when the whole feed goes quiet", () => {
    let vm = reduce(INITIAL_VIEW, person("p1", 0, 0), T0);
    vm = reduce(vm, person("p2", 0.2, 0), T0);
    const frame = frameAt(vm, T0 + 5000, OPTS);
    expect(frame.status).toBe("stale");
    expect(frame.people).toHaveLength(2);
  });

  it("a nobody-visible frame clears everyone", () => {
    let vm = reduce(INITIAL_VIEW, person("p1", 0, 0), T0);
    vm = reduce(vm, observation({ person_detected: false, keypoints: null }), T0 + 100);
    const frame = frameAt(vm, T0 + 100, OPTS);
    expect(frame.people).toEqual([]);
    expect(frame.personDetected).toBe(false);
  });

  it("marks only the alerted person", () => {
    let vm = reduce(INITIAL_VIEW, person("p1", 0, 0), T0);
    vm = reduce(vm, person("p2", 0.2, 0), T0);
    const frame = frameAt(vm, T0, { ...OPTS, alertPersonIds: ["p2"] });
    expect(Object.fromEntries(frame.people.map((p) => [p.personId, p.alert]))).toEqual({ p1: false, p2: true });
  });
});

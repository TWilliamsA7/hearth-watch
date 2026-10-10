import { describe, expect, it } from "vitest";

import type { LiveFrame, LivePerson } from "./generated/sim";
import { LiveClient } from "./live";

function person(id: string, x: number, label = "walking"): LivePerson {
  return {
    person_id: id, heading: 0, height: 1.7, joints: { nose: [x, 1.6, 0], neck: [x, 1.45, 0] },
    label, activity: label, intent: "wander", posture: "standing", destination: null, path: [], conversation: null,
    incident: null, visible: 1,
  };
}

function frame(t: number, people: LivePerson[], state: LiveFrame["state"] = "playing"): LiveFrame {
  return { t, state, speed: 1, people, conversations: [], incidents: [] };
}

describe("LiveClient", () => {
  it("blends people between the frames around the render time, keeping what they're doing", () => {
    const c = new LiveClient("ws://unused");
    c.close();
    c.receive({ type: "frame", data: frame(1.0, [person("p1", 0)]) }, 0);
    c.receive({ type: "frame", data: frame(1.1, [person("p1", 1, "standing")]) }, 100);
    const s = c.sample(1.05);
    expect(s.people[0].joints.nose![0]).toBeCloseTo(0.5);
    expect(s.people[0].label).toBe("walking");
  });

  it("renders a little behind the newest frame while playing, and at it when paused", () => {
    const c = new LiveClient("ws://unused");
    c.close();
    c.receive({ type: "frame", data: frame(5.0, [person("p1", 0)]) }, 0);
    for (let i = 1; i <= 30; i++) c.receive({ type: "frame", data: frame(5.0 + i / 15, [person("p1", 0)]) }, (i * 1000) / 15);
    const t = c.renderTime(2000);
    expect(t).toBeLessThanOrEqual(c.latest!.t);
    expect(t).toBeGreaterThan(c.latest!.t - 0.5);
    c.receive({ type: "frame", data: frame(c.latest!.t, [person("p1", 0)], "paused") }, 2100);
    for (let i = 0; i < 100; i++) c.renderTime(2100 + i * 16);
    expect(c.renderTime(4000)).toBeCloseTo(c.latest!.t, 2);
  });

  it("starts over on a new world or when time goes back (a reset)", () => {
    const c = new LiveClient("ws://unused");
    c.close();
    c.receive({ type: "frame", data: frame(9, [person("p1", 0)]) });
    c.receive({ type: "frame", data: frame(0.05, [person("p1", 0)]) });
    expect(c.frames).toHaveLength(1);
    const rev = c.revision;
    c.receive({ type: "world", data: { run_id: "r" } });
    expect(c.revision).toBe(rev + 1);
    expect(c.frames).toHaveLength(0);
  });
});

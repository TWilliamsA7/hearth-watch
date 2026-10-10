import { describe, expect, it } from "vitest";

import { RecordedFeed, parseStream, type StreamRecord } from "./feed";
import type { ServerMessage } from "./types";

const START = "2026-01-01T00:00:00Z";
const at = (s: number) => new Date(Date.parse(START) + s * 1000).toISOString();

function obs(s: number, id: string): StreamRecord {
  return {
    type: "observation",
    data: {
      schema_version: "1", obs_id: id, timestamp: at(s), source: "sim", source_type: "simulation",
      room_id: "room-01", scenario_id: "t", run_id: null, person_id: null, person_detected: false, keypoints: null,
      features: null, posture_prediction: null, hand_gesture_prediction: null, face_present: false, face_confidence: null,
    },
  };
}
function beat(s: number): StreamRecord {
  return {
    type: "heartbeat",
    data: { schema_version: "1", timestamp: at(s), source: "sim", source_type: "simulation", room_id: "room-01",
      scenario_id: "t", run_id: null, fps: 12, inference_ms: null, mem_used_mb: null },
  };
}

function collect(feed: RecordedFeed): string[] {
  const seen: string[] = [];
  feed.subscribe((m: ServerMessage) => m.type === "observation" && seen.push(m.data.obs_id));
  return seen;
}

describe("parseStream", () => {
  it("reads the simulator's stream file format", () => {
    const text = [beat(0), obs(0, "a")].map((r) => JSON.stringify(r)).join("\n") + "\n";
    expect(parseStream(text).map((r) => r.type)).toEqual(["heartbeat", "observation"]);
  });
});

describe("RecordedFeed", () => {
  const records = [beat(0), obs(0, "a"), obs(0.5, "b"), beat(1), obs(1.0, "c"), obs(2.0, "d")];

  it("emits observations as ServerMessages when due, and never heartbeats", () => {
    const feed = new RecordedFeed(records, { start: START, duration: 3 });
    const seen = collect(feed);
    feed.tick(0);
    expect(seen).toEqual(["a"]);
    feed.tick(1000);
    expect(seen).toEqual(["a", "b", "c"]);
  });

  it("scales time by speed and reports the feed clock in record time", () => {
    const feed = new RecordedFeed(records, { start: START, duration: 3 });
    const seen = collect(feed);
    feed.setSpeed(4);
    feed.tick(500);
    expect(seen).toEqual(["a", "b", "c", "d"]);
    expect(feed.now()).toBe(Date.parse(START) + 2000);
  });

  it("seek replays the last second so the view can rebuild everyone", () => {
    const feed = new RecordedFeed(records, { start: START, duration: 3 });
    const seen = collect(feed);
    feed.seek(1.2);
    expect(seen).toEqual(["a", "b", "c"]); // the second up to the last observation (1.0 s)
    feed.tick(1000);
    expect(seen).toEqual(["a", "b", "c", "d"]);
  });

  it("stops at the end of the run", () => {
    const feed = new RecordedFeed(records, { start: START, duration: 3 });
    feed.tick(10_000);
    expect(feed.state.t).toBe(3);
  });
});

describe("arrival times", () => {
  it("stamps late deliveries (seek) with when the record was due, so stale data looks stale", () => {
    const feed = new RecordedFeed([obs(0, "a"), obs(1, "b")], { start: START, duration: 10 });
    const arrivals: number[] = [];
    feed.subscribe((_m, receivedAt) => arrivals.push(receivedAt));
    feed.seek(8);
    expect(arrivals).toEqual([Date.parse(START), Date.parse(START) + 1000]);
    expect(feed.now() - arrivals[arrivals.length - 1]).toBe(7000);
  });
});

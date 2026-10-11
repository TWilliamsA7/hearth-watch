import { describe, expect, it } from "vitest";

import type { LiveWorld } from "./generated/sim";
import { actionsFor, nearestSlot } from "./pick";

const world = {
  room: { objects: [{ kind: "couch", variant: "couch" }, { kind: "chair", variant: "chair" }] },
  slots: [
    { slot_id: "piece-0-sit1", piece: 0, kind: "sit", at: [-0.5, -2], heading: 0, height: 0.43, group: "piece-0" },
    { slot_id: "piece-0-sit2", piece: 0, kind: "sit", at: [0.5, -2], heading: 0, height: 0.43, group: "piece-0" },
    { slot_id: "piece-0-lie", piece: 0, kind: "lie", at: [0, -2], heading: 1.57, height: 0.43, group: "piece-0" },
    { slot_id: "piece-1-sit", piece: 1, kind: "sit", at: [2, -1], heading: 3.14, height: 0.485, group: "piece-1" },
  ],
} as unknown as LiveWorld;

describe("picking", () => {
  it("takes the slot on the clicked piece nearest the click", () => {
    expect(nearestSlot(world.slots, 0, 0.45, -1.9)?.slot_id).toBe("piece-0-sit2");
    expect(nearestSlot(world.slots, 1, 0, 0)?.slot_id).toBe("piece-1-sit");
    expect(nearestSlot(world.slots, 2, 0, 0)).toBeNull();
  });

  it("offers sitting, and lying where the piece has room to lie", () => {
    expect(actionsFor({ kind: "slot", slotId: "piece-0-sit1", x: 0, z: 0 }, world).map(([, c]) => c.action)).toEqual(["sit", "lie"]);
    expect(actionsFor({ kind: "slot", slotId: "piece-1-sit", x: 0, z: 0 }, world).map(([, c]) => c.action)).toEqual(["sit"]);
    expect(actionsFor({ kind: "slot", slotId: "piece-0-lie", x: 0, z: 0 }, world).map(([, c]) => c.action)).toEqual(["lie"]);
  });

  it("offers floor actions at the clicked point", () => {
    const acts = actionsFor({ kind: "floor", x: 1.23456, z: -2.5 }, world);
    expect(acts.map(([, c]) => c.action)).toEqual(["go", "kneel", "pick_up", "fall"]);
    expect(acts[0][1].at).toEqual([1.235, -2.5]);
  });
});

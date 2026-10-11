import * as THREE from "three";
import { describe, expect, it } from "vitest";

import { buildFurniture } from "./furniture3d";
import type { WorldObject } from "./types";

/** The top of the model's parts whose footprint covers the point (x, z), in the piece's own axes. */
function topAt(g: THREE.Group, x: number, z: number, below = Infinity): number {
  g.position.set(0, 0, 0);
  g.rotation.set(0, 0, 0);
  g.updateMatrixWorld(true);
  let top = 0;
  for (const child of g.children) {
    const box = new THREE.Box3().setFromObject(child);
    if (box.min.x <= x && x <= box.max.x && box.min.z <= z && z <= box.max.z && box.max.y < below) top = Math.max(top, box.max.y);
  }
  return top;
}

const obj = (variant: string, size: [number, number, number], extra: Partial<WorldObject> = {}): WorldObject =>
  ({ kind: "chair", center: [0, size[1] / 2, 0], size, rotation: 0, variant, ...extra }) as WorldObject;

describe("buildFurniture", () => {
  it("puts a chair's seat at the given seat height", () => {
    const g = buildFurniture(obj("chair", [0.48, 0.9, 0.5], { seat_height: 0.52, back_depth: 0.05 }));
    expect(topAt(g, 0, 0.05, 0.7)).toBeCloseTo(0.52, 2);
  });

  it("puts a couch's cushions at the seat height and its arms where they're said to be", () => {
    const g = buildFurniture(obj("couch", [2.0, 0.85, 0.9], { seat_height: 0.45, arm_width: 0.16, arm_height: 0.62, back_depth: 0.31 }));
    expect(topAt(g, 0, 0.2, 0.5)).toBeCloseTo(0.45, 2);
    expect(topAt(g, 0.95, 0.2)).toBeCloseTo(0.62, 2);
    expect(topAt(g, 0.80, 0.2, 0.5)).toBeCloseTo(0.45, 2); // just inside the arm: still cushion
  });

  it("brings a couch's back cushions forward to the back depth", () => {
    const g = buildFurniture(obj("couch", [2.0, 0.85, 0.9], { seat_height: 0.43, back_depth: 0.31 }));
    g.updateMatrixWorld(true);
    // The back cushions: above the seat, between the arms, narrower than the frame.
    const cushions = g.children
      .map((c) => new THREE.Box3().setFromObject(c))
      .filter((b) => b.min.y > 0.4 && b.max.y > 0.6 && b.max.x - b.min.x < 1.0);
    expect(cushions).toHaveLength(3);
    const front = Math.max(...cushions.map((b) => b.max.z));
    expect(front).toBeCloseTo(-0.45 + 0.31, 2);
  });

  it("puts a bed's top at the lying surface", () => {
    const g = buildFurniture(obj("bed", [1.05, 0.55, 2.05], { surface_height: 0.6, seat_height: 0.6, back_depth: 0.06 }));
    expect(topAt(g, 0, 0.5, 0.62)).toBeCloseTo(0.6, 2);
  });

  it("keeps its old proportions when the object doesn't say (scripted rooms)", () => {
    const g = buildFurniture(obj("couch", [2.0, 0.85, 0.9]));
    expect(topAt(g, 0, 0.2, 0.5)).toBeCloseTo(0.43, 2);
  });
});

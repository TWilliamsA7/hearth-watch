import * as THREE from "three";
import { describe, expect, it } from "vitest";

import { buildFurnitureDebug, DEBUG_COLORS } from "./furnitureDebug";
import type { LiveWorld } from "./generated/sim";

const world = {
  slots: [
    { slot_id: "piece-0-sit1", piece: 0, kind: "sit", at: [1, -2], heading: Math.PI / 2, height: 0.43, group: "piece-0", stand: [1.35, -2], approach: [1.9, -2] },
    { slot_id: "piece-0-lie", piece: 0, kind: "lie", at: [1, -2.5], heading: 0, height: 0.43, group: "piece-0", stand: null, approach: null },
  ],
  places: [{ name: "middle", at: [0, -3], heading: 0 }],
  nav: { cell: 0.1, origin: [-2, -4], cols: 4, rows: ["0110", "0000", "1000"] },
} as unknown as LiveWorld;

const slotObjects = (g: THREE.Object3D, id: string) => {
  const out: THREE.Object3D[] = [];
  g.traverse((o) => o.userData.slotId === id && out.push(o));
  return out;
};

describe("buildFurnitureDebug", () => {
  it("marks each slot where its hips go, at the seat's height", () => {
    const d = buildFurnitureDebug(world);
    d.slots.updateMatrixWorld(true);
    const [sit] = slotObjects(d.slots, "piece-0-sit1");
    const at = sit.getWorldPosition(new THREE.Vector3());
    expect([at.x, at.z]).toEqual([1, -2]);
    expect(at.y).toBeCloseTo(0.442, 3);
    expect(slotObjects(d.slots, "piece-0-lie")).toHaveLength(1);
    d.dispose();
  });

  it("points a seat's arrow the way the person will face", () => {
    const d = buildFurnitureDebug(world);
    d.slots.updateMatrixWorld(true);
    const arrow = slotObjects(d.slots, "piece-0-sit1").find((o) => (o as THREE.Mesh).geometry?.attributes.position.count === 3)! as THREE.Mesh;
    const tip = new THREE.Vector3(0, 0, 0.24).applyMatrix4(arrow.matrixWorld);
    expect(tip.x).toBeCloseTo(1.24, 3); // heading pi/2 faces +x
    expect(tip.z).toBeCloseTo(-2, 3);
    d.dispose();
  });

  it("tints only the blocked floor cells", () => {
    const d = buildFurnitureDebug(world);
    const cells = d.nav as THREE.InstancedMesh;
    expect(cells.count).toBe(3);
    const m = new THREE.Matrix4();
    cells.getMatrixAt(2, m);
    const p = new THREE.Vector3().setFromMatrixPosition(m);
    expect(p.x).toBeCloseTo(-2, 5); // row 2, column 0
    expect(p.z).toBeCloseTo(-3.8, 5);
    d.dispose();
  });

  it("colours a slot as taken while someone holds it", () => {
    const d = buildFurnitureDebug(world);
    const mat = () => ((slotObjects(d.slots, "piece-0-sit1")[0] as THREE.Mesh).material as THREE.MeshBasicMaterial).color.getHexString();
    expect(`#${mat()}`).toBe(DEBUG_COLORS.free);
    d.update({ "piece-0-sit1": "p2" });
    expect(`#${mat()}`).toBe(DEBUG_COLORS.taken);
    d.update({});
    expect(`#${mat()}`).toBe(DEBUG_COLORS.free);
    d.dispose();
  });
});

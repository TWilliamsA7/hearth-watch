import * as THREE from "three";

import type { LiveWorld } from "./generated/sim";

/**
 * What the simulator plans with, drawn over the room for the Sim Studio: where people sit
 * and lie (green when free, amber when taken), the way up to each seat (open floor to turn
 * round on, then the spot to step back to), places people stand on purpose, and the floor
 * a body's centre can walk on (the rest tinted red). Static per world; update() recolours
 * the slots as people take and leave them.
 */
export interface FurnitureDebug {
  /** Slots, their way in, and places. Meshes carry userData.slotId for picking. */
  slots: THREE.Group;
  /** Blocked floor. */
  nav: THREE.Object3D;
  update(taken: Readonly<Record<string, string>>): void;
  dispose(): void;
}

export const DEBUG_COLORS = { free: "#4ade80", taken: "#f59e0b", lie: "#a78bfa", way: "#7cc4ff", place: "#f472b6", blocked: "#ef4444" };


export function buildFurnitureDebug(world: LiveWorld): FurnitureDebug {
  const slots = new THREE.Group();
  slots.name = "furniture-slots";
  const owned: { dispose(): void }[] = [];
  const own = <T extends { dispose(): void }>(x: T): T => {
    owned.push(x);
    return x;
  };
  const slotMats = new Map<string, THREE.MeshBasicMaterial | THREE.LineBasicMaterial>();

  const disc = own(new THREE.CircleGeometry(0.11, 24));
  const arrow = own(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.06, 0, 0.1), new THREE.Vector3(0.06, 0, 0.1), new THREE.Vector3(0, 0, 0.24)]));
  arrow.setIndex([0, 1, 2]);
  const dot = own(new THREE.CircleGeometry(0.05, 16));
  const wayMat = own(new THREE.LineDashedMaterial({ color: DEBUG_COLORS.way, dashSize: 0.06, gapSize: 0.05, transparent: true, opacity: 0.9 }));
  const wayDot = own(new THREE.MeshBasicMaterial({ color: DEBUG_COLORS.way, side: THREE.DoubleSide }));
  const placeMat = own(new THREE.MeshBasicMaterial({ color: DEBUG_COLORS.place, side: THREE.DoubleSide }));
  const diamond = own(new THREE.CircleGeometry(0.08, 4));

  const flat = (geo: THREE.BufferGeometry, mat: THREE.Material, x: number, y: number, z: number, heading = 0) => {
    const m = new THREE.Mesh(geo, mat);
    m.rotation.set(-Math.PI / 2, 0, 0);
    const holder = new THREE.Group();
    holder.add(m);
    holder.position.set(x, y, z);
    holder.rotation.y = heading; // shapes face +z; heading 0 faces +z, pi/2 faces +x
    return { holder, mesh: m };
  };

  for (const s of world.slots ?? []) {
    const y = s.height + 0.012;
    if (s.kind === "sit") {
      const mat = own(new THREE.MeshBasicMaterial({ color: DEBUG_COLORS.free, side: THREE.DoubleSide, transparent: true, opacity: 0.85, depthTest: false }));
      slotMats.set(s.slot_id, mat);
      const { holder, mesh } = flat(disc, mat, s.at[0], y, s.at[1], s.heading);
      mesh.userData.slotId = s.slot_id;
      const tip = new THREE.Mesh(arrow, mat);
      tip.userData.slotId = s.slot_id;
      holder.add(tip);
      holder.renderOrder = 2;
      slots.add(holder);
      // The way in: open floor to turn round on, then a step back to the stand point.
      const pts = [s.approach, s.stand].filter((p): p is [number, number] => !!p).map(([x, z]) => new THREE.Vector3(x, 0.02, z));
      if (pts.length) {
        pts.push(new THREE.Vector3(s.at[0], 0.02, s.at[1]));
        const line = new THREE.Line(own(new THREE.BufferGeometry().setFromPoints(pts)), wayMat);
        line.computeLineDistances();
        slots.add(line);
        for (const p of pts.slice(0, -1)) slots.add(flat(dot, wayDot, p.x, 0.018, p.z).holder);
      }
    } else {
      // Lying: the outline of an average body along the heading, head to feet.
      const mat = own(new THREE.LineBasicMaterial({ color: DEBUG_COLORS.lie, transparent: true, opacity: 0.9, depthTest: false }));
      slotMats.set(s.slot_id, mat);
      const [hw, hl] = [0.22, 0.85];
      const outline = new THREE.LineLoop(own(new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-hw, 0, -hl), new THREE.Vector3(hw, 0, -hl), new THREE.Vector3(hw, 0, hl), new THREE.Vector3(-hw, 0, hl),
      ])), mat);
      outline.position.set(s.at[0], y, s.at[1]);
      outline.rotation.y = s.heading;
      outline.userData.slotId = s.slot_id;
      outline.renderOrder = 2;
      slots.add(outline);
    }
  }
  for (const p of world.places ?? []) {
    const { holder } = flat(diamond, placeMat, p.at[0], 0.02, p.at[1], p.heading);
    const tick = new THREE.Mesh(arrow, placeMat);
    tick.scale.setScalar(0.6);
    holder.add(tick);
    slots.add(holder);
  }

  // Blocked floor: one instanced square per cell a body's centre can't be on.
  let nav: THREE.Object3D = new THREE.Group();
  const grid = world.nav;
  if (grid) {
    const blocked: [number, number][] = [];
    grid.rows.forEach((row, j) => {
      for (let i = 0; i < row.length; i++) if (row[i] === "1") blocked.push([i, j]);
    });
    const square = own(new THREE.PlaneGeometry(grid.cell * 0.92, grid.cell * 0.92));
    square.rotateX(-Math.PI / 2);
    const mat = own(new THREE.MeshBasicMaterial({ color: DEBUG_COLORS.blocked, transparent: true, opacity: 0.22, depthWrite: false }));
    const cells = new THREE.InstancedMesh(square, mat, Math.max(blocked.length, 1));
    cells.count = blocked.length;
    const m = new THREE.Matrix4();
    blocked.forEach(([i, j], k) => cells.setMatrixAt(k, m.makeTranslation(grid.origin[0] + i * grid.cell, 0.01, grid.origin[1] + j * grid.cell)));
    cells.instanceMatrix.needsUpdate = true;
    cells.frustumCulled = false;
    cells.name = "nav-blocked";
    own(cells);
    nav = cells;
  }

  let last = "";
  return {
    slots,
    nav,
    update(taken) {
      const key = JSON.stringify(taken);
      if (key === last) return;
      last = key;
      for (const [id, mat] of slotMats) {
        const lie = mat instanceof THREE.LineBasicMaterial;
        mat.color.set(taken[id] ? DEBUG_COLORS.taken : lie ? DEBUG_COLORS.lie : DEBUG_COLORS.free);
      }
    },
    dispose() {
      for (const x of owned) x.dispose();
    },
  };
}

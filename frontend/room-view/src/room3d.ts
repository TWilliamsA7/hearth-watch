import * as THREE from "three";

import type { Theme } from "./draw";
import { buildFurniture } from "./furniture3d";
import { paintedWall, woodFloor } from "./textures";
import type { WorldRoom } from "./types";

/**
 * The simulated room as a lit, furnished space: plank floor, painted walls with
 * baseboards, door openings, a window letting light in, furniture models, a ceiling lamp
 * and the simulated camera with its field of view.
 *
 * Walls between the viewer and the room fade out as the view orbits (a dollhouse
 * cutaway), so the room is always visible from outside without hiding its far walls.
 *
 * Live rooms carry their walls (bounds), doors and windows. Scripted rooms don't; they get
 * the box the scripted 3D view always drew (as wide as the camera sees at the back wall,
 * with a little floor in front of depth 0) and no openings.
 */

export type Lighting = "day" | "evening" | "dim";

export interface Room3D {
  group: THREE.Group;
  /** The floor's walls, faded when the viewer is outside them. Call every frame. */
  update(camera: THREE.Camera): void;
  /** The simulated camera's frustum lines (toggle with .visible). */
  frustum: THREE.Object3D;
  dispose(): void;
}

const FRONT_MARGIN = 0.6;
const BACK_MARGIN = 0.4;
const WALL_T = 0.1;
const DOOR_H = 2.05;

interface Wall {
  mesh: THREE.Object3D;
  /** Inward normal (x, z) and a point on the wall line. */
  normal: [number, number];
  at: [number, number];
  materials: THREE.MeshStandardMaterial[];
}

export function roomBounds(room: WorldRoom): [number, number, number, number] {
  if (room.bounds) return room.bounds as [number, number, number, number];
  const w = room.width_back;
  return [-w / 2, -room.depth - BACK_MARGIN, w / 2, FRONT_MARGIN];
}

export function buildRoom3D(room: WorldRoom, theme: Theme, lighting: Lighting = (room.lighting as Lighting) ?? "day"): Room3D {
  const group = new THREE.Group();
  const [x0, z0, x1, z1] = roomBounds(room);
  const W = x1 - x0;
  const D = z1 - z0;
  const H = room.height;
  const cx = (x0 + x1) / 2;
  const cz = (z0 + z1) / 2;

  // Floor.
  const floorTex = woodFloor().clone();
  floorTex.needsUpdate = true;
  floorTex.repeat.set(W / 2.2, D / 2.2);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(W, D), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.55, metalness: 0.0 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(cx, 0, cz);
  floor.receiveShadow = true;
  group.add(floor);

  // A plinth under the room so it reads as a model on a table, not a floating plane.
  const plinth = new THREE.Mesh(
    new THREE.BoxGeometry(W + 2 * WALL_T + 0.3, 0.12, D + 2 * WALL_T + 0.3),
    new THREE.MeshStandardMaterial({ color: theme.background, roughness: 1 }),
  );
  plinth.position.set(cx, -0.061, cz);
  plinth.receiveShadow = true;
  group.add(plinth);

  // Walls, with door and window openings cut out.
  const wallColor = lighting === "day" ? "#e9e4da" : lighting === "evening" ? "#d9cbb5" : "#b9b4ab";
  const walls: Wall[] = [];
  const doors = room.doors ?? [];
  const windows = room.windows ?? [];
  const sides: { name: "back" | "front" | "left" | "right"; a: [number, number]; b: [number, number]; n: [number, number] }[] = [
    { name: "back", a: [x0, z0], b: [x1, z0], n: [0, 1] },
    { name: "right", a: [x1, z0], b: [x1, z1], n: [-1, 0] },
    { name: "front", a: [x1, z1], b: [x0, z1], n: [0, -1] },
    { name: "left", a: [x0, z1], b: [x0, z0], n: [1, 0] },
  ];
  for (const s of sides) {
    const len = Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]);
    const holes: { u: number; w: number; y0: number; y1: number }[] = [];
    const along = (p: [number, number]) => Math.hypot(p[0] - s.a[0], p[1] - s.a[1]);
    for (const d of doors) if (d.wall === s.name) holes.push({ u: along(d.center), w: d.width, y0: 0, y1: DOOR_H });
    for (const w of windows) if (w.wall === s.name) holes.push({ u: along(w.center), w: w.width, y0: w.sill, y1: w.sill + w.height });
    walls.push(buildWall(s.a, s.b, s.n, len, H, holes, wallColor));
    group.add(walls[walls.length - 1].mesh);
  }

  // Door frames and doors standing ajar; window frames and glass.
  const trim = new THREE.MeshStandardMaterial({ color: "#f4f1ea", roughness: 0.5 });
  const doorMat = new THREE.MeshStandardMaterial({ color: "#8b6a4a", roughness: 0.6 });
  for (const d of doors) {
    const f = frameAt(d.center, d.wall, d.width, 0, DOOR_H, trim);
    group.add(f);
    const leaf = new THREE.Mesh(new THREE.BoxGeometry(d.width - 0.06, DOOR_H - 0.04, 0.04), doorMat);
    leaf.castShadow = true;
    const hinge = new THREE.Group();
    const [nx, nz] = inward(d.wall);
    const [tx, tz] = [-nz, nx]; // along the wall
    hinge.position.set(d.center[0] - tx * (d.width / 2), 0, d.center[1] - tz * (d.width / 2));
    hinge.rotation.y = Math.atan2(tx, tz) - Math.PI / 2 + 1.25; // swung open into the room
    leaf.position.set((d.width - 0.06) / 2, (DOOR_H - 0.04) / 2, 0);
    hinge.add(leaf);
    group.add(hinge);
    // A dim corridor beyond the opening.
    const beyond = new THREE.Mesh(new THREE.PlaneGeometry(d.width, DOOR_H), new THREE.MeshBasicMaterial({ color: "#2a2723" }));
    beyond.position.set(d.center[0] - nx * 0.35, DOOR_H / 2, d.center[1] - nz * 0.35);
    beyond.rotation.y = Math.atan2(nx, nz);
    group.add(beyond);
  }
  const outside = lighting === "day" ? "#cfe6ff" : lighting === "evening" ? "#2b3550" : "#141822";
  for (const w of windows) {
    group.add(frameAt(w.center, w.wall, w.width, w.sill, w.sill + w.height, trim));
    const [nx, nz] = inward(w.wall);
    const glass = new THREE.Mesh(
      new THREE.PlaneGeometry(w.width, w.height),
      new THREE.MeshBasicMaterial({ color: outside, toneMapped: false }),
    );
    glass.position.set(w.center[0] - nx * 0.04, w.sill + w.height / 2, w.center[1] - nz * 0.04);
    glass.rotation.y = Math.atan2(nx, nz);
    group.add(glass);
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.04, w.height, 0.04), trim);
    bar.position.copy(glass.position);
    bar.rotation.y = glass.rotation.y;
    group.add(bar);
  }

  // Furniture.
  for (const o of room.objects) group.add(buildFurniture(o));

  // Lights: a sky fill, a key light (through the window by day), and a ceiling lamp.
  group.add(...lights(room, lighting, [x0, z0, x1, z1], H));

  // The simulated camera and its field of view.
  const cam = cameraMarker(room, theme);
  group.add(cam.body);
  group.add(cam.frustum);

  const v = new THREE.Vector3();
  return {
    group,
    frustum: cam.frustum,
    update(camera: THREE.Camera) {
      camera.getWorldPosition(v);
      for (const w of walls) {
        // Is the viewer on the outside of this wall (and above the floor)?
        const outsideIt = (v.x - w.at[0]) * w.normal[0] + (v.z - w.at[1]) * w.normal[1] < 0;
        const target = outsideIt && v.y > 0.3 ? 0.08 : 1;
        for (const m of w.materials) {
          m.opacity += (target - m.opacity) * 0.25;
          const see = m.opacity < 0.98;
          m.transparent = see;
          m.depthWrite = !see;
        }
        w.mesh.traverse((o) => {
          if ((o as THREE.Mesh).isMesh) (o as THREE.Mesh).castShadow = !outsideIt;
        });
      }
    },
    dispose() {
      disposeTree(group);
    },
  };
}

function inward(wall: string): [number, number] {
  return ({ back: [0, 1], front: [0, -1], left: [1, 0], right: [-1, 0] } as Record<string, [number, number]>)[wall] ?? [0, 1];
}

/** A wall as a shape with holes, extruded to its thickness, plus a baseboard. */
function buildWall(
  a: [number, number], b: [number, number], n: [number, number], len: number, H: number,
  holes: { u: number; w: number; y0: number; y1: number }[], color: string,
): Wall {
  const sorted = [...holes].sort((p, q) => p.u - q.u);
  const clip = (h: { u: number; w: number }) => [Math.max(0.05, h.u - h.w / 2), Math.min(len - 0.05, h.u + h.w / 2)];
  // Doors reach the floor: notches in the outline. Windows: holes.
  const notches = sorted.filter((h) => h.y0 <= 0.001);
  const outline: [number, number][] = [[0, 0]];
  for (const h of notches) {
    const [u0, u1] = clip(h);
    outline.push([u0, 0], [u0, h.y1], [u1, h.y1], [u1, 0]);
  }
  outline.push([len, 0], [len, H], [0, H]);
  const shape = new THREE.Shape(outline.map(([x, y]) => new THREE.Vector2(x, y)));
  for (const h of sorted.filter((h) => h.y0 > 0.001)) {
    const [u0, u1] = clip(h);
    shape.holes.push(new THREE.Path([new THREE.Vector2(u0, h.y0), new THREE.Vector2(u1, h.y0),
      new THREE.Vector2(u1, h.y1), new THREE.Vector2(u0, h.y1)]));
  }
  const tex = paintedWall();
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.95, map: tex, side: THREE.DoubleSide });
  const geo = new THREE.ExtrudeGeometry(shape, { depth: WALL_T, bevelEnabled: false });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.receiveShadow = true;
  // Local x along the wall from a to b, y up, extruded outwards (away from the room).
  const g = new THREE.Group();
  g.add(mesh);
  const angle = Math.atan2(-(b[1] - a[1]), b[0] - a[0]);
  g.rotation.y = angle;
  g.position.set(a[0], 0, a[1]);
  // ExtrudeGeometry fills local z from 0 to WALL_T: put that on the outside of the wall line.
  const zOut = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), angle);
  mesh.position.z = zOut.x * -n[0] + zOut.z * -n[1] > 0 ? 0 : -WALL_T;
  // Baseboard along the inside face.
  const baseMat = new THREE.MeshStandardMaterial({ color: "#f2efe8", roughness: 0.6 });
  const segs = [...notches].sort((p, q) => p.u - q.u);
  let u = 0;
  const addBoard = (from: number, to: number) => {
    if (to - from < 0.02) return;
    const board = new THREE.Mesh(new THREE.BoxGeometry(to - from, 0.1, 0.018), baseMat);
    board.position.set((from + to) / 2, 0.05, 0);
    g.add(board);
  };
  for (const h of segs) {
    addBoard(u, h.u - h.w / 2);
    u = h.u + h.w / 2;
  }
  addBoard(u, len);
  // Boards sit on the room side of the wall line.
  const inLocal = new THREE.Vector3(n[0], 0, n[1]).applyAxisAngle(new THREE.Vector3(0, 1, 0), -angle);
  for (const c of g.children) if (c !== mesh) c.position.z = inLocal.z * 0.009;
  return { mesh: g, normal: n, at: a, materials: [mat, baseMat] };
}

function frameAt(center: [number, number], wall: string, width: number, y0: number, y1: number, mat: THREE.Material): THREE.Group {
  const g = new THREE.Group();
  const [nx, nz] = inward(wall);
  const t = 0.06;
  const parts: [number, number, number, number][] = [
    [-width / 2 - t / 2, (y0 + y1) / 2, t, y1 - y0 + t],
    [width / 2 + t / 2, (y0 + y1) / 2, t, y1 - y0 + t],
    [0, y1 + t / 2, width + 2 * t, t],
  ];
  if (y0 > 0) parts.push([0, y0 - t / 2, width + 2 * t, t]);
  for (const [x, y, w, h] of parts) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, WALL_T + 0.04), mat);
    m.position.set(x, y, 0);
    m.castShadow = true;
    g.add(m);
  }
  g.position.set(center[0] - nx * (WALL_T / 2), 0, center[1] - nz * (WALL_T / 2));
  g.rotation.y = Math.atan2(nx, nz);
  return g;
}

function lights(room: WorldRoom, lighting: Lighting, b: [number, number, number, number], H: number): THREE.Object3D[] {
  const [x0, z0, x1, z1] = b;
  const cx = (x0 + x1) / 2;
  const cz = (z0 + z1) / 2;
  const out: THREE.Object3D[] = [];
  const sky = { day: 1.1, evening: 0.35, dim: 0.12 }[lighting];
  out.push(new THREE.HemisphereLight(lighting === "day" ? "#eef4ff" : "#ffe2c4", "#5a4a3a", sky));
  out.push(new THREE.AmbientLight("#ffffff", { day: 0.25, evening: 0.12, dim: 0.05 }[lighting]));

  // Key light: from the window by day, else from the ceiling lamp's direction.
  const win = room.windows?.[0];
  const key = new THREE.DirectionalLight(lighting === "day" ? "#fff6e8" : "#ffd9a8", { day: 2.4, evening: 0.6, dim: 0.15 }[lighting]);
  if (win && lighting === "day") {
    const [nx, nz] = inward(win.wall);
    key.position.set(win.center[0] - nx * 3 + 1.5, H + 2.5, win.center[1] - nz * 3);
  } else {
    key.position.set(cx + 1.5, H + 3, cz + 2);
  }
  key.target.position.set(cx, 0, cz);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  const span = Math.max(x1 - x0, z1 - z0) * 0.75 + 1;
  Object.assign(key.shadow.camera, { left: -span, right: span, top: span, bottom: -span, near: 0.5, far: 25 });
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.02;
  key.shadow.radius = 4;
  out.push(key, key.target);

  // The ceiling lamp: a fixture and a warm point light.
  const lampOn = lighting !== "day";
  const lamp = new THREE.PointLight("#ffd29a", { day: 0, evening: 14, dim: 4 }[lighting], 0, 2);
  lamp.position.set(cx, H - 0.35, cz);
  lamp.castShadow = lampOn;
  lamp.shadow.mapSize.set(1024, 1024);
  lamp.shadow.bias = -0.001;
  out.push(lamp);
  const shade = new THREE.Mesh(
    new THREE.CylinderGeometry(0.12, 0.28, 0.2, 24, 1, true),
    new THREE.MeshStandardMaterial({ color: "#f1e6d0", side: THREE.DoubleSide, emissive: lampOn ? "#ffcf8a" : "#000000", emissiveIntensity: lampOn ? 0.6 : 0 }),
  );
  shade.position.set(cx, H - 0.3, cz);
  const cord = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.2, 6), new THREE.MeshBasicMaterial({ color: "#333" }));
  cord.position.set(cx, H - 0.1, cz);
  out.push(shade, cord);
  return out;
}

function cameraMarker(room: WorldRoom, theme: Theme): { body: THREE.Object3D; frustum: THREE.Object3D } {
  const [px, py, pz] = room.camera_position;
  const [tx, ty, tz] = room.camera_target;
  const body = new THREE.Group();
  const shell = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.09, 0.16), new THREE.MeshStandardMaterial({ color: "#2b2f36", roughness: 0.4 }));
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.03, 16), new THREE.MeshStandardMaterial({ color: "#111", metalness: 0.5, roughness: 0.2 }));
  lens.rotation.x = Math.PI / 2;
  lens.position.z = 0.09;
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.008, 8, 6), new THREE.MeshBasicMaterial({ color: theme.badge.simulation }));
  led.position.set(0.04, 0.03, 0.08);
  body.add(shell, lens, led);
  body.position.set(px, py, pz);
  body.lookAt(tx, ty, tz);

  // Frustum: rays from the lens through the image corners, out to where they meet the floor or 9 m.
  const aspect = room.camera_aspect ?? room.width_front / Math.max(1e-3, room.height);
  const cam = new THREE.PerspectiveCamera(room.camera_fov, room.camera_aspect ?? 4 / 3, 0.05, 30);
  cam.position.set(px, py, pz);
  cam.lookAt(tx, ty, tz);
  cam.updateMatrixWorld(true);
  void aspect;
  const origin = new THREE.Vector3(px, py, pz);
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) => {
    const p = new THREE.Vector3(x, y, 0.5).unproject(cam);
    const dir = p.sub(origin).normalize();
    // Out to wherever the ray leaves the room: floor, ceiling or a wall.
    const [bx0, bz0, bx1, bz1] = roomBounds(room);
    let t = 12;
    const hit = (num: number, den: number) => {
      if (Math.abs(den) > 1e-6) {
        const s = num / den;
        if (s > 0.05) t = Math.min(t, s);
      }
    };
    hit(-py, dir.y);
    hit(room.height - py, dir.y);
    hit(bx0 - px, dir.x);
    hit(bx1 - px, dir.x);
    hit(bz0 - pz, dir.z);
    hit(bz1 - pz, dir.z);
    return origin.clone().addScaledVector(dir, t);
  });
  const pts: THREE.Vector3[] = [];
  for (const c of corners) pts.push(origin, c);
  corners.forEach((c, i) => pts.push(c, corners[(i + 1) % 4]));
  const lines = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: theme.badge.simulation, transparent: true, opacity: 0.5 }),
  );
  const frustum = new THREE.Group();
  frustum.add(lines);
  const coverage = floorCoverage(cam, origin, room, theme);
  if (coverage) frustum.add(coverage);
  return { body, frustum };
}

/**
 * The patch of floor the camera sees (where a person's feet are in view), clipped to the room:
 * rays through the image border, met with the floor. Shown with the frustum, to see which parts
 * of a tracked area the camera covers.
 */
function floorCoverage(cam: THREE.PerspectiveCamera, origin: THREE.Vector3, room: WorldRoom, theme: Theme): THREE.Object3D | null {
  const [bx0, bz0, bx1, bz1] = roomBounds(room);
  const border: [number, number][] = [];
  const N = 24;
  for (let i = 0; i < N; i++) border.push([-1 + (2 * i) / N, -1]);
  for (let i = 0; i < N; i++) border.push([1, -1 + (2 * i) / N]);
  for (let i = 0; i < N; i++) border.push([1 - (2 * i) / N, 1]);
  for (let i = 0; i < N; i++) border.push([-1, 1 - (2 * i) / N]);
  const shape: THREE.Vector2[] = [];
  for (const [x, y] of border) {
    const dir = new THREE.Vector3(x, y, 0.5).unproject(cam).sub(origin).normalize();
    // A ray above the horizon never meets the floor: cap it where it leaves the room.
    const s = dir.y < -1e-3 ? -origin.y / dir.y : 30;
    const px = Math.min(Math.max(origin.x + dir.x * s, bx0), bx1);
    const pz = Math.min(Math.max(origin.z + dir.z * s, bz0), bz1);
    shape.push(new THREE.Vector2(px, -pz)); // the mesh is laid flat by a -90 degree turn about x
  }
  if (shape.length < 3) return null;
  const mesh = new THREE.Mesh(
    new THREE.ShapeGeometry(new THREE.Shape(shape)),
    new THREE.MeshBasicMaterial({ color: theme.badge.simulation, transparent: true, opacity: 0.12, depthWrite: false, side: THREE.DoubleSide }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.012;
  mesh.renderOrder = 1;
  return mesh;
}

export function disposeTree(object: THREE.Object3D): void {
  object.traverse((o) => {
    const mesh = o as THREE.Mesh;
    mesh.geometry?.dispose();
    const mats = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
    for (const m of mats) {
      // Shared procedural textures stay cached; clones are disposed with their material.
      m.dispose();
    }
  });
}

/** Renderer settings for the lit room: colour management, filmic tone mapping, soft shadows. */
export function setupRenderer(renderer: THREE.WebGLRenderer): void {
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
}

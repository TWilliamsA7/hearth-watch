import * as THREE from "three";

import { fabric, mulberry32, rugTexture } from "./textures";
import type { WorldObject } from "./types";

/**
 * Furniture as recognisable low-poly models built from primitives: chairs with legs and a
 * back, couches with arms and cushions, a made bed, a table, shelves with books, a pot
 * plant, a rug. Each fits the object's box exactly (so it matches what the simulator
 * plans around) and faces the way `rotation` says (its front is local +z).
 *
 * Live rooms say which piece it is in `variant`; scripted rooms only have `kind`, so
 * those get the closest model.
 */

const WOODS = ["#8a5a3b", "#a0714f", "#6e4a33", "#b48a62"];
const FABRICS = ["#5b6f8a", "#8a5b5b", "#6b7d5a", "#9b8a6e", "#5e5a7a", "#7a6a5a", "#466b6b"];

interface Look {
  wood: THREE.MeshStandardMaterial;
  cloth: THREE.MeshStandardMaterial;
  accent: THREE.MeshStandardMaterial;
  rnd: () => number;
}

function look(o: WorldObject): Look {
  // Stable per object: seeded from where it stands.
  const seed = Math.round((o.center[0] * 1000 + 7) * 31 + o.center[2] * 1000 * 17);
  const rnd = mulberry32(seed);
  const pick = <T,>(xs: T[]) => xs[Math.floor(rnd() * xs.length)];
  const cloth = new THREE.MeshStandardMaterial({ color: pick(FABRICS), roughness: 0.92, map: fabric() });
  return {
    wood: new THREE.MeshStandardMaterial({ color: pick(WOODS), roughness: 0.6 }),
    cloth,
    accent: new THREE.MeshStandardMaterial({ color: pick(FABRICS), roughness: 0.85, map: fabric() }),
    rnd,
  };
}

function box(w: number, h: number, d: number, mat: THREE.Material, x = 0, y = 0, z = 0, radius = 0): THREE.Mesh {
  const geo = radius > 0 ? roundedBox(w, h, d, radius) : new THREE.BoxGeometry(w, h, d);
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.castShadow = m.receiveShadow = true;
  return m;
}

/** A box with softened edges (an extruded rounded rectangle), for cushions. */
function roundedBox(w: number, h: number, d: number, r: number): THREE.BufferGeometry {
  r = Math.min(r, w / 2 - 1e-3, h / 2 - 1e-3);
  const s = new THREE.Shape();
  s.moveTo(-w / 2 + r, -h / 2);
  s.lineTo(w / 2 - r, -h / 2);
  s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  s.lineTo(w / 2, h / 2 - r);
  s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  s.lineTo(-w / 2 + r, h / 2);
  s.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
  s.lineTo(-w / 2, -h / 2 + r);
  s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  const bevel = Math.min(r * 0.6, d / 4);
  const geo = new THREE.ExtrudeGeometry(s, { depth: d - 2 * bevel, bevelEnabled: true, bevelSize: bevel, bevelThickness: bevel, bevelSegments: 2, curveSegments: 4 });
  geo.translate(0, 0, -(d - 2 * bevel) / 2);
  return geo;
}

function legs(w: number, d: number, h: number, mat: THREE.Material, inset = 0.04, r = 0.022): THREE.Mesh[] {
  const geo = new THREE.CylinderGeometry(r, r * 0.8, h, 8);
  return [-1, 1].flatMap((sx) =>
    [-1, 1].map((sz) => {
      const m = new THREE.Mesh(geo, mat);
      m.position.set(sx * (w / 2 - inset), h / 2, sz * (d / 2 - inset));
      m.castShadow = true;
      return m;
    }),
  );
}

function chair(w: number, d: number, h: number, l: Look): THREE.Object3D[] {
  const seat = 0.45;
  return [
    ...legs(w, d, seat - 0.03, l.wood),
    box(w, 0.05, d, l.wood, 0, seat - 0.025, 0),
    box(w - 0.04, 0.04, d - 0.06, l.cloth, 0, seat + 0.015, 0.01, 0.015),
    box(w, h - seat - 0.03, 0.04, l.wood, 0, (h + seat) / 2, -d / 2 + 0.03),
  ];
}

function sofa(w: number, d: number, h: number, l: Look, seats: number): THREE.Object3D[] {
  const arm = Math.min(0.16, w * 0.12);
  const base = 0.24;
  const out: THREE.Object3D[] = [
    box(w, base, d, l.cloth, 0, base / 2 + 0.06, 0, 0.03),
    box(w, h - 0.06, 0.2, l.cloth, 0, (h + 0.06) / 2, -d / 2 + 0.1, 0.05),
    box(arm, 0.62 - 0.06, d, l.cloth, -w / 2 + arm / 2, 0.06 + (0.62 - 0.06) / 2, 0, 0.05),
    box(arm, 0.62 - 0.06, d, l.cloth, w / 2 - arm / 2, 0.06 + (0.62 - 0.06) / 2, 0, 0.05),
  ];
  const inner = w - 2 * arm;
  const cw = inner / seats;
  for (let i = 0; i < seats; i++) {
    const x = -inner / 2 + cw * (i + 0.5);
    out.push(box(cw - 0.02, 0.13, d - 0.24, l.cloth, x, base + 0.06 + 0.065, 0.1, 0.05));
    out.push(box(cw - 0.04, 0.38, 0.14, l.accent, x, base + 0.06 + 0.3, -d / 2 + 0.24, 0.06));
  }
  for (const m of legs(w, d, 0.06, l.wood, 0.06, 0.025)) out.push(m);
  return out;
}

function bed(w: number, d: number, _h: number, l: Look): THREE.Object3D[] {
  const white = new THREE.MeshStandardMaterial({ color: "#eceae4", roughness: 0.95, map: fabric() });
  return [
    box(w, 0.28, d, l.wood, 0, 0.14 + 0.04, 0),
    ...legs(w, d, 0.06, l.wood, 0.05),
    box(w - 0.06, 0.2, d - 0.06, white, 0, 0.42, 0, 0.04),
    box(w - 0.02, 0.06, d * 0.62, l.cloth, 0, 0.535, d * 0.19, 0.03),
    box(w * 0.72, 0.1, 0.32, white, 0, 0.56, -d / 2 + 0.26, 0.05),
    box(w + 0.02, 0.95, 0.06, l.wood, 0, 0.475, -d / 2 + 0.03),
  ];
}

function table(w: number, d: number, h: number, l: Look): THREE.Object3D[] {
  return [box(w, 0.04, d, l.wood, 0, h - 0.02, 0), ...legs(w, d, h - 0.04, l.wood, 0.06, 0.03)];
}

function cabinet(w: number, d: number, h: number, l: Look): THREE.Object3D[] {
  const dark = new THREE.MeshStandardMaterial({ color: "#16181c", roughness: 0.3, metalness: 0.2 });
  const screen = new THREE.MeshStandardMaterial({ color: "#20262f", roughness: 0.2, emissive: "#0d1520" });
  return [
    box(w, h - 0.06, d, l.wood, 0, (h - 0.06) / 2 + 0.06),
    ...legs(w, d, 0.06, dark, 0.05),
    box(w * 0.48, 0.006, 0.01, dark, -w * 0.25, h * 0.55, d / 2 + 0.002),
    box(w * 0.48, 0.006, 0.01, dark, w * 0.25, h * 0.55, d / 2 + 0.002),
    box(w * 0.72, 0.46, 0.04, screen, 0, h + 0.27, 0),
    box(0.25, 0.04, 0.12, dark, 0, h + 0.02, 0),
  ];
}

function bookshelf(w: number, d: number, h: number, l: Look): THREE.Object3D[] {
  const out: THREE.Object3D[] = [
    box(0.03, h, d, l.wood, -w / 2 + 0.015, h / 2, 0),
    box(0.03, h, d, l.wood, w / 2 - 0.015, h / 2, 0),
    box(w, h, 0.02, l.wood, 0, h / 2, -d / 2 + 0.01),
  ];
  const shelves = 5;
  const colors = ["#7b3b3b", "#3d4f66", "#c9a24a", "#4d6b4a", "#6b4f7a", "#d8d2c4", "#2e2e2e"];
  for (let i = 0; i <= shelves; i++) {
    const y = 0.02 + (i * (h - 0.04)) / shelves;
    out.push(box(w - 0.06, 0.025, d - 0.02, l.wood, 0, y, 0.01));
    if (i === shelves) break;
    let x = -w / 2 + 0.05;
    while (x < w / 2 - 0.08) {
      const bw = 0.025 + l.rnd() * 0.035;
      const bh = 0.18 + l.rnd() * 0.12;
      if (l.rnd() < 0.12) {
        x += bw * 2;
        continue;
      }
      const mat = new THREE.MeshStandardMaterial({ color: colors[Math.floor(l.rnd() * colors.length)], roughness: 0.8 });
      out.push(box(bw, bh, d * 0.7, mat, x + bw / 2, y + 0.0125 + bh / 2, 0.01));
      x += bw + 0.004;
    }
  }
  return out;
}

function plant(w: number, _d: number, h: number, l: Look): THREE.Object3D[] {
  const pot = new THREE.Mesh(
    new THREE.CylinderGeometry(w * 0.32, w * 0.24, 0.36, 18),
    new THREE.MeshStandardMaterial({ color: "#b5643c", roughness: 0.85 }),
  );
  pot.position.y = 0.18;
  pot.castShadow = pot.receiveShadow = true;
  const out: THREE.Object3D[] = [pot];
  const leaf = new THREE.MeshStandardMaterial({ color: "#3f7a46", roughness: 0.75, flatShading: true });
  for (let i = 0; i < 9; i++) {
    const r = 0.12 + l.rnd() * 0.12;
    const m = new THREE.Mesh(new THREE.IcosahedronGeometry(r, 0), leaf);
    const a = l.rnd() * Math.PI * 2;
    const rr = l.rnd() * w * 0.3;
    m.position.set(Math.cos(a) * rr, 0.45 + l.rnd() * (h - 0.6), Math.sin(a) * rr);
    m.rotation.set(l.rnd() * 3, l.rnd() * 3, l.rnd() * 3);
    m.castShadow = true;
    out.push(m);
  }
  return out;
}

function rug(w: number, d: number, l: Look): THREE.Object3D[] {
  const m = new THREE.Mesh(
    new THREE.PlaneGeometry(w, d),
    new THREE.MeshStandardMaterial({ map: rugTexture(Math.floor(l.rnd() * 3)), roughness: 1 }),
  );
  m.rotation.x = -Math.PI / 2;
  m.position.y = 0.006;
  m.receiveShadow = true;
  return [m];
}

function screen(w: number, _d: number, h: number, l: Look): THREE.Object3D[] {
  // A folding screen: hides people behind it from the camera, as scripted occluders do.
  const panels = 3;
  const pw = w / panels;
  return Array.from({ length: panels }, (_, i) => {
    const m = box(pw - 0.02, h - 0.05, 0.03, l.cloth, -w / 2 + pw * (i + 0.5), (h + 0.05) / 2, (i % 2 ? 0.04 : -0.04));
    m.rotation.y = i % 2 ? 0.25 : -0.25;
    return m;
  });
}

/** One piece of furniture, positioned and turned in the room. */
export function buildFurniture(o: WorldObject): THREE.Group {
  const g = new THREE.Group();
  const rot = o.rotation ?? 0;
  const side = Math.abs(Math.sin(rot)) > 0.5;
  // size is the world-axis box; the model is built in its own axes.
  const [sx, h, sz] = o.size;
  const w = side ? sz : sx;
  const d = side ? sx : sz;
  const l = look(o);
  const variant = o.variant ?? { bed: "bed", chair: "chair", couch: "couch", mat: "rug", occluder: "screen" }[o.kind];
  let parts: THREE.Object3D[];
  switch (variant) {
    case "chair":
      parts = chair(w, d, h, l);
      break;
    case "armchair":
      parts = sofa(w, d, h, l, 1);
      break;
    case "couch":
      parts = sofa(w, d, h, l, w > 1.8 ? 3 : 2);
      break;
    case "loveseat":
      parts = sofa(w, d, h, l, 2);
      break;
    case "bed":
      parts = bed(w, d, h, l);
      break;
    case "table":
    case "side_table":
      parts = table(w, d, h, l);
      break;
    case "cabinet":
      parts = cabinet(w, d, h, l);
      break;
    case "bookshelf":
      parts = bookshelf(w, d, h, l);
      break;
    case "plant":
      parts = plant(w, d, h, l);
      break;
    case "rug":
      parts = rug(w, d, l);
      break;
    default:
      parts = screen(w, d, h, l);
  }
  g.add(...parts);
  // The object's box is centred at center (y = half its height); the model stands on the floor.
  g.position.set(o.center[0], o.center[1] - h / 2, o.center[2]);
  g.rotation.y = rot;
  g.userData.variant = variant;
  return g;
}

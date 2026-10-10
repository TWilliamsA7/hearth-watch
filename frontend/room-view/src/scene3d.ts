import * as THREE from "three";

import type { Theme } from "./draw";
import { BONES, HEAD_POINTS } from "./skeleton";
import type { WorldPerson, WorldRoom } from "./types";

/**
 * How one person is drawn in the 3D view. The view creates one per person_id, calls
 * update() every frame with that person's joints (metres, three.js axes) and heading,
 * and disposes it when they leave. The stick figure below is the default; a rigged
 * model (GLTF + SkinnedMesh driven by the same joints and heading) can replace it by
 * implementing this interface, without changing the view.
 */
export interface PersonRig {
  readonly object: THREE.Object3D;
  update(person: WorldPerson, state: RigState): void;
  dispose(): void;
}

export interface RigState {
  /** The dashboard has an open alert for this person. */
  alert: boolean;
  theme: Theme;
}

export type RigFactory = (personId: string | null) => PersonRig;

const BONE_RADIUS = 0.025;
const JOINT_RADIUS = 0.035;
const HEAD_RADIUS = 0.1;
const UP = new THREE.Vector3(0, 1, 0);

/** The default rig: bones as cylinders, joints as spheres, a head and an anonymous tag (P1). */
export const stickFigureRig: RigFactory = (personId) => {
  const group = new THREE.Group();
  const boneGeo = new THREE.CylinderGeometry(1, 1, 1, 10);
  const jointGeo = new THREE.SphereGeometry(JOINT_RADIUS, 12, 8);
  const headGeo = new THREE.SphereGeometry(HEAD_RADIUS, 20, 14);
  const boneMat = new THREE.MeshStandardMaterial({ roughness: 0.6 });
  const jointMat = new THREE.MeshStandardMaterial({ roughness: 0.4 });

  const bones = BONES.map(() => new THREE.Mesh(boneGeo, boneMat));
  const joints = new Map<string, THREE.Mesh>();
  const head = new THREE.Mesh(headGeo, boneMat);
  group.add(...bones, head);

  const tag = personId ? makeTag(personId.toUpperCase()) : null;
  if (tag) group.add(tag.sprite);

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  let color = "";

  return {
    object: group,
    update(person, { alert, theme }) {
      const next = alert ? theme.alert : theme.bone;
      if (next !== color) {
        color = next;
        boneMat.color.set(next);
        jointMat.color.set(alert ? theme.alert : theme.joint);
        tag?.draw(alert ? theme.alert : theme.muted);
      }
      BONES.forEach(([from, to], i) => {
        const p = person.joints[from];
        const q = person.joints[to];
        const bone = bones[i];
        bone.visible = !!p && !!q;
        if (!p || !q) return;
        a.set(...p);
        b.set(...q);
        const len = a.distanceTo(b);
        bone.position.copy(a).add(b).multiplyScalar(0.5);
        bone.scale.set(BONE_RADIUS, len, BONE_RADIUS);
        bone.quaternion.setFromUnitVectors(UP, b.sub(a).normalize());
      });
      for (const [name, p] of Object.entries(person.joints)) {
        let mesh = joints.get(name);
        if (!mesh) {
          mesh = new THREE.Mesh(jointGeo, jointMat);
          joints.set(name, mesh);
          group.add(mesh);
        }
        mesh.position.set(...p);
      }
      const headPts = HEAD_POINTS.map((n) => person.joints[n]).filter(Boolean);
      head.visible = headPts.length > 0;
      if (head.visible) {
        head.position.set(0, 0, 0);
        for (const p of headPts) head.position.add(a.set(...p));
        head.position.divideScalar(headPts.length);
        tag?.sprite.position.copy(head.position).add(a.set(0, 0.3, 0));
      }
    },
    dispose() {
      boneGeo.dispose();
      jointGeo.dispose();
      headGeo.dispose();
      boneMat.dispose();
      jointMat.dispose();
      tag?.dispose();
    },
  };
};

/** Anonymous track label (P1) as a sprite that always faces the viewer. */
export function makeTag(text: string) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 64;
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.4, 0.2, 1);
  sprite.renderOrder = 10; // stays readable through furniture
  return {
    sprite,
    draw(color: string) {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = color;
      ctx.font = "600 44px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, canvas.width / 2, canvas.height / 2);
      texture.needsUpdate = true;
    },
    dispose() {
      texture.dispose();
      material.dispose();
    },
  };
}

const FRONT_MARGIN = 0.6; // floor in front of depth 0, so people at the front stand on something
const BACK_MARGIN = 0.4; // people at depth 1 stand just inside the back wall

/**
 * The room: floor, walls, furniture and the simulated camera with its field of view.
 * Walls are drawn from the inside only, so from any orbit angle the near walls drop away
 * like a dollhouse.
 */
export function buildRoom(room: WorldRoom, theme: Theme): THREE.Group {
  const group = new THREE.Group();
  const width = room.width_back;
  const depth = room.depth + FRONT_MARGIN + BACK_MARGIN;
  const wall = new THREE.MeshStandardMaterial({ color: theme.background, side: THREE.BackSide, roughness: 0.9 });
  const floor = new THREE.MeshStandardMaterial({ color: theme.floor, side: THREE.BackSide, roughness: 1 });
  // BoxGeometry faces: +x, -x, +y (ceiling), -y (floor), +z, -z.
  const shell = new THREE.Mesh(new THREE.BoxGeometry(width, room.height, depth), [wall, wall, wall, floor, wall, wall]);
  shell.position.set(0, room.height / 2, FRONT_MARGIN - depth / 2);
  group.add(shell);

  const grid = new THREE.GridHelper(Math.max(width, depth), Math.round(Math.max(width, depth) * 2), theme.muted, theme.muted);
  (grid.material as THREE.Material).opacity = 0.15;
  (grid.material as THREE.Material).transparent = true;
  grid.position.set(0, 0.002, FRONT_MARGIN - depth / 2);
  group.add(grid);

  for (const o of room.objects) {
    const occluder = o.kind === "occluder";
    const mat = new THREE.MeshStandardMaterial({
      color: theme.furniture[o.kind],
      roughness: 0.8,
      transparent: occluder, // see who it hides from the camera
      opacity: occluder ? 0.55 : 1,
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...o.size), mat);
    mesh.position.set(...o.center);
    group.add(mesh);
  }

  group.add(cameraMarker(room, theme));
  return group;
}

/** The simulated camera and the edges of what it sees, out to the back wall. */
function cameraMarker(room: WorldRoom, theme: Theme): THREE.Group {
  const group = new THREE.Group();
  const [cx, cy, cz] = room.camera_position;
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.18, 0.12, 0.14),
    new THREE.MeshStandardMaterial({ color: theme.badge.simulation }),
  );
  body.position.set(cx, cy, cz);
  group.add(body);

  // Corners of the frame where it meets depth 0, extended to the back wall.
  const halfH = Math.tan(THREE.MathUtils.degToRad(room.camera_fov / 2)) * cz;
  const halfW = room.width_front / 2;
  const reach = (cz + room.depth) / cz;
  const points: THREE.Vector3[] = [];
  const corners = [[-halfW, -halfH], [halfW, -halfH], [halfW, halfH], [-halfW, halfH]];
  const far = corners.map(([x, y]) => new THREE.Vector3(cx + x * reach, cy + y * reach, -room.depth));
  for (const f of far) points.push(new THREE.Vector3(cx, cy, cz), f);
  far.forEach((f, i) => points.push(f, far[(i + 1) % far.length]));
  const lines = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: theme.badge.simulation, transparent: true, opacity: 0.45 }),
  );
  group.add(lines);
  return group;
}

/** Free every geometry and material under an object (the room is rebuilt per run). */
export function disposeTree(object: THREE.Object3D): void {
  object.traverse((o) => {
    const mesh = o as THREE.Mesh;
    mesh.geometry?.dispose();
    const mats = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
    for (const m of mats) m.dispose();
  });
}

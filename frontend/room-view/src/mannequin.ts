import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { clone as cloneSkinned } from "three/addons/utils/SkeletonUtils.js";

import { makeTag, stickFigureRig, type PersonRig, type RigFactory, type RigState } from "./scene3d";
import { HEAD_POINTS } from "./skeleton";
import type { WorldPerson } from "./types";
import type { Vec3 } from "./world";

/** Quaternius "Universal Animation Library" mannequin, animations stripped (CC0; see assets/models). */
export const DEFAULT_MANNEQUIN_URL = new URL("../assets/models/resident.glb", import.meta.url).href;

type Point = (j: WorldPerson["joints"]) => Vec3 | undefined;
const joint = (name: string): Point => (j) => j[name];
const mean = (pts: (Vec3 | undefined)[]): Vec3 | undefined => {
  const ok = pts.filter((p): p is Vec3 => !!p);
  if (!ok.length) return undefined;
  return [0, 1, 2].map((i) => ok.reduce((s, p) => s + p[i], 0) / ok.length) as Vec3;
};
const head: Point = (j) => mean(HEAD_POINTS.map((n) => j[n]));
const hipMid: Point = (j) => mean([j.left_hip, j.right_hip]);

/**
 * Each bone is turned to point from one keypoint to the next, parents before children.
 * Bones not listed (spine, clavicles, hands, feet, fingers) keep their rest pose relative
 * to their parent; COCO keypoints say nothing about them.
 */
const AIMS: ReadonlyArray<{ bone: string; child: string; from: Point; to: Point }> = [
  { bone: "neck_01", child: "Head", from: joint("neck"), to: head },
  { bone: "upperarm_l", child: "lowerarm_l", from: joint("left_shoulder"), to: joint("left_elbow") },
  { bone: "lowerarm_l", child: "hand_l", from: joint("left_elbow"), to: joint("left_wrist") },
  { bone: "upperarm_r", child: "lowerarm_r", from: joint("right_shoulder"), to: joint("right_elbow") },
  { bone: "lowerarm_r", child: "hand_r", from: joint("right_elbow"), to: joint("right_wrist") },
  { bone: "thigh_l", child: "calf_l", from: joint("left_hip"), to: joint("left_knee") },
  { bone: "calf_l", child: "foot_l", from: joint("left_knee"), to: joint("left_ankle") },
  { bone: "thigh_r", child: "calf_r", from: joint("right_hip"), to: joint("right_knee") },
  { bone: "calf_r", child: "foot_r", from: joint("right_knee"), to: joint("right_ankle") },
];

const _qp = new THREE.Quaternion();
const _qr = new THREE.Quaternion();
const _v = new THREE.Vector3();

/**
 * Turn `bone` (from its rest rotation `rest`) by the smallest rotation that makes the
 * direction to its child, `restDir` in the bone's own frame, point along `target` in
 * world space. The parent's world rotation must already be final.
 */
export function aimBone(bone: THREE.Object3D, rest: THREE.Quaternion, restDir: THREE.Vector3, target: THREE.Vector3): void {
  bone.parent!.getWorldQuaternion(_qp);
  const current = _v.copy(restDir).applyQuaternion(rest).applyQuaternion(_qp);
  _qr.setFromUnitVectors(current, target);
  // local = parent⁻¹ · R · parent · rest
  bone.quaternion.copy(_qp).invert().multiply(_qr).multiply(_qp).multiply(rest);
}

/**
 * Rotation from a body's own axes to the world: up from the hips to the neck, left from
 * the right hip to the left. Returns null if the frame is degenerate.
 */
export function bodyBasis(hipL: THREE.Vector3, hipR: THREE.Vector3, neck: THREE.Vector3): THREE.Matrix4 | null {
  const mid = hipL.clone().add(hipR).multiplyScalar(0.5);
  const up = neck.clone().sub(mid);
  const left = hipL.clone().sub(hipR);
  if (up.lengthSq() < 1e-8 || left.lengthSq() < 1e-8) return null;
  up.normalize();
  left.addScaledVector(up, -left.dot(up));
  if (left.lengthSq() < 1e-8) return null;
  left.normalize();
  const forward = new THREE.Vector3().crossVectors(left, up); // +z when facing the camera
  return new THREE.Matrix4().makeBasis(left, up, forward);
}

/** Measurements of the model at rest, shared by every person drawn with it. */
export interface Template {
  scene: THREE.Object3D;
  rest: Map<string, { quaternion: THREE.Quaternion; dir: THREE.Vector3 }>;
  /** World axes of the rest body, inverted: undoes the model's own orientation. */
  restBasisInv: THREE.Matrix4;
  legLength: number;
}

const templates = new Map<string, Promise<Template>>();

function loadTemplate(url: string): Promise<Template> {
  let t = templates.get(url);
  if (!t) {
    t = new GLTFLoader().loadAsync(url).then(({ scene }) => mannequinTemplate(scene));
    templates.set(url, t);
  }
  return t;
}

/** Measure a loaded mannequin's rest pose. Throws if it lacks the bones the rig drives. */
export function mannequinTemplate(scene: THREE.Object3D): Template {
  scene.updateMatrixWorld(true);
  const pos = (name: string) => scene.getObjectByName(name)!.getWorldPosition(new THREE.Vector3());
  const rest = new Map<string, { quaternion: THREE.Quaternion; dir: THREE.Vector3 }>();
  for (const { bone, child } of AIMS) {
    const b = scene.getObjectByName(bone);
    const c = scene.getObjectByName(child);
    if (!b || !c) throw new Error(`mannequin is missing bone ${!b ? bone : child}`);
    rest.set(bone, { quaternion: b.quaternion.clone(), dir: c.position.clone().normalize() });
  }
  const basis = bodyBasis(pos("thigh_l"), pos("thigh_r"), pos("neck_01"));
  if (!basis) throw new Error("mannequin rest pose has no usable hips");
  const leg = (s: string) => pos(`thigh_${s}`).distanceTo(pos(`calf_${s}`)) + pos(`calf_${s}`).distanceTo(pos(`foot_${s}`));
  return { scene, rest, restBasisInv: basis.invert(), legLength: (leg("l") + leg("r")) / 2 };
}

const v3 = (p: Vec3) => new THREE.Vector3(p[0], p[1], p[2]);

/** One person's copy of the model, posed from their keypoints every frame. */
export function mannequinBody(t: Template, personId: string | null): PersonRig {
  const model = cloneSkinned(t.scene);
  const bones = new Map(AIMS.map(({ bone }) => [bone, model.getObjectByName(bone)!]));
  const headBone = model.getObjectByName("Head");
  const thighL = model.getObjectByName("thigh_l")!;
  const thighR = model.getObjectByName("thigh_r")!;
  const materials: { mat: THREE.MeshStandardMaterial; role: "main" | "joints" }[] = [];
  model.traverse((o) => {
    const mesh = o as THREE.SkinnedMesh;
    if (!mesh.isMesh) return;
    mesh.frustumCulled = false; // bounds don't follow posed bones
    const own = (mesh.material as THREE.MeshStandardMaterial).clone();
    mesh.material = own;
    materials.push({ mat: own, role: /joint/i.test(own.name) ? "joints" : "main" });
  });
  const group = new THREE.Group();
  group.add(model);
  const tag = personId ? makeTag(personId.toUpperCase()) : null;
  if (tag) group.add(tag.sprite);

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const dir = new THREE.Vector3();
  const rot = new THREE.Matrix4();
  let color = "";

  return {
    object: group,
    update(person: WorldPerson, { alert, theme }: RigState) {
      const next = alert ? theme.alert : theme.bone;
      if (next !== color) {
        color = next;
        for (const { mat, role } of materials) mat.color.set(role === "joints" && !alert ? theme.joint : next);
        tag?.draw(alert ? theme.alert : theme.muted);
      }
      const j = person.joints;
      const mid = hipMid(j);
      if (!j.left_hip || !j.right_hip || !j.neck || !mid) {
        model.visible = false;
        return;
      }
      model.visible = true;

      // Whole body: orientation from the hips and neck, size from the legs, hips on the hips.
      const basis = bodyBasis(v3(j.left_hip), v3(j.right_hip), v3(j.neck));
      if (basis) model.quaternion.setFromRotationMatrix(rot.multiplyMatrices(basis, t.restBasisInv));
      const leg = (s: "left" | "right") => {
        const [h, k, an] = [j[`${s}_hip`], j[`${s}_knee`], j[`${s}_ankle`]];
        return h && k && an ? v3(h).distanceTo(v3(k)) + v3(k).distanceTo(v3(an)) : null;
      };
      const legs = [leg("left"), leg("right")].filter((l): l is number => l !== null);
      if (legs.length) model.scale.setScalar(legs.reduce((s, l) => s + l, 0) / legs.length / t.legLength);
      for (const [name, bone] of bones) bone.quaternion.copy(t.rest.get(name)!.quaternion);
      model.position.set(0, 0, 0);
      model.updateMatrixWorld(true);
      const hips = thighL.getWorldPosition(a).add(thighR.getWorldPosition(b)).multiplyScalar(0.5);
      model.position.copy(v3(mid).sub(hips));

      // Limbs and neck: point each bone at the next keypoint.
      for (const { bone, from, to } of AIMS) {
        const p = from(j);
        const q = to(j);
        if (!p || !q) continue;
        dir.set(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
        if (dir.lengthSq() < 1e-10) continue;
        const r = t.rest.get(bone)!;
        aimBone(bones.get(bone)!, r.quaternion, r.dir, dir.normalize());
      }
      model.updateMatrixWorld(true);
      if (tag && headBone) tag.sprite.position.copy(headBone.getWorldPosition(a)).add(b.set(0, 0.35, 0));
    },
    dispose() {
      for (const { mat } of materials) mat.dispose(); // geometry is shared with the template
      tag?.dispose();
    },
  };
}

/**
 * A rig factory that draws people as a rigged mannequin. The model loads once per URL;
 * until it arrives, or if it can't be loaded, people are drawn as stick figures.
 * Create it once (e.g. at module scope), not on every render.
 */
export function createMannequinRig(url: string = DEFAULT_MANNEQUIN_URL): RigFactory {
  return (personId) => {
    const group = new THREE.Group();
    let fallback: PersonRig | null = stickFigureRig(personId);
    let body: PersonRig | null = null;
    let disposed = false;
    group.add(fallback.object);
    loadTemplate(url).then(
      (t) => {
        if (disposed) return;
        body = mannequinBody(t, personId);
        group.remove(fallback!.object);
        fallback!.dispose();
        fallback = null;
        group.add(body.object);
      },
      (err) => console.warn(`mannequin ${url} failed to load; keeping stick figures`, err),
    );
    return {
      object: group,
      update: (person, state) => (body ?? fallback)!.update(person, state),
      dispose() {
        disposed = true;
        fallback?.dispose();
        body?.dispose();
      },
    };
  };
}

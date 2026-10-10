import * as THREE from "three";
import { describe, expect, it } from "vitest";

import { aimBone, bodyBasis } from "./mannequin";

/** A two-bone arm hanging straight down from a parent that is itself turned. */
function arm() {
  const root = new THREE.Bone();
  root.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 3);
  const upper = new THREE.Bone();
  const lower = new THREE.Bone();
  lower.position.set(0, -0.3, 0);
  root.add(upper);
  upper.add(lower);
  root.updateMatrixWorld(true);
  return { root, upper, lower };
}

const worldDir = (from: THREE.Object3D, to: THREE.Object3D) => {
  from.updateMatrixWorld(true);
  const a = from.getWorldPosition(new THREE.Vector3());
  return to.getWorldPosition(new THREE.Vector3()).sub(a).normalize();
};

describe("aimBone", () => {
  it("points the bone at the target whatever its parent's rotation", () => {
    const { root, upper, lower } = arm();
    const rest = upper.quaternion.clone();
    const restDir = lower.position.clone().normalize();
    for (const target of [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(1, 1, -1).normalize()]) {
      aimBone(upper, rest, restDir, target);
      root.updateMatrixWorld(true);
      expect(worldDir(upper, lower).distanceTo(target)).toBeLessThan(1e-6);
    }
  });

  it("starts from the rest rotation each time, so repeated calls don't drift", () => {
    const { root, upper, lower } = arm();
    const rest = upper.quaternion.clone();
    const restDir = lower.position.clone().normalize();
    const target = new THREE.Vector3(0, 0, -1);
    aimBone(upper, rest, restDir, target);
    const once = upper.quaternion.clone();
    aimBone(upper, rest, restDir, target);
    root.updateMatrixWorld(true);
    expect(upper.quaternion.angleTo(once)).toBeLessThan(1e-6); // acos limits precision
  });
});

describe("bodyBasis", () => {
  const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);

  it("is the identity for a body facing the camera upright", () => {
    const m = bodyBasis(v(0.1, 0.9, 0), v(-0.1, 0.9, 0), v(0, 1.4, 0))!;
    expect(m.equals(new THREE.Matrix4())).toBe(true);
  });

  it("faces +x for a body turned to image right", () => {
    const m = bodyBasis(v(0, 0.9, -0.1), v(0, 0.9, 0.1), v(0, 1.4, 0))!;
    const forward = new THREE.Vector3(0, 0, 1).applyMatrix4(m);
    expect(forward.distanceTo(v(1, 0, 0))).toBeLessThan(1e-9);
  });

  it("lies the body down when the neck is level with the hips", () => {
    const m = bodyBasis(v(0, 0.1, -0.1), v(0, 0.1, 0.1), v(0.6, 0.1, 0))!;
    expect(new THREE.Vector3(0, 1, 0).applyMatrix4(m).distanceTo(v(1, 0, 0))).toBeLessThan(1e-9);
  });

  it("rejects a degenerate frame", () => {
    expect(bodyBasis(v(0, 1, 0), v(0, 1, 0), v(0, 1.5, 0))).toBeNull();
  });
});

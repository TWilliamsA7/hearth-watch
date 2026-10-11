// Poses the real mannequin (assets/models/resident.glb) from recorded world-track poses and
// checks that its limbs land on the keypoints. Fixtures: src/fixtures/*.json.
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { beforeAll, describe, expect, it } from "vitest";

import { DEFAULT_THEME } from "./draw";
import walking from "./fixtures/walking-p2.json";
import poses from "./fixtures/world-poses.json";
import { mannequinBody, mannequinTemplate, type Template } from "./mannequin";
import type { WorldPerson } from "./types";

const ENDS: [bone: string, keypoint: string][] = [
  ["upperarm_l", "left_shoulder"], ["lowerarm_l", "left_elbow"], ["hand_l", "left_wrist"],
  ["upperarm_r", "right_shoulder"], ["lowerarm_r", "right_elbow"], ["hand_r", "right_wrist"],
  ["calf_l", "left_knee"], ["foot_l", "left_ankle"], ["calf_r", "right_knee"], ["foot_r", "right_ankle"],
];

let template: Template;
beforeAll(async () => {
  const buf = readFileSync(new URL("../assets/models/resident.glb", import.meta.url));
  const gltf = await new GLTFLoader().parseAsync(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), "");
  template = mannequinTemplate(gltf.scene);
});

describe("mannequin rig", () => {
  for (const { source, ...person } of poses as unknown as (WorldPerson & { source: string })[]) {
    it(`puts limbs on the keypoints: ${person.person_id ?? "single"} in ${source}`, () => {
      const body = mannequinBody(template, null); // no tag: it needs a DOM canvas
      body.update(person, { alert: false, theme: DEFAULT_THEME });
      for (const [bone, kp] of ENDS) {
        const at = body.object.getObjectByName(bone)!.getWorldPosition(new THREE.Vector3());
        expect(at.distanceTo(new THREE.Vector3(...person.joints[kp])), `${bone} vs ${kp}`).toBeLessThan(0.15);
      }
      body.dispose();
    });
  }

  it("stands on its feet: the feet land on the ankles, not above them", () => {
    // Straight legs on the floor. (Older recordings' sitting pose has a stretched shin, which
    // no rig of fixed bone lengths can follow.)
    const legs = [["foot_l", "left_ankle", "left_hip"], ["foot_r", "right_ankle", "right_hip"]] as const;
    let checked = 0;
    for (const person of poses as unknown as WorldPerson[]) {
      const body = mannequinBody(template, null);
      body.update(person, { alert: false, theme: DEFAULT_THEME });
      for (const [bone, kp, hip] of legs) {
        const ankle = new THREE.Vector3(...person.joints[kp]);
        if (ankle.y > 0.1 || ankle.distanceTo(new THREE.Vector3(...person.joints[hip])) < 0.47 * person.height) continue;
        const at = body.object.getObjectByName(bone)!.getWorldPosition(new THREE.Vector3());
        expect(Math.abs(at.y - ankle.y), `${bone} height vs ${kp}`).toBeLessThan(0.02);
        checked++;
      }
      body.dispose();
    }
    expect(checked).toBeGreaterThan(2);
  });

  it("keeps the same size through a walking cycle, though the gait stretches the legs", () => {
    const body = mannequinBody(template, null);
    const scales = (walking as unknown as WorldPerson[]).map((person) => {
      body.update(person, { alert: false, theme: DEFAULT_THEME });
      return body.object.children[0].scale.x;
    });
    expect(new Set(scales).size).toBe(1);
    expect(scales[0]).toBeCloseTo((walking[0] as { height: number }).height / template.height, 9);
    body.dispose();
  });
});

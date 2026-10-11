import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { DEFAULT_THEME, type Theme } from "./draw";
import type { LivePerson } from "./generated/sim";
import type { LiveClient } from "./live";
import { createMannequinRig } from "./mannequin";
import { buildRoom3D, roomBounds, setupRenderer, type Lighting, type Room3D } from "./room3d";
import type { PersonRig, RigFactory } from "./scene3d";

export interface Overlays {
  ids: boolean; // name tags
  activity: boolean; // what each person is doing, in their tag
  paths: boolean;
  destinations: boolean;
  space: boolean; // personal space rings
  conversations: boolean;
  fov: boolean; // the simulated camera's field of view
  visibility: boolean; // fade people the camera can't see
  incidents: boolean;
}

export const DEFAULT_OVERLAYS: Overlays = {
  ids: true, activity: true, paths: true, destinations: true, space: false, conversations: true, fov: true,
  visibility: false, incidents: true,
};

export type CameraMode = "orbit" | "top" | "camera" | "follow";

export interface LiveView3DProps {
  client: LiveClient;
  overlays?: Overlays;
  cameraMode?: CameraMode;
  selected?: string | null;
  onSelect?: (personId: string | null) => void;
  /** person_ids with an open alert: drawn red. */
  alertPersonIds?: readonly string[];
  /** Hide ground-truth activity labels (for blind demos). */
  hideTruth?: boolean;
  lighting?: Lighting;
  rigFactory?: RigFactory;
  theme?: Theme;
  style?: React.CSSProperties;
}

const defaultRig = createMannequinRig();

/** A real person from a live camera (LivePerson.kind "live"): one look, so they stand out. */
export const LIVE_APPEARANCE = { skin: "#c9a48a", top: "#e5484d", bottom: "#2f3542", hair: "#5c4a3c", build: 1 };

/**
 * The live simulated room in 3D, for the Sim Studio: everyone's true body, what they're
 * doing and where they're going, with toggleable overlays. Click a person to select them.
 * Simulator-only; this is not what the camera reports.
 */
export function LiveView3D(props: LiveView3DProps) {
  const ref = useRef<HTMLDivElement>(null);
  const live = useRef(props);
  live.current = props;

  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    setupRenderer(renderer);
    renderer.domElement.style.display = "block";
    host.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 80);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.minDistance = 1.5;
    controls.maxDistance = 25;

    let room: Room3D | null = null;
    let roomRev = -1;
    let roomLighting: Lighting | null = null;
    const rigs = new Map<string, PersonRig>();
    let rigsFrom: RigFactory | null = null;
    const overlay = new THREE.Group();
    scene.add(overlay);
    let lastMode: CameraMode | null = null;

    // Overlay pieces, made on demand per person and reused.
    const paths = new Map<string, THREE.Line>();
    const dests = new Map<string, THREE.Mesh>();
    const rings = new Map<string, THREE.Mesh>();
    const markers = new Map<string, THREE.Mesh>();
    const talk = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: "#ffd166", transparent: true, opacity: 0.85 }));
    overlay.add(talk);
    const pathMat = new THREE.LineDashedMaterial({ color: "#7cc4ff", dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.9 });
    const destGeo = new THREE.RingGeometry(0.1, 0.15, 28);
    const destMat = new THREE.MeshBasicMaterial({ color: "#7cc4ff", side: THREE.DoubleSide, transparent: true, opacity: 0.85 });
    const spaceGeo = new THREE.RingGeometry(0.36, 0.375, 40);
    const spaceMat = new THREE.MeshBasicMaterial({ color: "#ffffff", side: THREE.DoubleSide, transparent: true, opacity: 0.35 });
    const markGeo = new THREE.ConeGeometry(0.09, 0.22, 16);
    const markMat = new THREE.MeshBasicMaterial({ color: "#ff5d5d" });

    const take = <T extends THREE.Object3D>(map: Map<string, T>, id: string, make: () => T): T => {
      let o = map.get(id);
      if (!o) {
        o = make();
        map.set(id, o);
        overlay.add(o);
      }
      return o;
    };
    const hideRest = (map: Map<string, THREE.Object3D>, keep: Set<string>) => {
      for (const [id, o] of map) if (!keep.has(id)) o.visible = false;
    };

    const fit = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      if (!w || !h) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (renderer.getPixelRatio() !== dpr) renderer.setPixelRatio(dpr);
      const size = renderer.getSize(new THREE.Vector2());
      if (size.x !== w || size.y !== h) {
        renderer.setSize(w, h, true);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
    };

    const frameRoom = () => {
      const world = live.current.client.world;
      if (!world) return;
      const [x0, z0, x1, z1] = roomBounds(world.room);
      const cx = (x0 + x1) / 2;
      const cz = (z0 + z1) / 2;
      const span = Math.max(x1 - x0, z1 - z0);
      controls.target.set(cx, 0.6, cz);
      camera.position.set(cx - span * 0.35, span * 0.95 + 1.5, cz + span * 1.05 + 1);
      controls.update();
    };

    // Picking: the person whose body is nearest the click on screen.
    let people: LivePerson[] = [];
    const v = new THREE.Vector3();
    const onClick = (ev: MouseEvent) => {
      if (dragged) return;
      const r = renderer.domElement.getBoundingClientRect();
      const mx = ev.clientX - r.left;
      const my = ev.clientY - r.top;
      let best: string | null = null;
      let bestD = 46;
      for (const p of people) {
        for (const joint of ["neck", "left_hip", "nose", "left_knee"] as const) {
          const q = p.joints[joint];
          if (!q || !p.person_id) continue;
          v.set(q[0], q[1], q[2]).project(camera);
          const d = Math.hypot((v.x * 0.5 + 0.5) * r.width - mx, (-v.y * 0.5 + 0.5) * r.height - my);
          if (d < bestD) {
            bestD = d;
            best = p.person_id;
          }
        }
      }
      live.current.onSelect?.(best);
    };
    let dragged = false;
    let down: [number, number] | null = null;
    const onDown = (ev: PointerEvent) => {
      down = [ev.clientX, ev.clientY];
      dragged = false;
    };
    const onMove = (ev: PointerEvent) => {
      if (down && Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 5) dragged = true;
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("click", onClick);

    let raf = 0;
    const render = () => {
      raf = requestAnimationFrame(render);
      fit();
      const p = live.current;
      const theme = p.theme ?? DEFAULT_THEME;
      const ov = p.overlays ?? DEFAULT_OVERLAYS;
      const world = p.client.world;
      scene.background = new THREE.Color(theme.background);
      const lighting = p.lighting ?? ((world?.room.lighting as Lighting) || "day");

      if (world && (p.client.revision !== roomRev || lighting !== roomLighting)) {
        const fresh = p.client.revision !== roomRev;
        if (room) {
          scene.remove(room.group);
          room.dispose();
        }
        room = buildRoom3D(world.room, theme, lighting);
        scene.add(room.group);
        if (fresh) {
          for (const rig of rigs.values()) {
            scene.remove(rig.object);
            rig.dispose();
          }
          rigs.clear();
          frameRoom();
        }
        roomRev = p.client.revision;
        roomLighting = lighting;
      }
      const factory = p.rigFactory ?? defaultRig;
      if (factory !== rigsFrom) {
        for (const rig of rigs.values()) {
          scene.remove(rig.object);
          rig.dispose();
        }
        rigs.clear();
        rigsFrom = factory;
      }

      const sample = p.client.sample(p.client.renderTime());
      people = sample.people;
      const looks = new Map((world?.actors ?? []).map((a) => [a.person_id, a.appearance]));
      const alerts = new Set(p.alertPersonIds ?? []);
      const seen = new Set<string>();
      for (const person of people) {
        const id = person.person_id ?? "";
        seen.add(id);
        let rig = rigs.get(id);
        if (!rig) {
          rig = factory(person.person_id);
          rigs.set(id, rig);
          scene.add(rig.object);
        }
        const live = person.kind === "live";
        const tag = !ov.ids ? null
          : live ? `LIVE · ${person.label}`
          : ov.activity && !p.hideTruth ? `${id.toUpperCase()} · ${person.incident ? person.incident.replace("_", " ") : person.activity}`
          : id.toUpperCase();
        rig.update(person, {
          alert: alerts.has(id) || (ov.incidents && !p.hideTruth && !!person.incident && person.posture === "floor"),
          theme,
          appearance: looks.get(id) ?? (live ? LIVE_APPEARANCE : undefined),
          selected: p.selected === id,
          dimmed: ov.visibility && person.visible < 0.25,
          label: tag,
        });
      }
      for (const [id, rig] of rigs) {
        if (seen.has(id)) continue;
        scene.remove(rig.object);
        rig.dispose();
        rigs.delete(id);
      }

      // Overlays.
      const keep = new Set<string>();
      for (const person of people) {
        const id = person.person_id ?? "";
        keep.add(id);
        const hip = person.joints.left_hip;
        const root: [number, number] = hip ? [hip[0], hip[2]] : [0, 0];
        const line = take(paths, id, () => {
          const l = new THREE.Line(new THREE.BufferGeometry(), pathMat);
          l.frustumCulled = false;
          return l;
        });
        const showPath = ov.paths && !p.hideTruth && person.path.length > 1;
        line.visible = showPath;
        if (showPath) {
          const pts = [new THREE.Vector3(root[0], 0.015, root[1]), ...person.path.slice(1).map(([x, z]) => new THREE.Vector3(x, 0.015, z))];
          line.geometry.setFromPoints(pts);
          line.computeLineDistances();
        }
        const dest = take(dests, id, () => {
          const m = new THREE.Mesh(destGeo, destMat);
          m.rotation.x = -Math.PI / 2;
          return m;
        });
        dest.visible = ov.destinations && !p.hideTruth && !!person.destination;
        if (person.destination) dest.position.set(person.destination[0], 0.016, person.destination[1]);
        const ring = take(rings, id, () => {
          const m = new THREE.Mesh(spaceGeo, spaceMat);
          m.rotation.x = -Math.PI / 2;
          return m;
        });
        ring.visible = ov.space && person.posture === "standing";
        ring.position.set(root[0], 0.014, root[1]);
        const mark = take(markers, id, () => {
          const m = new THREE.Mesh(markGeo, markMat);
          m.rotation.x = Math.PI;
          return m;
        });
        mark.visible = ov.incidents && !p.hideTruth && !!person.incident;
        if (mark.visible) {
          const top = person.joints.nose ?? person.joints.neck;
          const bob = Math.sin(performance.now() / 250) * 0.04;
          if (top) mark.position.set(top[0], Math.max(top[1], 0.3) + 0.55 + bob, top[2]);
        }
      }
      hideRest(paths, keep);
      hideRest(dests, keep);
      hideRest(rings, keep);
      hideRest(markers, keep);
      const convPts: THREE.Vector3[] = [];
      if (ov.conversations && !p.hideTruth && sample.frame) {
        const at = new Map(people.map((q) => [q.person_id, q]));
        for (const c of sample.frame.conversations) {
          const heads = c.members.map((m) => at.get(m)?.joints.neck).filter((x): x is [number, number, number] => !!x);
          for (let i = 0; i < heads.length; i++)
            for (let j = i + 1; j < heads.length; j++)
              convPts.push(new THREE.Vector3(heads[i][0], heads[i][1] + 0.25, heads[i][2]), new THREE.Vector3(heads[j][0], heads[j][1] + 0.25, heads[j][2]));
        }
      }
      talk.geometry.setFromPoints(convPts);
      talk.visible = convPts.length > 0;
      if (room) room.frustum.visible = ov.fov;

      // Camera.
      const mode = p.cameraMode ?? "orbit";
      if (mode !== lastMode) {
        controls.enabled = mode !== "camera";
        controls.enableRotate = mode !== "top";
        if (world && mode === "top") {
          const [x0, z0, x1, z1] = roomBounds(world.room);
          const span = Math.max(x1 - x0, z1 - z0);
          controls.target.set((x0 + x1) / 2, 0, (z0 + z1) / 2);
          camera.position.set((x0 + x1) / 2, span * 1.35 + 2, (z0 + z1) / 2 + 0.01);
        } else if (mode === "orbit" && lastMode && lastMode !== "follow") {
          frameRoom();
        }
        camera.fov = 45;
        camera.updateProjectionMatrix();
        lastMode = mode;
      }
      if (mode === "camera" && world) {
        const [x, y, z] = world.room.camera_position;
        const [tx, ty, tz] = world.room.camera_target;
        camera.position.set(x, y, z);
        camera.lookAt(tx, ty, tz);
        if (camera.fov !== world.room.camera_fov) {
          camera.fov = world.room.camera_fov;
          camera.updateProjectionMatrix();
        }
      } else if (mode === "follow" && p.selected) {
        const who = people.find((q) => q.person_id === p.selected);
        const hip = who?.joints.left_hip;
        if (hip) {
          const goal = new THREE.Vector3(hip[0], 0.9, hip[2]);
          const delta = goal.clone().sub(controls.target).multiplyScalar(0.08);
          controls.target.add(delta);
          camera.position.add(delta);
        }
      }
      if (mode !== "camera") controls.update();
      room?.update(camera);
      renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("click", onClick);
      for (const rig of rigs.values()) rig.dispose();
      room?.dispose();
      controls.dispose();
      renderer.dispose();
      host.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={ref} style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden", ...props.style }} />;
}

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { DEFAULT_THEME, type Theme } from "./draw";
import { buildRoom, buildRoute, disposeTree, stickFigureRig, type PersonRig, type RigFactory } from "./scene3d";
import type { WorldTrack } from "./types";
import { routesAt, sampleWorld } from "./world";

export interface RoomView3DProps {
  /** The run's world track: <run>.world.json for a recording, GET /sim/world for a live run. */
  world: WorldTrack | null;
  /**
   * Current scenario seconds, read every frame. A recording passes its playback time
   * (feed.state.t); a live dashboard extrapolates SimStatus.t between polls.
   */
  time: () => number;
  /** Draws the red border. */
  alertActive?: boolean;
  /** person_ids with an open alert; their figures turn red. */
  alertPersonIds?: readonly string[];
  /** Draw each walk's planned route on the floor while it's being walked. */
  showRoutes?: boolean;
  /** How each person is drawn. Default: stick figures. Swap in a rigged-model factory here. */
  rigFactory?: RigFactory;
  theme?: Theme;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * The simulated room in 3D: every resident's true body, including people the camera
 * cannot see. Simulator-only; this is not what the camera reports, and it is badged as
 * such. Drag to orbit, scroll to zoom. Sizes itself to its container's width (4:3).
 */
export function RoomView3D({
  world,
  time,
  alertActive = false,
  alertPersonIds,
  showRoutes = false,
  rigFactory = stickFigureRig,
  theme = DEFAULT_THEME,
  className,
  style,
}: RoomView3DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const props = useRef({ world, time, alertPersonIds, showRoutes, rigFactory, theme });
  props.current = { world, time, alertPersonIds, showRoutes, rigFactory, theme };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    const scene = new THREE.Scene();
    scene.background = new THREE.Color();
    const camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.05, 50);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(2, 5, 3);
    scene.add(sun);

    const rigs = new Map<string | null, PersonRig>();
    let roomOf: WorldTrack | null = null;
    let roomTheme: Theme | null = null;
    let rigsFrom: RigFactory | null = null;
    let room: THREE.Group | null = null;

    const clearRigs = () => {
      for (const rig of rigs.values()) {
        scene.remove(rig.object);
        rig.dispose();
      }
      rigs.clear();
    };

    let raf = 0;
    const render = () => {
      const p = props.current;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      if (w && h && (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr))) {
        renderer.setPixelRatio(dpr);
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
      (scene.background as THREE.Color).set(p.theme.background);

      if (p.world !== roomOf || p.theme !== roomTheme) {
        const newRun = p.world?.scenario_id !== roomOf?.scenario_id || p.world?.seed !== roomOf?.seed;
        if (room) {
          scene.remove(room);
          disposeTree(room);
          room = null;
        }
        clearRigs();
        if (p.world) {
          room = buildRoom(p.world.room, p.theme);
          for (const r of p.world.routes ?? []) {
            const line = buildRoute(r, p.theme);
            line.visible = false;
            line.userData.route = r;
            room.add(line);
          }
          scene.add(room);
          if (newRun) frameRoom(camera, controls, p.world.room);
        }
        roomOf = p.world;
        roomTheme = p.theme;
      }

      if (p.rigFactory !== rigsFrom) {
        clearRigs(); // a new kind of figure
        rigsFrom = p.rigFactory;
      }

      const now = p.time();
      const walking = new Set(p.showRoutes ? routesAt(p.world, now) : []);
      for (const child of room?.children ?? []) {
        if (child.userData.route) child.visible = walking.has(child.userData.route);
      }

      const frame = sampleWorld(p.world, now);
      const seen = new Set<string | null>();
      for (const person of frame.people) {
        seen.add(person.person_id);
        let rig = rigs.get(person.person_id);
        if (!rig) {
          rig = p.rigFactory(person.person_id);
          rigs.set(person.person_id, rig);
          scene.add(rig.object);
        }
        const alert = person.person_id != null && (p.alertPersonIds ?? []).includes(person.person_id);
        rig.update(person, { alert, theme: p.theme });
      }
      for (const [id, rig] of rigs) {
        if (seen.has(id)) continue;
        scene.remove(rig.object);
        rig.dispose();
        rigs.delete(id);
      }

      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(raf);
      clearRigs();
      if (room) disposeTree(room);
      controls.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      className={className}
      style={{
        position: "relative",
        width: "100%",
        aspectRatio: "4 / 3",
        outline: alertActive ? `4px solid ${theme.alert}` : undefined,
        outlineOffset: -4,
        ...style,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100%", height: "100%" }}
        role="img"
        aria-label="Simulated room, 3D world view"
      />
      <div style={{ ...badgeStyle, background: theme.badge.simulation, color: theme.text }}>
        SIMULATED · WORLD VIEW
      </div>
      <div style={{ ...noteStyle, color: theme.muted }}>Not what the camera sees · drag to orbit</div>
      {!world && <div style={{ ...messageStyle, color: theme.text }}>No world data</div>}
    </div>
  );
}

/** Start looking over the simulated camera's shoulder, a little higher, at the room's middle. */
function frameRoom(camera: THREE.PerspectiveCamera, controls: OrbitControls, room: WorldTrack["room"]): void {
  const [, cy, cz] = room.camera_position;
  camera.position.set(0, Math.max(cy + 1.4, room.height), cz + 1.2);
  controls.target.set(0, 0.7, -room.depth / 2);
  controls.update();
}

const badgeStyle: React.CSSProperties = {
  position: "absolute",
  top: 10,
  left: 10,
  padding: "4px 10px",
  font: "700 13px system-ui, sans-serif",
  letterSpacing: 0.5,
  pointerEvents: "none",
};

const noteStyle: React.CSSProperties = {
  position: "absolute",
  bottom: 8,
  left: 10,
  font: "12px system-ui, sans-serif",
  pointerEvents: "none",
};

const messageStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  font: "600 16px system-ui, sans-serif",
  pointerEvents: "none",
};

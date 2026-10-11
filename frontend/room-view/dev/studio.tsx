// Sim Studio: drive the live simulation engine and watch it in 3D, beside what the camera reports.
// Start the backend (python -m uvicorn hearth.api.app:app --port 8000), then npm run dev and open /studio.html.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

import {
  DEFAULT_OVERLAYS,
  LiveClient,
  LIVE_APPEARANCE,
  LiveView3D,
  RoomView,
  actionsFor,
  describePick,
  readSettings,
  roomDiffers,
  websocketFeed,
  writeSettings,
  wsUrl,
  type ActorDetail,
  type CameraMode,
  type EngineSpec,
  type Lighting,
  type LivePerson,
  type Overlays,
  type Pick,
  type ServerMessage,
  type StudioSettings,
} from "../src";


type FallMotion = "trip" | "faint" | "collapse" | "sit_miss" | "stand_up";

const ROOM_ID = "room-01";
const OVERLAY_NAMES: [keyof Overlays, string][] = [
  ["ids", "Tags"], ["activity", "Activity"], ["paths", "Paths"], ["destinations", "Destinations"],
  ["conversations", "Conversations"], ["space", "Personal space"], ["fov", "Camera FOV"],
  ["visibility", "Camera visibility"], ["incidents", "Incidents"], ["slots", "Furniture slots"], ["nav", "Walkable floor"],
];

async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    let msg = text;
    try {
      msg = JSON.parse(text).detail ?? text;
    } catch {
      /* plain text */
    }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json() as Promise<T>;
}

function useLive(client: LiveClient) {
  const [, setTick] = useState(0);
  useEffect(() => {
    let pending = false;
    return client.subscribe(() => {
      if (pending) return;
      pending = true;
      // Re-render the panels at most ~6 times a second; the 3D view draws itself every frame.
      setTimeout(() => {
        pending = false;
        setTick((n) => n + 1);
      }, 160);
    });
  }, [client]);
  return client;
}

interface LogLine {
  key: string;
  at: string;
  kind: string;
  person: string | null;
  text: string;
  alert: boolean;
  live: boolean;
}

interface LinkStatus {
  enabled: boolean;
  source: string;
  state: "off" | "waiting" | "live" | "stale" | "outside";
  person_id: string;
  last_seen: string | null;
  posture: string | null;
  label: string | null;
  alert_open: boolean;
  camera: { preset: string; height: number; tilt: number; fov: number; corner: string; pan: number } | null;
}

interface CalibrateResult {
  ok: boolean;
  step: "walk" | "centre";
  pan: number | null;
  off_by: number | null;
  frames: number;
  height: number | null;
  tilt: number | null;
  rms: number | null;
  reason: string;
}

// Sources that send keypoints (the fall model sends none, so it can't be placed in the room).
const LIVE_SOURCES = ["jetson-01", "replay-01"];

function useMonitorFeed(liveSource: string | null) {
  const feed = useMemo(() => websocketFeed(wsUrl(`/ws/room/${ROOM_ID}`)), []);
  const [log, setLog] = useState<LogLine[]>([]);
  const [alerts, setAlerts] = useState<Record<string, { person: string | null; open: boolean }>>({});
  const eventsById = useRef(new Map<string, string | null>());
  useEffect(() => {
    const off = feed.subscribe((msg: ServerMessage) => {
      if (msg.type === "event") {
        const e = msg.data;
        const live = e.source === liveSource;
        if (e.source !== "sim" && !live) return;
        // The live camera tracks one unnamed person: they're the live avatar in the room.
        const person = live ? `live-${e.source}` : e.person_id;
        eventsById.current.set(e.event_id, person);
        if (e.event_type === "face_present" || e.event_type === "face_absent" || e.event_type === "person_visible") return;
        setLog((l) => [{ key: e.event_id, at: e.timestamp, kind: e.event_type, person, text: "", live,
          alert: e.event_type === "possible_fall" }, ...l].slice(0, 80));  // prettier-ignore
      } else if (msg.type === "alert") {
        const a = msg.data;
        if (a.source !== "sim" && a.source !== liveSource) return;
        const person = eventsById.current.get(a.event_id) ?? (a.source === liveSource ? `live-${a.source}` : null);
        setAlerts((m) => ({ ...m, [a.alert_id]: { person, open: a.status !== "resolved" && !a.recovered_at } }));
      }
    });
    return () => {
      off();
      feed.close();
    };
  }, [feed, liveSource]);
  const alertPersonIds = Object.values(alerts).filter((a) => a.open && a.person).map((a) => a.person!);
  return { feed, log, alertPersonIds };
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <div className="row">
      <label>{props.label}</label>
      <input type="range" min={props.min} max={props.max} step={props.step} value={props.value} onChange={(e) => props.onChange(Number(e.target.value))} />
      <span className="val">{props.fmt ? props.fmt(props.value) : props.value}</span>
    </div>
  );
}

/** What a fresh studio starts with; the URL carries only what differs (see studioUrl). */
const STUDIO_DEFAULTS: StudioSettings = {
  seed: 7, actors: 6, width: 7, depth: 6, density: 0.6, preset: "corner_left", camHeight: 2.4, camTilt: 12, camFov: 62,
  lighting: "day", activity: 1, social: 1, overlays: DEFAULT_OVERLAYS, cameraMode: "orbit",
};

function Studio() {
  const initial = useMemo(() => readSettings(window.location.search, STUDIO_DEFAULTS), []);
  // Opened from a link: offer to load its room if the backend is running a different one.
  const [linked, setLinked] = useState(() => new URLSearchParams(window.location.search).has("seed"));
  const [copied, setCopied] = useState(false);
  const client = useLive(useMemo(() => new LiveClient(wsUrl("/sim/live/ws")), []));
  const [link, setLink] = useState<LinkStatus | null>(null);
  const [linkSource, setLinkSource] = useState(LIVE_SOURCES[0]);
  const [liveHeight, setLiveHeight] = useState(1.7);
  const [showReal, setShowReal] = useState(false);
  const [calibrating, setCalibrating] = useState<"walk" | "centre" | null>(null);
  // The tracked area: a rectangle taped on the real floor, with the camera above a front corner.
  const [areaW, setAreaW] = useState(5);
  const [areaD, setAreaD] = useState(4);
  const [camCorner, setCamCorner] = useState<"left" | "right">("left");
  const [calibration, setCalibration] = useState<CalibrateResult | null>(null);
  const { feed, log, alertPersonIds } = useMonitorFeed(link?.enabled ? link.source : null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [overlays, setOverlays] = useState<Overlays>(initial.overlays);
  const [cameraMode, setCameraMode] = useState<CameraMode>(initial.cameraMode);
  const [hideTruth, setHideTruth] = useState(false);
  const [lighting, setLighting] = useState<Lighting>(initial.lighting);
  const [detail, setDetail] = useState<ActorDetail | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number; pick: Pick } | null>(null);
  const [fallMotion, setFallMotion] = useState<FallMotion | "">("");
  const [recover, setRecover] = useState(true);

  // Room and population settings for the next "Generate".
  const [seed, setSeed] = useState(initial.seed);
  const [actors, setActors] = useState(initial.actors);
  const [width, setWidth] = useState(initial.width);
  const [depth, setDepth] = useState(initial.depth);
  const [density, setDensity] = useState(initial.density);
  const [preset, setPreset] = useState<EngineSpec["camera"]["preset"]>(initial.preset);
  const [camHeight, setCamHeight] = useState(initial.camHeight);
  const [camTilt, setCamTilt] = useState(initial.camTilt);
  const [camFov, setCamFov] = useState(initial.camFov);
  const [activity, setActivity] = useState(initial.activity);
  const [social, setSocial] = useState(initial.social);
  const settings: StudioSettings = { seed, actors, width, depth, density, preset, camHeight, camTilt, camFov, lighting, activity,
    social, overlays, cameraMode };
  // Keep the URL in step with the settings, so the address is always a link to this view.
  const search = writeSettings(settings, STUDIO_DEFAULTS);
  useEffect(() => {
    if (search !== window.location.search) window.history.replaceState(null, "", `${window.location.pathname}${search}${window.location.hash}`);
  }, [search]);
  const copyLink = () => {
    void navigator.clipboard?.writeText(window.location.href).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  const frame = client.latest;
  const world = client.world;
  const state = frame?.state ?? (world ? "paused" : "idle");

  const run = useCallback(async (fn: () => Promise<unknown>) => {
    try {
      setError(null);
      await fn();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }, []);

  const generate = (play = true) =>
    run(() =>
      api("/sim/live/new", {
        seed,
        play,
        spec: {
          actors,
          room: { width, depth, furniture_density: density },
          camera: { preset, height: camHeight, tilt: camTilt, fov: camFov },
          lighting,
          activity_frequency: activity,
          social_frequency: social,
        },
      }).then(() => {
        setSelected(null);
        setLinked(false);
      }),
    );

  const togglePlay = () => run(() => api(state === "playing" ? "/sim/live/pause" : "/sim/live/play", {}));
  const setSpeed = (speed: number) => run(() => api("/sim/live/speed", { speed }));
  const incident = (kind: string, extra: Record<string, unknown> = {}) =>
    run(() => api("/sim/live/incidents", { kind, actor: selected, ...extra }));

  // Live camera link: poll its state.
  useEffect(() => {
    let alive = true;
    const poll = () => api<LinkStatus>("/sim/live/link").then((l) => alive && setLink(l)).catch(() => alive && setLink(null));
    poll();
    const id = setInterval(poll, 1000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  // Follow the backend: the link may already be on (another tab, or set up before this page).
  useEffect(() => {
    if (link?.enabled) setLinkSource(link.source);
  }, [link?.enabled, link?.source]);
  const calibrate = (step: "walk" | "centre") =>
    run(async () => {
      setCalibrating(step);
      setCalibration(null);
      try {
        const r = await api<CalibrateResult>("/sim/live/link/calibrate", { step, height: liveHeight });
        setCalibration(r);
        if (r.ok && r.height != null && r.tilt != null) {
          setCamHeight(r.height);
          setCamTilt(r.tilt);
          setCamFov(43);
        }
      } finally {
        setCalibrating(null);
      }
    });
  const setLinkEnabled = (enabled: boolean, source = linkSource, height = liveHeight) =>
    run(() => api<LinkStatus>("/sim/live/link", { enabled, source, height }).then(setLink));
  // Live only: the simulated room is the tracked area, empty, seen by the corner camera.
  const liveOnly = () =>
    run(async () => {
      await api("/sim/live/new", {
        seed, play: true,
        spec: {
          actors: 0, room: { width: areaW, depth: areaD, height: 3.2, furniture_density: 0 },
          camera: { preset: "jetson_corner", corner: camCorner, height: camHeight, tilt: camTilt, fov: camFov, pan: 0 }, lighting,
        },
      });  // prettier-ignore
      setPreset("jetson_corner");
      setLink(await api<LinkStatus>("/sim/live/link", { enabled: true, source: linkSource, height: liveHeight }));
    });

  // Inspector: poll the selected person's details.
  useEffect(() => {
    if (!selected || selected.startsWith("live-")) {
      setDetail(null);
      return;
    }
    let live = true;
    const poll = () => api<ActorDetail>(`/sim/live/actors/${selected}`).then((d) => live && setDetail(d)).catch(() => live && setDetail(null));
    poll();
    const id = setInterval(poll, 600);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [selected, world?.run_id]);

  // Keyboard: space play/pause, F follow, 1-4 camera, Esc deselect, H hide truth.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "f" || e.key === "F") setCameraMode((m) => (m === "follow" ? "orbit" : "follow"));
      else if (e.key === "Escape") {
        if (menu) setMenu(null);
        else setSelected(null);
      }
      else if (e.key === "h" || e.key === "H") setHideTruth((h) => !h);
      else if (["1", "2", "3", "4"].includes(e.key)) setCameraMode((["orbit", "top", "camera", "follow"] as CameraMode[])[Number(e.key) - 1]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const looks = new Map((world?.actors ?? []).map((a) => [a.person_id, a]));
  const people: LivePerson[] = [...(frame?.people ?? [])].sort((a, b) =>
    a.kind !== b.kind ? (a.kind === "live" ? -1 : 1) : (a.person_id ?? "").localeCompare(b.person_id ?? "", undefined, { numeric: true }));
  const away = (world?.actors ?? []).filter((a) => !people.some((p) => p.person_id === a.person_id));
  const me = people.find((p) => p.person_id === selected) ?? null;
  const t = frame?.t ?? 0;

  return (
    <div className="studio">
      <header className="topbar">
        <div className="brand">Hearth <span>Sim Studio</span></div>
        <span className="pill sim">SIMULATED</span>
        {link?.enabled && <span className={`pill live ${link.state}`}>LIVE · {link.source} · {link.state.toUpperCase()}</span>}
        <span className={`pill ${state === "playing" ? "playing" : ""} ${client.connected ? "" : "off"}`}>
          {client.connected ? state.toUpperCase() : "DISCONNECTED"}
        </span>
        <span className="clock">t = {fmtTime(t)}</span>
        {world && <span className="hint">{world.scenario_id} · {world.room.bounds ? `${(world.room.bounds[2] - world.room.bounds[0]).toFixed(1)} × ${(world.room.bounds[3] - world.room.bounds[1]).toFixed(1)} m` : ""} · {world.actors.length} residents</span>}
        <div className="spacer" />
        <button className="primary" disabled={!world} onClick={togglePlay}>{state === "playing" ? "Pause" : "Play"}</button>
        <button disabled={!world} onClick={() => run(() => api("/sim/live/reset", {}))}>Reset</button>
        <div className="seg">
          {[0.5, 1, 2, 4, 8].map((s) => (
            <button key={s} className={frame?.speed === s ? "on" : ""} disabled={!world} onClick={() => setSpeed(s)}>{s}×</button>
          ))}
        </div>
      </header>

      <aside className="panel left">
        <section>
          <h3>Environment</h3>
          <div className="row">
            <label>Seed</label>
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ width: 90 }} />
            <button title="Random seed" onClick={() => setSeed(Math.floor(Math.random() * 100000))}>🎲</button>
          </div>
          <Slider label="Residents" value={actors} min={0} max={20} step={1} onChange={setActors} />
          <Slider label="Width" value={width} min={4} max={10} step={0.1} onChange={setWidth} fmt={(v) => `${v.toFixed(1)} m`} />
          <Slider label="Depth" value={depth} min={4} max={9} step={0.1} onChange={setDepth} fmt={(v) => `${v.toFixed(1)} m`} />
          <Slider label="Furniture" value={density} min={0} max={1} step={0.05} onChange={setDensity} fmt={(v) => `${Math.round(v * 100)}%`} />
          <div className="row">
            <label>Camera</label>
            <select value={preset} onChange={(e) => { const v = e.target.value as typeof preset; setPreset(v); if (v === "jetson" || v === "jetson_corner") setCamFov(43); }}>
              <option value="corner_left">Corner (left)</option>
              <option value="corner_right">Corner (right)</option>
              <option value="front">Front wall</option>
              <option value="ceiling">Ceiling</option>
              <option value="jetson">Jetson, front wall</option>
              <option value="jetson_corner">Jetson, area corner</option>
            </select>
            {(preset === "jetson" || preset === "jetson_corner") && camFov !== 43 && (
              <button title="Logitech C270: 43° vertical" onClick={() => setCamFov(43)}>C270 lens</button>
            )}
          </div>
          <Slider label="Camera height" value={camHeight} min={1.0} max={3.2} step={0.05} onChange={setCamHeight} fmt={(v) => `${v.toFixed(2)} m`} />
          {(preset === "jetson" || preset === "jetson_corner") && (
            <Slider label="Tilt down" value={camTilt} min={-15} max={60} step={1} onChange={setCamTilt} fmt={(v) => `${v}°`} />
          )}
          <Slider label="Field of view" value={camFov} min={30} max={110} step={1} onChange={setCamFov} fmt={(v) => `${v}°`} />
          <div className="row">
            <label>Lighting</label>
            <div className="seg">
              {(["day", "evening", "dim"] as Lighting[]).map((l) => (
                <button key={l} className={lighting === l ? "on" : ""} onClick={() => setLighting(l)}>{l}</button>
              ))}
            </div>
          </div>
          <div className="btns" style={{ marginTop: 10 }}>
            <button className="primary" onClick={() => generate(true)}>Generate room</button>
            <button onClick={() => generate(false)}>Generate paused</button>
            <button title="Copy a link to this room and view" onClick={copyLink}>{copied ? "Copied" : "Copy link"}</button>
          </div>
          {error && <div className="err">{error}</div>}
        </section>

        <section>
          <h3>Live camera</h3>
          <div className="row">
            <label>Source</label>
            <select value={linkSource} onChange={(e) => { setLinkSource(e.target.value); if (link?.enabled) setLinkEnabled(true, e.target.value); }}>
              {LIVE_SOURCES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <Slider label="Their height" value={liveHeight} min={1.4} max={2.0} step={0.01} fmt={(v) => `${v.toFixed(2)} m`}
            onChange={(v) => { setLiveHeight(v); if (link?.enabled) setLinkEnabled(true, linkSource, v); }} />
          <Slider label="Area width" value={areaW} min={2} max={10} step={0.1} onChange={setAreaW} fmt={(v) => `${v.toFixed(1)} m`} />
          <Slider label="Area depth" value={areaD} min={2} max={9} step={0.1} onChange={setAreaD} fmt={(v) => `${v.toFixed(1)} m`} />
          <div className="row">
            <label>Camera at</label>
            <div className="seg">
              {(["left", "right"] as const).map((c) => (
                <button key={c} className={camCorner === c ? "on" : ""} onClick={() => setCamCorner(c)}>{c} corner</button>
              ))}
            </div>
          </div>
          <div className="btns">
            <button className="primary" onClick={liveOnly} title="The simulated room becomes the taped area, seen from the corner camera">
              Live: tracked area
            </button>
            <button className={link?.enabled ? "on" : ""} disabled={!world} onClick={() => setLinkEnabled(!link?.enabled)}>
              {link?.enabled ? "Unlink" : "Link into this room"}
            </button>
          </div>
          <div className="btns">
            <button disabled={!link?.enabled || calibrating !== null} onClick={() => calibrate("walk")}
              title="12 s: walk slowly all over the area (near, far, left, right), upright, head and hips or knees in view">
              {calibrating === "walk" ? "1 · Walk around the area…" : "1 · Calibrate: walk"}
            </button>
            <button disabled={!link?.enabled || calibrating !== null || link?.camera?.preset !== "jetson_corner"} onClick={() => calibrate("centre")}
              title="3 s: stand still on the X at the centre of the area">
              {calibrating === "centre" ? "2 · Stand on the X…" : "2 · Calibrate: stand on X"}
            </button>
          </div>
          {calibration && (
            <p className="hint" style={{ color: calibration.ok ? "var(--ok)" : "var(--alert)" }}>
              {!calibration.ok ? `Calibration failed: ${calibration.reason}`
                : calibration.step === "centre" ? `Direction set: turned ${calibration.pan?.toFixed(1)}° (you stood ${Math.round((calibration.off_by ?? 0) * 100)} cm from where the X should be)`
                : `Camera set: ${calibration.height?.toFixed(2)} m up, tilted ${calibration.tilt?.toFixed(1)}° (fit ±${Math.round((calibration.rms ?? 0) * 480)} px, ${calibration.frames} frames). Now step 2.`}
            </p>
          )}
          {link?.camera && (
            <p className="hint">Simulated camera: {link.camera.preset === "jetson_corner" ? `${link.camera.corner} corner` : link.camera.preset}, {link.camera.height.toFixed(2)} m, tilt {link.camera.tilt.toFixed(1)}°, pan {link.camera.pan.toFixed(1)}°, fov {Math.round(link.camera.fov)}°</p>
          )}
          {link?.enabled && (
            <p className="hint">
              {link.state === "live" ? `In the room: ${link.label ?? link.posture}${link.alert_open ? " · ALERT OPEN" : ""}`
                : link.state === "waiting" ? `Waiting for ${link.source}: start run_edge.py or edge/tools/replay.py`
                : link.state === "outside" ? "Outside the tracked area: step back inside the tape"
                : `No fresh frames from ${link.source}`}
            </p>
          )}
          <p className="hint">Tape a rectangle on the floor (the tracked area) with an X at its centre, mount the camera
            2.2-2.5 m up above a front corner, aimed at the X and tilted 30-35° down. Press "Live: tracked area", then
            calibrate: walk, then stand on the X. The shaded floor (Camera FOV overlay) is what the camera sees.</p>
        </section>

        <section>
          <h3>Behaviour</h3>
          <Slider label="Activity" value={activity} min={0.25} max={4} step={0.25} fmt={(v) => `${v}×`}
            onChange={(v) => { setActivity(v); if (world) run(() => api("/sim/live/config", { activity_frequency: v })); }} />
          <Slider label="Socialising" value={social} min={0} max={4} step={0.25} fmt={(v) => `${v}×`}
            onChange={(v) => { setSocial(v); if (world) run(() => api("/sim/live/config", { social_frequency: v })); }} />
          <p className="hint">Activity scales how long people stay with what they're doing; socialising how often they start conversations.</p>
        </section>

        <section>
          <h3>Incidents {selected ? `· ${selected.toUpperCase()}` : "· anyone suitable"}</h3>
          <div className="row">
            <label>Fall</label>
            <select value={fallMotion} onChange={(e) => setFallMotion(e.target.value as FallMotion | "")}>
              <option value="">suits posture</option>
              <option value="trip">trip</option>
              <option value="faint">faint</option>
              <option value="collapse">collapse</option>
              <option value="sit_miss">miss the seat</option>
              <option value="stand_up">while standing up</option>
            </select>
          </div>
          <div className="row">
            <label>Recovers</label>
            <input type="checkbox" checked={recover} onChange={(e) => setRecover(e.target.checked)} />
            <span className="hint">after 30 s</span>
          </div>
          <div className="btns">
            <button className="danger" disabled={!world} onClick={() => incident("fall", { motion: fallMotion || null, recover, duration: 30 })}>Fall</button>
            <button disabled={!world} onClick={() => incident("inactivity", { duration: 120 })}>Inactivity</button>
            <button disabled={!world} onClick={() => incident("pacing", { repeats: 6 })}>Pacing</button>
            <button disabled={!world} onClick={() => incident("exit_loiter", { repeats: 3 })}>Exit loitering</button>
            <button disabled={!selected} onClick={() => run(() => api(`/sim/live/actors/${selected}/reset`, {}))}>Reset person</button>
          </div>
          <p className="hint">Falls are staged where the camera has a clear view. Select someone the camera can see (visibility in the inspector) for a fair demo.</p>
        </section>

        <section>
          <h3>Residents</h3>
          <div className="roster">
            {people.map((p) => {
              const info = looks.get(p.person_id ?? "");
              const live = p.kind === "live";
              return (
                <div key={p.person_id} className={`person ${live ? "live" : ""} ${selected === p.person_id ? "sel" : ""}`} onClick={() => setSelected(selected === p.person_id ? null : p.person_id)}>
                  <span className="swatch" style={{ background: live ? LIVE_APPEARANCE.top : info?.appearance.top }} />
                  <span className="pid">{live ? "LIVE" : p.person_id?.toUpperCase()}</span>
                  <span className="doing">{p.incident ? `⚠ ${p.incident}` : p.activity}</span>
                  {!hideTruth && <span className={`tag ${p.label}`}>{p.label.replace("_intentional", "")}</span>}
                </div>
              );
            })}
            {away.map((a) => (
              <div key={a.person_id} className="person" style={{ opacity: 0.5 }}>
                <span className="swatch" style={{ background: a.appearance.top }} />
                <span className="pid">{a.person_id.toUpperCase()}</span>
                <span className="doing">out of the room</span>
                <span className="tag absent">away</span>
              </div>
            ))}
          </div>
        </section>
      </aside>

      <main className="view">
        <LiveView3D client={client} overlays={overlays} cameraMode={cameraMode} selected={selected}
          onSelect={(id) => {
            setMenu(null);
            setSelected(id);
          }}
          onPick={(pick, at) => {
            if (!selected || !world) return false;
            setMenu({ ...at, pick });
            return true;
          }}
          alertPersonIds={alertPersonIds} hideTruth={hideTruth} lighting={lighting} />
        {linked && world && roomDiffers(settings, world) && (
          <div className="link-banner">
            <span>This link is for a different room (seed {seed}, {actors} people). The backend is running seed {world.seed}.</span>
            <button className="primary" onClick={() => {
              setLinked(false);
              generate(true);
            }}>Load this room</button>
            <button onClick={() => setLinked(false)}>Keep the current one</button>
          </div>
        )}
        {menu && selected && world && (
          <div className="cmd-menu" style={{ left: menu.x, top: menu.y }}>
            <div className="cmd-title">{selected.toUpperCase()} â†’ {describePick(menu.pick, world)}</div>
            {actionsFor(menu.pick, world).map(([label, cmd]) => (
              <button key={label} onClick={() => {
                setMenu(null);
                run(() => api(`/sim/live/actors/${selected}/command`, cmd));
              }}>{label}</button>
            ))}
            {actionsFor(menu.pick, world).length === 0 && <div className="hint">Nothing to do here.</div>}
            <button className="cancel" onClick={() => setMenu(null)}>Cancel</button>
          </div>
        )}
        <div className="overlay-bar">
          <div className="chips">
            {OVERLAY_NAMES.map(([k, name]) => (
              <span key={k} className={`chip ${overlays[k] ? "on" : ""}`} onClick={() => setOverlays({ ...overlays, [k]: !overlays[k] })}>{name}</span>
            ))}
            <span className={`chip ${hideTruth ? "on" : ""}`} onClick={() => setHideTruth(!hideTruth)}>Hide ground truth</span>
          </div>
          <div className="seg" style={{ background: "rgba(15,18,22,0.8)" }}>
            {(["orbit", "top", "camera", "follow"] as CameraMode[]).map((m, i) => (
              <button key={m} className={cameraMode === m ? "on" : ""} disabled={m === "follow" && !selected} onClick={() => setCameraMode(m)} title={`Key ${i + 1}`}>
                {{ orbit: "Orbit", top: "Top", camera: "Camera POV", follow: "Follow" }[m]}
              </button>
            ))}
          </div>
        </div>
        <div className="badge-world">SIMULATED · WORLD VIEW — true bodies, not what the camera reports · <kbd>Space</kbd> play <kbd>1-4</kbd> views <kbd>F</kbd> follow <kbd>H</kbd> hide truth</div>
        {!world && (
          <div className="empty">
            <div>
              <b>No room yet</b>
              Pick settings on the left and press <i>Generate room</i>.
              {!client.connected && <div className="err">Not connected to the backend (WS /sim/live/ws). Is it running?</div>}
            </div>
          </div>
        )}
      </main>

      <aside className="panel right">
        <section>
          <h3>Inspector</h3>
          {!me && <p className="hint">Click a person in the room or the list to inspect them.</p>}
          {me && me.kind !== "live" && <p className="hint">Click furniture or the floor to tell them what to do there.</p>}
          {me && me.kind === "live" && (
            <dl className="kv">
              <dt>Live person</dt>
              <dd><b>{link?.source ?? me.person_id}</b> · a real camera, not simulated</dd>
              <dt>Monitor says</dt>
              <dd>{me.label} ({me.posture})</dd>
              <dt>Feed</dt>
              <dd>{link ? `${link.state}${link.last_seen ? ` · last frame ${new Date(link.last_seen).toLocaleTimeString()}` : ""}` : "…"}</dd>
              <dt>Alert</dt>
              <dd>{link?.alert_open ? <span style={{ color: "var(--alert)" }}>possible fall open</span> : "—"}</dd>
              <dt>Helper</dt>
              <dd>{people.find((q) => q.intent === "help")?.person_id?.toUpperCase() ?? "—"}</dd>
            </dl>
          )}
          {me && me.kind !== "live" && (
            <dl className="kv">
              <dt>Resident</dt>
              <dd><b>{me.person_id?.toUpperCase()}</b> · {detail?.info.archetype ?? looks.get(me.person_id ?? "")?.archetype}</dd>
              <dt>Looks</dt>
              <dd>
                {(() => {
                  const ap = looks.get(me.person_id ?? "")?.appearance;
                  return ap ? [ap.top, ap.bottom, ap.skin, ap.hair].map((c) => <span key={c} className="swatch" style={{ display: "inline-block", background: c, marginRight: 4 }} />) : null;
                })()}
              </dd>
              <dt>Height · pace</dt>
              <dd>{detail ? `${detail.info.height.toFixed(2)} m · ${detail.info.speed.toFixed(2)} m/s` : "…"}</dd>
              <dt>Sociability</dt>
              <dd>{detail ? Math.round(detail.info.sociability * 100) + "%" : "…"} · {detail?.info.mobility}</dd>
              <dt>Doing</dt>
              <dd>{me.activity}</dd>
              <dt>Plan</dt>
              <dd>{me.intent}{detail ? ` · for ${fmtTime(t - detail.since)}` : ""}</dd>
              {!hideTruth && <><dt>Truth label</dt><dd><span className={`tag ${me.label}`}>{me.label}</span></dd></>}
              <dt>Posture</dt>
              <dd>{me.posture}</dd>
              <dt>Heading to</dt>
              <dd>{me.destination ? `(${me.destination[0].toFixed(2)}, ${me.destination[1].toFixed(2)})` : "—"}</dd>
              <dt>Talking</dt>
              <dd>{me.conversation ? (frame?.conversations.find((c) => c.conv_id === me.conversation)?.members.filter((m) => m !== me.person_id).map((m) => m.toUpperCase()).join(", ") ?? "…") : "—"}</dd>
              <dt>Camera sees</dt>
              <dd>
                <span style={{ color: me.visible > 0.6 ? "var(--ok)" : me.visible > 0.25 ? "var(--warn)" : "var(--alert)" }}>{Math.round(me.visible * 100)}% of joints</span>
              </dd>
              <dt>Incident</dt>
              <dd>{me.incident ?? "—"}</dd>
              <dt>Next</dt>
              <dd>{detail?.plan.length ? detail.plan.slice(0, 6).join(" → ") : "—"}</dd>
              <dt>Recently</dt>
              <dd className="history">{detail?.history.slice(-8).reverse().map(([i, at], k) => <span key={k} className="tag">{i} {fmtTime(at)}</span>)}</dd>
              {detail && Object.keys(detail.cooldowns).length > 0 && (<><dt>Cooling down</dt><dd>{Object.entries(detail.cooldowns).map(([k, v]) => `${k} ${Math.max(0, Math.round(v - t))}s`).join(", ")}</dd></>)}
              {detail && detail.incidents.length > 0 && (
                <>
                  <dt>Incidents</dt>
                  <dd>
                    {detail.incidents.map((r) => (
                      <div key={r.incident_id}>
                        {r.kind}{r.motion ? ` (${r.motion})` : ""} at {fmtTime(r.t_start)}
                        {r.t_floor != null ? `, down ${fmtTime(r.t_floor)}` : ""}
                        {r.t_recovered != null ? `, up ${fmtTime(r.t_recovered)}` : r.t_end == null ? ", ongoing" : ""}
                      </div>
                    ))}
                  </dd>
                </>
              )}
            </dl>
          )}
        </section>
        <section>
          <h3>Conversations</h3>
          {(frame?.conversations.length ?? 0) === 0 && <p className="hint">Nobody is talking right now.</p>}
          {frame?.conversations.map((c) => (
            <div key={c.conv_id} className="row">
              <span className="tag">{c.conv_id}</span>
              {c.members.map((m) => (
                <button key={m} className={selected === m ? "on" : ""} onClick={() => setSelected(m)}>{m.toUpperCase()}</button>
              ))}
            </div>
          ))}
        </section>
      </aside>

      <footer className="bottom">
        <div className="cam">
          <h3 style={{ margin: "0 0 6px", fontSize: 11, color: "var(--muted)", letterSpacing: 0.8 }}>
            WHAT THE {showReal && link?.enabled ? `REAL CAMERA (${link.source.toUpperCase()})` : "SIMULATED CAMERA"} REPORTS (MONITOR INPUT)
            {link?.enabled && (
              <span className={`chip ${showReal ? "on" : ""}`} style={{ marginLeft: 8 }} onClick={() => setShowReal(!showReal)}>Real camera</span>
            )}
          </h3>
          <div className="cam-inner">
            {showReal && link?.enabled ? (
              <RoomView key={link.source} feed={feed} scene={null} roomId={ROOM_ID} source={link.source}
                alertActive={link.alert_open} style={{ height: "100%", width: "auto", maxWidth: "100%" }} />
            ) : (
              <RoomView feed={feed} scene={world?.scene ?? null} roomId={ROOM_ID} source="sim" alertPersonIds={alertPersonIds} alertActive={alertPersonIds.length > 0}
                style={{ height: "100%", width: "auto", maxWidth: "100%" }} />
            )}
          </div>
        </div>
        <div className="log">
          <h3 style={{ margin: "0 0 6px", fontSize: 11, color: "var(--muted)", letterSpacing: 0.8 }}>MONITOR EVENTS (SIMULATED{link?.enabled ? " + LIVE" : ""})</h3>
          <ul>
            {log.length === 0 && <li><span /><span className="hint">No events yet.</span></li>}
            {log.map((l) => (
              <li key={l.key} className={l.alert ? "alert" : ""}>
                <span className="when">{new Date(l.at).toLocaleTimeString()}</span>
                <span>{l.kind.replace(/_/g, " ")}</span>
                <span>{l.live ? "LIVE" : l.person?.toUpperCase() ?? ""}</span>
                <span className="hint">{l.text}</span>
              </li>
            ))}
          </ul>
        </div>
      </footer>
    </div>
  );
}

function fmtTime(s: number): string {
  s = Math.max(0, s);
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

createRoot(document.getElementById("root")!).render(<Studio />);

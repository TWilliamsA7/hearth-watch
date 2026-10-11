// Dev page: plays recorded simulator runs through the same RoomView (and RoomView3D) the dashboard uses.
// Generate runs first:  python -m hearth.sim run --all
import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";

import {
  RecordedFeed,
  RoomView,
  RoomView3D,
  createMannequinRig,
  stickFigureRig,
  parseStream,
  type GroundTruth,
  type Observation,
  type PlaybackState,
  type RunInfo,
  type WorldTrack,
} from "../src";

function useRuns(): RunInfo[] {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  useEffect(() => {
    fetch("/dev-api/runs").then((r) => r.json()).then(setRuns);
  }, []);
  return runs;
}

function useRun(run: RunInfo | null) {
  const [loaded, setLoaded] = useState<{ feed: RecordedFeed; truth: GroundTruth[]; world: WorldTrack | null } | null>(null);
  useEffect(() => {
    if (!run) return;
    let feed: RecordedFeed | null = null;
    Promise.all([
      fetch(`/dev-api/runs/${run.run_id}/stream`).then((r) => r.text()),
      fetch(`/dev-api/runs/${run.run_id}/truth`).then((r) => r.text()),
      // Runs recorded before the world track existed have none: re-run the simulator.
      fetch(`/dev-api/runs/${run.run_id}/world`).then((r) => (r.ok ? (r.json() as Promise<WorldTrack>) : null)),
    ]).then(([stream, truth, world]) => {
      feed = new RecordedFeed(parseStream(stream), { start: run.start, duration: run.duration });
      setLoaded({ feed, truth: truth.split("\n").filter(Boolean).map((l) => JSON.parse(l)), world });
    });
    return () => feed?.dispose();
  }, [run]);
  return loaded;
}

// Deep links for debugging and demo prep: ?run=possible-fall-01-s42&t=13.5&view=both
const params = new URLSearchParams(window.location.search);

type View = "2d" | "3d" | "both";
const VIEWS: View[] = ["2d", "3d", "both"];

// ?rig=stick to start with stick figures.
const RIGS = { mannequin: createMannequinRig(), stick: stickFigureRig };
type Rig = keyof typeof RIGS;

function App() {
  const runs = useRuns();
  const [runId, setRunId] = useState<string>(params.get("run") ?? "");
  const run = useMemo(() => runs.find((r) => r.run_id === runId) ?? runs[0] ?? null, [runs, runId]);
  const loaded = useRun(run);
  const [state, setState] = useState<PlaybackState | null>(null);
  const [last, setLast] = useState<Observation | null>(null);
  const [view, setView] = useState<View>(VIEWS.find((v) => v === params.get("view")) ?? "both");
  const [rig, setRig] = useState<Rig>(params.get("rig") === "stick" ? "stick" : "mannequin");
  const [routes, setRoutes] = useState(params.get("routes") !== "0");

  useEffect(() => {
    if (!loaded) return;
    setLast(null);
    const offState = loaded.feed.onState(setState);
    const offMsg = loaded.feed.subscribe((m) => m.type === "observation" && setLast(m.data));
    loaded.feed.seek(Number(params.get("t") ?? 0));
    return () => {
      offState();
      offMsg();
    };
  }, [loaded]);

  if (!runs.length)
    return (
      <p>
        No runs found. Generate some with <code>python -m hearth.sim run --all</code>, or open the{" "}
        <a href="./studio.html" style={{ color: "#b9aaff" }}>Sim Studio</a> to run the live engine.
      </p>
    );
  const t = state?.t ?? 0;
  const now = loaded?.truth.filter((g) => g.t_start <= t && t < g.t_end) ?? [];

  return (
    <>
      <div className="bar">
        <a href="./studio.html" style={{ color: "#b9aaff" }}>Sim Studio (live) →</a>
        <select value={run?.run_id} onChange={(e) => setRunId(e.target.value)}>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} — {r.title}
            </option>
          ))}
        </select>
        <button onClick={() => (state?.playing ? loaded?.feed.pause() : loaded?.feed.play())}>
          {state?.playing ? "Pause" : "Play"}
        </button>
        {[0.5, 1, 2, 4].map((s) => (
          <button key={s} disabled={state?.speed === s} onClick={() => loaded?.feed.setSpeed(s)}>
            {s}×
          </button>
        ))}
        {VIEWS.map((v) => (
          <button key={v} disabled={view === v} onClick={() => setView(v)}>
            {v.toUpperCase()}
          </button>
        ))}
        {(Object.keys(RIGS) as Rig[]).map((r) => (
          <button key={r} disabled={rig === r || view === "2d"} onClick={() => setRig(r)}>
            {r[0].toUpperCase() + r.slice(1)}
          </button>
        ))}
        <label>
          <input type="checkbox" checked={routes} disabled={view === "2d"} onChange={(e) => setRoutes(e.target.checked)} />
          Routes
        </label>
      </div>
      {loaded && run && (
        <div className="views">
          {view !== "3d" && <RoomView feed={loaded.feed} scene={run.scene} roomId={run.room_id} />}
          {view !== "2d" && <RoomView3D world={loaded.world} time={() => loaded.feed.state.t} rigFactory={RIGS[rig]} showRoutes={routes} />}
        </div>
      )}
      <div className="bar">
        <input type="range" min={0} max={state?.duration ?? 0} step={0.05} value={t}
          onChange={(e) => loaded?.feed.seek(Number(e.target.value))} />
        <span>{t.toFixed(2)} / {(state?.duration ?? 0).toFixed(1)} s</span>
      </div>
      <div className="truth">
        Ground truth (dev only, never shown to the monitor):{" "}
        {now.length === 0 && <b>—</b>}
        {now.map((g) => (
          <span key={g.segment_id}>
            {g.person_id ? `${g.person_id.toUpperCase()} ` : ""}
            <b>{g.truth_label}</b>
            {g.expects_alert ? " (expects alert)" : ""}
            {"  "}
          </span>
        ))}
      </div>
      <pre>{last ? JSON.stringify({ person_detected: last.person_detected, face_present: last.face_present, ...last.features }, null, 1) : ""}</pre>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<App />);

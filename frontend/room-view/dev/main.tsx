// Dev page: plays recorded simulator runs through the same RoomView the dashboard uses.
// Generate runs first:  python -m hearth.sim run --all
import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";

import { RecordedFeed, RoomView, parseStream, type GroundTruth, type Observation, type PlaybackState, type RunInfo } from "../src";

function useRuns(): RunInfo[] {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  useEffect(() => {
    fetch("/dev-api/runs").then((r) => r.json()).then(setRuns);
  }, []);
  return runs;
}

function useRun(run: RunInfo | null) {
  const [loaded, setLoaded] = useState<{ feed: RecordedFeed; truth: GroundTruth[] } | null>(null);
  useEffect(() => {
    if (!run) return;
    let feed: RecordedFeed | null = null;
    Promise.all([
      fetch(`/dev-api/runs/${run.run_id}/stream`).then((r) => r.text()),
      fetch(`/dev-api/runs/${run.run_id}/truth`).then((r) => r.text()),
    ]).then(([stream, truth]) => {
      feed = new RecordedFeed(parseStream(stream), { start: run.start, duration: run.duration });
      setLoaded({ feed, truth: truth.split("\n").filter(Boolean).map((l) => JSON.parse(l)) });
    });
    return () => feed?.dispose();
  }, [run]);
  return loaded;
}

// Deep links for debugging and demo prep: ?run=possible-fall-01-s42&t=13.5
const params = new URLSearchParams(window.location.search);

function App() {
  const runs = useRuns();
  const [runId, setRunId] = useState<string>(params.get("run") ?? "");
  const run = useMemo(() => runs.find((r) => r.run_id === runId) ?? runs[0] ?? null, [runs, runId]);
  const loaded = useRun(run);
  const [state, setState] = useState<PlaybackState | null>(null);
  const [last, setLast] = useState<Observation | null>(null);

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

  if (!runs.length) return <p>No runs found. Generate some with <code>python -m hearth.sim run --all</code>.</p>;
  const t = state?.t ?? 0;
  const segment = loaded?.truth.find((g) => g.t_start <= t && t < g.t_end) ?? null;

  return (
    <>
      <div className="bar">
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
      </div>
      {loaded && run && <RoomView feed={loaded.feed} scene={run.scene} roomId={run.room_id} />}
      <div className="bar">
        <input type="range" min={0} max={state?.duration ?? 0} step={0.05} value={t}
          onChange={(e) => loaded?.feed.seek(Number(e.target.value))} />
        <span>{t.toFixed(2)} / {(state?.duration ?? 0).toFixed(1)} s</span>
      </div>
      <div className="truth">
        Ground truth (dev only, never shown to the monitor): <b>{segment?.truth_label ?? "—"}</b>
        {segment?.expects_alert ? " · expects alert" : ""}
      </div>
      <pre>{last ? JSON.stringify({ person_detected: last.person_detected, face_present: last.face_present, ...last.features }, null, 1) : ""}</pre>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<App />);

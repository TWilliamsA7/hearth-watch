import { useEffect, useState } from 'react';
import { scenarios, simStatus, simCommand, evaluation } from './api';
import type { ScenarioInfo, SimStatus, EvaluationReport, EvaluationSummary } from './generated/simulation';
import Icon from './Icon';

export default function SimulatorPanel({ base }: { base: string | null }) {
  const [catalog, setCatalog] = useState<ScenarioInfo[]>([]);
  const [selected, setSelected] = useState('');
  const [status, setStatus] = useState<SimStatus | null>(null);
  const [report, setReport] = useState<EvaluationReport | EvaluationSummary | null>(null);
  const [error, setError] = useState('');
  const [resultError, setResultError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setCatalog([]); setStatus(null); setReport(null); setError(''); setResultError('');
    if (!base) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    scenarios(base).then(values => { if (!cancelled) { setCatalog(values); setSelected(values[0]?.scenario_id ?? ''); } }).catch(e => { if (!cancelled) setError(e.message); });
    async function poll() {
      try { const value = await simStatus(base!); if (!cancelled) setStatus(value); }
      catch (e) { if (!cancelled) { setStatus(null); setError((e as Error).message); } }
      if (!cancelled) timer = setTimeout(poll, 1000);
    }
    void poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [base]);
  useEffect(() => {
    setReport(null); setResultError('');
    if (!base || status?.state !== 'ended' || !status.run_id) return;
    let cancelled = false;
    evaluation(base, status.run_id).then(value => { if (!cancelled) setReport(value); }).catch(e => { if (!cancelled) setResultError(e.message); });
    return () => { cancelled = true; };
  }, [base, status?.state, status?.run_id]);
  async function command(action: 'load' | 'play' | 'pause' | 'reset' | 'replay') {
    if (!base) return;
    setBusy(true); setError('');
    try {
      if (action === 'replay') await simCommand(base, 'reset');
      setStatus(await simCommand(base, action === 'replay' ? 'play' : action, selected));
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function loadSummary() {
    if (!base) return;
    setBusy(true); setResultError(''); setReport(null);
    try { setReport(await evaluation(base)); } catch (e) { setResultError((e as Error).message); }
    finally { setBusy(false); }
  }
  const loaded = status && status.state !== 'idle';
  const delays = report && 'detection_delays' in report ? report.detection_delays : [];
  const delay = report && 'delay_median' in report ? report.delay_median : delays.length ? delays.reduce((a,b) => a+b,0) / delays.length : null;
  return <>
    <section className="card scenario-card" id="simulation"><div className="section-heading"><div><p className="eyebrow">Simulation</p><h2>Scripted scenario controls</h2></div><span className="status neutral">Scripted simulation</span></div>
      {!base ? <div className="empty-state compact"><span className="empty-icon"><Icon name="play" /></span><h3>Choose a scenario to practice</h3><p>Connect in Connection settings, then choose Simulation to run a scenario.</p></div> : <>
        <label>Scenario<select value={selected} onChange={e => setSelected(e.target.value)} disabled={busy || !catalog.length}>{catalog.map(item => <option key={item.scenario_id} value={item.scenario_id}>{item.title}</option>)}</select></label>
        <p>{catalog.find(item => item.scenario_id === selected)?.description}</p>
        <div className="buttons"><button disabled={busy || !selected} onClick={() => command('load')}>Load</button><button disabled={busy || !loaded || status?.state === 'playing'} onClick={() => command('play')}>Play</button><button disabled={busy || status?.state !== 'playing'} onClick={() => command('pause')}>Pause</button><button disabled={busy || !loaded} onClick={() => command('reset')}>Reset</button><button disabled={busy || !loaded} onClick={() => command('replay')}>Run again</button></div>
        <p aria-live="polite">{status ? `${status.state} · ${status.t.toFixed(1)} / ${status.duration?.toFixed(1) ?? '—'} seconds` : 'Simulator unavailable'}</p>
        {status?.duration != null && <progress className="scenario-progress" value={status.t} max={status.duration || 1} aria-label="Scenario progress" />}
        <p className="small">Playing a scripted scenario pauses the live 3D room. Its Play button resumes the live room. Run again resets the same seeded simulation. It is not a recorded-camera replay.</p>
        {error && <p role="alert" className="error">{error}</p>}
      </>}
    </section>
    <section className="card evaluation" id="evaluation"><div className="section-heading"><div><p className="eyebrow">After-action review</p><h2>Scenario evaluation</h2></div><button className="secondary" disabled={!base || busy} onClick={loadSummary}><Icon name="chart" />Load batch summary</button></div>
      <p>{report ? 'run_id' in report ? `Completed run: ${report.run_id}` : `${report.runs} simulated runs across ${report.scenarios} scenarios` : 'Awaiting a completed run or saved batch results.'}</p>
      <div className="metrics">{[
        [report && 'delay_median' in report ? 'Median detection delay' : 'Mean detection delay', delay == null ? '—' : `${delay.toFixed(2)} s`],
        ['True positives', report?.counts.true_positives ?? '—'], ['False alarms', report?.counts.false_positives ?? '—'],
        ['Missed incidents', report?.counts.missed ?? '—'], ['Run count', report ? 'runs' in report ? report.runs : 1 : '—']
      ].map(([label,value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
      {report && 'segments' in report && <div className="table-scroll"><table><caption>After-action comparison: ground truth versus monitor results</caption><thead><tr><th>Activity</th><th>Expected event</th><th>Detected</th><th>Delay</th><th>False alarms</th></tr></thead><tbody>{report.segments.map(segment => <tr key={`${segment.person_id}:${segment.segment_id}`}><td>{segment.truth_label.replaceAll('_',' ')}</td><td>{segment.expected_event ?? 'None'}</td><td>{segment.detected == null ? 'Not applicable' : segment.detected ? 'Yes' : 'No'}</td><td>{segment.delay == null ? '—' : `${segment.delay.toFixed(2)} s`}</td><td>{segment.false_alerts}</td></tr>)}</tbody></table></div>}
      {resultError && <p role="alert" className="error">{resultError}</p>}
      <p className="small">These are simulator measurements, not clinical or live-camera accuracy.</p>
    </section>
  </>;
}

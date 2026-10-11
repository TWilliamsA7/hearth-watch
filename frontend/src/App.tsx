import { useMemo, useState } from 'react';
import type { Event } from './generated/contract';
import { exampleEvent, exampleObservation, type Preview } from './demoData';
import { sourceKey, visibleObservations, type SourceMode } from './feedState';
import { acknowledgeAlert } from './api';
import { useLiveSimulation } from './useLiveSimulation';
import LiveSimulationView from './LiveSimulationView';
import { useRoomFeed } from './useRoomFeed';
import RoomView from './RoomView';
import SimulatorPanel from './SimulatorPanel';
import ResidentPanel from './ResidentPanel';
import Icon from './Icon';
import AppointmentPanel from './AppointmentPanel';
import FeatureSpark from './FeatureSpark';

const sourceLabels = { edge: 'Live camera', simulation: 'Simulation', replay: 'Recorded replay', manual: 'Manual request' };
function savedBackendAddress() {
  try {
    const saved = localStorage.getItem('hearth-monitoring-address');
    if (saved) {
      const url = new URL(saved);
      if (['http:', 'https:'].includes(url.protocol) && !url.username && !url.password) return url.origin;
    }
  } catch { /* Storage may be unavailable; use the standard local backend. */ }
  return 'http://127.0.0.1:8000';
}
export default function App() {
  const [preview, setPreview] = useState<Preview>('standing');
  const example = useMemo(() => exampleObservation(preview), [preview]);
  const [view, setView] = useState<'caregiver' | 'resident'>('caregiver');
  const [exampleEvents, setExampleEvents] = useState<Event[]>([]);
  const [acknowledged, setAcknowledged] = useState(false);
  const [address, setAddress] = useState(savedBackendAddress);
  const [base, setBase] = useState<string | null>(savedBackendAddress);
  const [mode, setMode] = useState<SourceMode>('simulation');
  const [selectedSource, setSelectedSource] = useState('');
  const [error, setError] = useState('');
  const [pending, setPending] = useState<string | null>(null);
  const [previewStarted, setPreviewStarted] = useState<number | null>(null);
  const feed = useRoomFeed(base, mode);
  const simulation = useLiveSimulation(base && mode === 'simulation' ? base : null);
  const allObservations = base ? visibleObservations(feed.state, feed.now, feed.connected).filter(item => item.source_type === mode) : preview === 'offline' ? [] : [example];
  const keys = [...new Set([...Object.keys(feed.state.sources), ...Object.values(feed.state.observations).map(item => sourceKey(item.data))])].filter(key => key.startsWith(`${mode}:`));
  const activeSource = keys.includes(selectedSource) ? selectedSource : keys[0];
  const observations = base ? allObservations.filter(item => sourceKey(item) === activeSource) : allObservations;
  const status = feed.state.sources[activeSource];
  const offline = !observations.length;
  const events = base ? feed.state.events.filter(event => event.source_type === mode || event.source_type === 'manual') : exampleEvents;
  const alerts = Object.values(feed.state.alerts).filter(alert => (alert.source_type === mode || alert.source_type === 'manual') && alert.status !== 'resolved');
  const incidentCount = base ? alerts.length : preview === 'alert' ? 1 : 0;
  const person = offline ? 'Unknown' : observations.some(o => o.person_detected) ? 'Visible' : 'Not visible';
  const face = offline ? 'Unknown' : observations.some(o => o.face_present === true) ? 'Visible' : observations.some(o => o.face_present === false) ? 'Not visible' : 'Not run';
  const features = observations.find(o => o.person_detected)?.features;
  const fall = alerts.find(alert => alert.kind === 'possible_fall');
  const fallAge = fall?.source_type === 'edge' ? Math.max(0, Math.floor((feed.now - Date.parse(fall.opened_at)) / 1000)) : null;
  function choosePreview(next: Preview) {
    setPreview(next); setAcknowledged(false);
    setPreviewStarted(next === 'alert' ? Date.now() : null);
    setExampleEvents([exampleEvent(next === 'alert' ? 'possible_fall' : next === 'offline' ? 'source_offline' : 'person_visible')]);
  }
  function connect() {
    try {
      const url = new URL(address);
      if (!['http:','https:'].includes(url.protocol) || url.username || url.password) throw new Error();
      setBase(url.origin); setError('');
      try { localStorage.setItem('hearth-monitoring-address', url.origin); } catch { /* Connection still works without storage. */ }
    } catch { setError('Enter an HTTP or HTTPS monitoring address, such as http://127.0.0.1:8000.'); }
  }
  async function acknowledge(id: string) {
    if (!base) return;
    setPending(id); setError('');
    try { await acknowledgeAlert(base,id); } catch (e) { setError((e as Error).message); }
    finally { setPending(null); }
  }
  async function runFall() {
    if (!base) { choosePreview('alert'); return; }
    await simulation.command('fall');
  }
  async function resetDemo() {
    if (!base) { choosePreview('standing'); return; }
    await simulation.command('reset');
  }
  const sourceLabel = base ? sourceLabels[mode] : 'UI preview';
  const roomStatus = offline ? 'Observation unknown' : incidentCount ? 'Possible fall / help alert' : base ? 'Observations available' : 'Preview · no incident';
  const sampleId = observations.find(o => o.person_detected)?.obs_id ?? 'unknown';
  const previewAge = previewStarted ? Math.max(0, Math.floor((feed.now - previewStarted) / 1000)) : 0;
  const displayAge = base ? fallAge : previewAge;
  return <main className="dashboard" id="overview">
    <a className="skip-link" href="#main-content">Skip to dashboard</a>
    <header className="page-header">
      <a className="brand" href="#overview"><span><h1 aria-label="HearthWatch">HearthWatc<span className="wordmark-tail" aria-hidden="true">h<svg className="heart-flourish" viewBox="0 0 36 36" fill="none"><path d="M 1 27 C 6 31 10 29 14 25 C 19 21 29 16 29 10 C 29 4 22 3 18 9 C 14 3 7 4 7 10 C 7 16 14 20 18 23" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" /></svg></span></h1></span></a>
      <div className="header-actions"><div className="view-toggle" role="group" aria-label="Dashboard view"><button aria-pressed={view === 'caregiver'} className={view === 'caregiver' ? 'selected' : ''} onClick={() => setView('caregiver')}>Caregiver</button><button aria-pressed={view === 'resident'} className={view === 'resident' ? 'selected' : ''} onClick={() => setView('resident')}>Patient</button></div><span className={`status ${offline ? 'unknown' : incidentCount ? 'warning' : ''}`}><span className="tiny-dot" />{roomStatus}</span>{view === 'caregiver' && <><button className="go" disabled={!!base && (mode !== 'simulation' || !simulation.ready || simulation.busy)} onClick={runFall}>Simulate fall</button><button disabled={!!base && (mode !== 'simulation' || !simulation.ready || simulation.busy)} onClick={resetDemo}>Reset</button></>}<details className="connection-panel"><summary aria-label="Connection settings" title="Connection settings"><Icon name="settings" /></summary><form className="connection" onSubmit={event => { event.preventDefault(); if (!base) connect(); }}><label>Monitoring server address<input value={address} onChange={e => setAddress(e.target.value)} disabled={!!base} /></label><button type="button" onClick={base ? () => { setBase(null); setError(''); } : connect}>{base ? 'Return to preview' : 'Connect to monitoring'}</button><div className="source-settings">          <details className="observation-help"><summary>Observation details & privacy</summary><dl className="facts"><div><dt>Person</dt><dd>{person}</dd></div><div><dt>Face detection</dt><dd>{face}</dd></div><div><dt>Source health</dt><dd>{!base ? 'Preview only' : !feed.connected ? 'Disconnected' : status?.online === false ? 'Offline' : status?.online ? 'Online' : 'Unknown'}</dd></div><div><dt>Posture</dt><dd>{!base ? preview === 'offline' ? 'Unknown' : preview === 'alert' ? 'Floor-level example' : 'Standing example' : offline ? 'Unknown' : status && 'posture' in status ? status.posture : 'Not reported'}</dd></div></dl><p className="small">Coordinates describe the camera image, not a floor map. The preview furniture is decorative. No camera video is displayed here. Dashed joints have unknown confidence.</p></details>{base ? <div className="source-tabs" role="group" aria-label="Observation source type">{(['edge','simulation','replay'] as const).map(value => <button type="button" key={value} className={mode === value ? 'selected' : ''} aria-pressed={mode === value} onClick={() => { setMode(value); setSelectedSource(''); }}>{sourceLabels[value]}</button>)}</div> : <div className="source-tabs" role="group" aria-label="UI preview examples">{(['standing','alert','offline'] as const).map(value => <button type="button" key={value} className={preview === value ? 'selected' : ''} aria-pressed={preview === value} onClick={() => choosePreview(value)}>{value === 'standing' ? 'Standing example' : value === 'alert' ? 'Possible-fall example' : 'Offline example'}</button>)}</div>}</div><p className="small">Use the server address provided during setup.</p></form></details>      <details className="demo-tools"><summary aria-label="Scenario controls and evaluation" title="Scenario controls and evaluation"><Icon name="play" /></summary><div className="lower-layout"><SimulatorPanel base={base && mode === 'simulation' ? base : null} /></div></details></div>
    </header>

    {(error || feed.error) && <p role="alert" className="error">{error || feed.error}</p>}
    <div id="main-content">
    <div className="resident-mount" hidden={view !== 'resident'}><ResidentPanel key={base ?? 'preview'} base={base} connected={feed.connected} alerts={feed.state.alerts} /></div>
    {view === 'caregiver' && <>
      <div className="monitor-grid">
        <section className="card main-feed"><div className="section-heading"><h2>Main feed · {sourceLabel}</h2><span className="feed-mode">{base ? (mode === 'simulation' ? simulation.client?.connected : feed.connected) ? 'Connected' : 'Disconnected' : 'Example data'}</span></div>
          {base && keys.length > 1 && <label>Observation source<select value={activeSource} onChange={e => setSelectedSource(e.target.value)}>{keys.map(key => <option key={key}>{key}</option>)}</select></label>}
          <div className="room-stage">{base && mode === 'simulation'
            ? <LiveSimulationView simulation={simulation} alertPersonIds={alerts.flatMap(alert => 'person_id' in alert && typeof alert.person_id === 'string' ? [alert.person_id] : [])} />
            : <RoomView observations={observations} illustrated={!base} />}</div>
          {base && mode === 'simulation' && <div className="simulation-controls"><button disabled={!simulation.ready || simulation.busy} onClick={() => simulation.command('toggle')}>{simulation.state === 'playing' ? 'Pause' : 'Play'}</button><span className="small">Simulated world view · monitoring panels show camera observations</span></div>}


        </section>
        <aside className="monitor-side">
          <section className="card skeleton-card"><div className="section-heading"><h2>Skeleton feed</h2><span className="small">pose estimation</span></div><RoomView observations={observations} skeleton /></section>
          <section className="card timeline-card">          <details className="alerts-card" id="alerts"><summary>Incident alerts · {incidentCount}</summary><div className="section-heading"><h2>Incident alerts</h2><span className="count-chip">{incidentCount}</span></div><div aria-live="polite">
            {!base && preview === 'alert' ? <div className="alert"><span className="alert-label">Example incident</span><h3>Possible fall</h3><p>Example posture and alert for interface testing.</p><button disabled={acknowledged} onClick={() => { setAcknowledged(true); setExampleEvents(previous => [exampleEvent('alert_acknowledged'),...previous]); }}>{acknowledged ? 'Acknowledged in preview' : 'Acknowledge example'}</button></div> : base && alerts.length ? alerts.map(alert => <div className="alert" key={alert.alert_id}><span className="alert-label">{alert.status}</span><h3>{alert.kind.replaceAll('_',' ')}</h3><p>{sourceLabels[alert.source_type]} · {alert.source}</p><p className="small">Opened {new Date(alert.opened_at).toLocaleTimeString()}</p>{alert.recovered_at && <p>Recovery reported; review still needed.</p>}<button disabled={!feed.connected || alert.status !== 'open' || pending === alert.alert_id} onClick={() => acknowledge(alert.alert_id)}>{alert.status === 'acknowledged' ? 'Acknowledged' : pending === alert.alert_id ? 'Acknowledging…' : 'Acknowledge'}</button></div>) : <p className="empty">{offline ? 'Observations unavailable. Room safety is unknown.' : 'No active alerts in this view.'}</p>}
            <p className="small">Acknowledging means the incident was seen. It does not resolve it.</p>
          </div></details><div className="section-heading"><h2>Event log</h2><Icon name="clock" /></div>{!events.length ? <p className="empty">No events yet.</p> : <ol className="timeline">{events.slice(0,30).map(event => <li key={event.event_id}><time>{new Date(event.timestamp).toLocaleTimeString()}</time><strong>{event.event_type.replaceAll('_',' ')}</strong><span>{base ? `${sourceLabels[event.source_type]} · ${event.source}` : 'UI example'}</span></li>)}</ol>}</section>
        </aside>
      </div>
      <section className="posture-metrics" aria-label="Observation measurements">
        <div className="card"><h2 className="metric-heading"><span>Torso angle</span>{!base && <span className="metric-caption">red line = 60° alert · preview</span>}</h2><strong className="measurement">{features?.torso_angle_deg == null ? '—' : `${features.torso_angle_deg.toFixed(0)}°`}</strong><div className="bar"><i style={{ width: features?.torso_angle_deg == null ? '0%' : `${Math.min(100, features.torso_angle_deg / (!base ? 90 : 180) * 100)}%` }} />{!base && <b aria-label="60 degree preview alert marker" />}</div><FeatureSpark key={`${base}:${mode}:${activeSource}`} value={features?.torso_angle_deg ?? null} sampleId={sampleId} max={180} preview={!base} /><p className="small">{base ? 'Reported posture feature' : 'Example measurement'} · 0° is upright</p></div>
        <div className="card"><h2 className="metric-heading"><span>Hip position</span><span className="metric-caption">height from floor · unavailable</span></h2><strong className="measurement">{features?.hip_y == null ? '—' : features.hip_y.toFixed(2)}</strong><FeatureSpark key={`${base}:${mode}:${activeSource}`} value={features?.hip_y ?? null} sampleId={sampleId} max={1} preview={!base} /><p className="small">Image coordinate: 0 top, 1 bottom. Not centimeters.</p></div>
        <div className="card"><h2 className="metric-heading"><span>{base ? 'Incident age' : 'Time down'}</span><span className="metric-caption">{base ? 'since possible-fall alert' : 'since fall detected · preview'}</span></h2><strong className="measurement">{displayAge == null || !Number.isFinite(displayAge) ? '—' : `${String(Math.floor(displayAge / 60)).padStart(2, '0')}:${String(displayAge % 60).padStart(2, '0')}`}</strong><p className="small">{!base ? preview === 'offline' ? 'Unknown · preview observations unavailable.' : preview === 'alert' ? 'Possible fall · example timer.' : 'No fall detected · preview only.' : fall ? fall.source_type === 'edge' ? 'Since this live possible-fall alert opened; not time on the floor.' : 'Not shown for simulation/replay clocks.' : 'No connected possible-fall incident.'}</p></div>
        <AppointmentPanel compact preview={!base} />
      </section>


    </>}
    </div>
  </main>;
}

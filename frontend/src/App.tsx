import { useState } from 'react';
import type { Event } from './generated/contract';
import { exampleEvent, exampleObservation, type Preview } from './demoData';
import { sourceKey, visibleObservations, type SourceMode } from './feedState';
import { acknowledgeAlert } from './api';
import { useRoomFeed } from './useRoomFeed';
import RoomView from './RoomView';
import SimulatorPanel from './SimulatorPanel';
import Icon from './Icon';

const sourceLabels = { edge: 'Live camera', simulation: 'Simulation', replay: 'Recorded replay', manual: 'Manual' };
export default function App() {
  const [example] = useState(exampleObservation);
  const [preview, setPreview] = useState<Preview>('standing');
  const [exampleEvents, setExampleEvents] = useState<Event[]>([]);
  const [acknowledged, setAcknowledged] = useState(false);
  const [address, setAddress] = useState('http://127.0.0.1:8000');
  const [base, setBase] = useState<string | null>(null);
  const [mode, setMode] = useState<SourceMode>('edge');
  const [selectedSource, setSelectedSource] = useState('');
  const [error, setError] = useState('');
  const [pending, setPending] = useState<string | null>(null);
  const feed = useRoomFeed(base, mode);
  const allObservations = base ? visibleObservations(feed.state, feed.now, feed.connected).filter(item => item.source_type === mode) : preview === 'offline' ? [] : [example];
  const keys = [...new Set([...Object.keys(feed.state.sources), ...Object.values(feed.state.observations).map(item => sourceKey(item.data))])].filter(key => key.startsWith(`${mode}:`));
  const activeSource = keys.includes(selectedSource) ? selectedSource : keys[0];
  const observations = base ? allObservations.filter(item => sourceKey(item) === activeSource) : allObservations;
  const status = feed.state.sources[activeSource];
  const offline = !observations.length;
  const events = base ? feed.state.events.filter(event => event.source_type === mode) : exampleEvents;
  const alerts = Object.values(feed.state.alerts).filter(alert => alert.source_type === mode && alert.status !== 'resolved');
  function choosePreview(next: Preview) {
    setPreview(next); setAcknowledged(false);
    setExampleEvents([exampleEvent(next === 'alert' ? 'possible_fall' : next === 'offline' ? 'source_offline' : 'person_visible')]);
  }
  function connect() {
    try {
      const url = new URL(address);
      if (!['http:','https:'].includes(url.protocol) || url.username || url.password) throw new Error();
      setBase(url.origin); setError('');
    } catch { setError('Enter an HTTP or HTTPS backend address, such as http://127.0.0.1:8000.'); }
  }
  async function acknowledge(id: string) {
    if (!base) return;
    setPending(id); setError('');
    try { await acknowledgeAlert(base,id); } catch (e) { setError((e as Error).message); }
    finally { setPending(null); }
  }
  const face = offline ? 'Unknown' : observations.some(o => o.face_present === true) ? 'Visible' : observations.some(o => o.face_present === false) ? 'Not visible' : 'Not run';
  const incidentCount = base ? alerts.length : preview === 'alert' ? 1 : 0;
  const lastArrival = Math.max(0, ...Object.values(feed.state.observations).filter(item => sourceKey(item.data) === activeSource).map(item => item.receivedAt));
  const ageSeconds = lastArrival ? Math.max(0, Math.floor((feed.now - lastArrival) / 1000)) : null;
  const observationSummary = !base ? 'Example data' : offline ? 'Unknown' : 'Receiving';
  return <div className="app-shell">
    <a className="skip-link" href="#overview">Skip to dashboard</a>
    <aside className="sidebar">
      <a className="brand" href="#overview"><span className="brand-mark"><Icon name="home" /></span><span>Hearth<span className="brand-subtitle">Care monitoring</span></span></a>
      <p className="nav-label">Workspace</p>
      <nav aria-label="Dashboard sections"><a href="#overview"><Icon name="room" /> Room overview</a><a href="#alerts"><Icon name="alert" /> Incident alerts<span className="nav-count">{incidentCount}</span></a><a href="#simulation"><Icon name="play" /> Scenarios</a><a href="#evaluation"><Icon name="chart" /> Evaluation</a></nav>
      <div className="sidebar-note"><span className="tiny-dot" /> One-room monitoring<p>Possible safety events for human review.</p></div>
    </aside>
    <main id="overview">
    <header className="page-header"><div><p className="eyebrow">Hearth · Room 01</p><h1>Room overview</h1><p>Review observations and possible safety events in one room.</p></div><span className="badge"><span className="tiny-dot" />{base ? sourceLabels[mode] : 'Preview · example data'}</span></header>
    <div className="workspace-toolbar">
      {base ? <div className="source-tabs" role="group" aria-label="Observation source type">{(['edge','simulation','replay'] as const).map(value => <button key={value} className={mode === value ? 'selected' : 'secondary'} aria-pressed={mode === value} onClick={() => { setMode(value); setSelectedSource(''); }}>{sourceLabels[value]}</button>)}</div> : <div className="source-tabs" role="group" aria-label="UI preview examples">{(['standing','alert','offline'] as const).map(value => <button key={value} className={preview === value ? 'selected' : 'secondary'} aria-pressed={preview === value} onClick={() => choosePreview(value)}>{value === 'standing' ? 'Person visible' : value === 'alert' ? 'Possible fall' : 'Source offline'}</button>)}</div>}
      <details className="connection-panel"><summary><Icon name="settings" />Connection settings</summary><form className="connection" onSubmit={event => { event.preventDefault(); if (!base) connect(); }}><label>Monitoring server address<input value={address} onChange={e => setAddress(e.target.value)} disabled={!!base} placeholder="http://127.0.0.1:8000" /></label><button type={base ? 'button' : 'submit'} onClick={base ? () => { setBase(null); setError(''); } : undefined}>{base ? 'Return to preview' : 'Connect to monitoring'}</button><p className="small">Use the server address provided during setup.</p></form></details>
    </div>
    <aside className={`notice ${base && !feed.connected ? 'notice-warning' : ''}`}><Icon name={base && !feed.connected ? 'alert' : 'signal'} /><span>{base ? `${feed.connected ? 'Monitoring connected' : 'Monitoring disconnected · reconnecting'}. Viewing ${sourceLabels[mode].toLowerCase()} observations only.` : 'This is a preview with example data. Open Connection settings to connect to monitoring.'}</span></aside>
    {(error || feed.error) && <p role="alert" className="error">{error || feed.error}</p>}
    <section className="overview-stats" aria-label="Monitoring summary">
      <div><span className="stat-icon"><Icon name="signal" /></span><div><p>Source connection</p><strong>{!base ? 'Preview mode' : !feed.connected ? 'Disconnected' : status?.online === false ? 'Offline' : status?.online ? 'Online' : 'Awaiting source'}</strong></div></div>
      <div><span className="stat-icon"><Icon name="room" /></span><div><p>Observations</p><strong>{observationSummary}</strong></div></div>
      <div><span className={`stat-icon ${incidentCount ? 'attention-icon' : ''}`}><Icon name="alert" /></span><div><p>{base ? 'Active alerts' : 'Example alerts'}</p><strong>{incidentCount}<span className="stat-caption">{incidentCount ? 'for review' : 'in this view'}</span></strong></div></div>
    </section>
    <div className="layout">
      <section className="card room"><div className="section-heading"><div><p className="eyebrow">Room observation</p><h2>Room 01</h2></div><span className={offline ? 'status unknown' : 'status'}><span className="tiny-dot" />{offline ? 'Unknown' : base ? 'Receiving observations' : 'Example source'}</span></div>
        {base && keys.length > 1 && <label>Observation source<select value={activeSource} onChange={e => setSelectedSource(e.target.value)}>{keys.map(key => <option key={key}>{key}</option>)}</select></label>}
        <div className="room-stage"><RoomView observations={observations} /><div className="room-stage-footer"><span><span className="tiny-dot" />{base ? sourceLabels[mode] : 'Example observation'}</span><span>{base ? ageSeconds == null ? 'Waiting for first observation' : `${ageSeconds === 0 ? 'Just now' : `${ageSeconds}s since receipt`}` : 'Synthetic skeleton'}</span></div></div>
        <dl className="facts"><div><dt>Person</dt><dd>{offline ? 'Unknown' : observations.some(o => o.person_detected) ? 'Visible' : 'Not visible'}</dd></div><div><dt>Face detection</dt><dd>{face}</dd></div><div><dt>Source health</dt><dd>{!base ? 'Preview only' : !feed.connected ? 'Disconnected' : status?.online === false ? 'Offline' : status?.online ? 'Online' : 'Unknown'}</dd></div><div><dt>Observed posture</dt><dd>{offline ? 'Unknown' : status && 'posture' in status ? status.posture : 'Not reported'}</dd></div></dl>
        <details className="observation-help"><summary>About this view</summary><p className="small">Skeleton coordinates describe the camera frame, not measured floor positions. Dashed joints indicate unknown confidence. Missing observations mean unknown room safety.</p></details>
      </section>
      <section className="card alerts-card" id="alerts"><div className="section-heading"><div><p className="eyebrow">For review</p><h2>Incident alerts</h2></div><span className="count-chip">{incidentCount}</span></div><div aria-live="polite">
        {!base && preview === 'alert' ? <div className="alert"><span className="alert-label"><Icon name="alert" />Example incident</span><h3>Possible fall</h3><p>Review this example incident. The room view remains a sample standing pose.</p><button disabled={acknowledged} onClick={() => { setAcknowledged(true); setExampleEvents(previous => [exampleEvent('alert_acknowledged'),...previous]); }}>{acknowledged ? 'Acknowledged in preview' : 'Acknowledge example'}</button></div> : base && alerts.length ? alerts.map(alert => <div className="alert" key={alert.alert_id}><span className="alert-label"><Icon name="alert" />{alert.status}</span><h3>{alert.kind.replaceAll('_',' ')}</h3><p>{sourceLabels[alert.source_type]} · {alert.source}</p><p className="small">Opened {new Date(alert.opened_at).toLocaleTimeString()}</p>{alert.recovered_at && <p>Recovery reported; incident still requires review.</p>}<button disabled={!feed.connected || alert.status !== 'open' || pending === alert.alert_id} onClick={() => acknowledge(alert.alert_id)}>{alert.status === 'acknowledged' ? 'Acknowledged' : pending === alert.alert_id ? 'Acknowledging…' : 'Acknowledge'}</button></div>) : <div className={`empty-state ${offline ? 'uncertain-state' : ''}`}><span className="empty-icon"><Icon name={offline ? 'signal' : 'check'} /></span><h3>{offline ? 'Visibility unavailable' : 'No active alerts'}</h3><p>{offline ? 'Observations are missing. Room safety is unknown.' : !base ? 'This example has no incident selected. Try “Possible fall” above.' : 'No active alerts have been reported for this source type.'}</p></div>}
        <p className="small">Acknowledgement does not resolve an incident. Connected alert status comes from the backend.</p>
      </div></section>
      <section className="card timeline-card"><div className="section-heading"><div><p className="eyebrow">Recent activity</p><h2>Event timeline</h2></div><Icon name="clock" /></div>{!events.length ? <div className="empty-state compact"><span className="empty-icon"><Icon name="clock" /></span><h3>No events yet</h3><p>{base ? 'New monitoring events will appear as they arrive.' : 'Select a preview example to see an event.'}</p></div> : <ol className="timeline">{events.slice(0,30).map(event => <li key={event.event_id} className={event.event_type === 'possible_fall' || event.event_type === 'source_offline' ? 'timeline-warning' : ''}><strong>{event.event_type.replaceAll('_',' ')}</strong><span>{new Date(event.timestamp).toLocaleTimeString()} · {base ? `${sourceLabels[event.source_type]} · ${event.source}` : 'UI example'}</span></li>)}</ol>}</section>
      <SimulatorPanel base={base && mode === 'simulation' ? base : null} />
    </div><footer><span>Hearth · Possible safety events for human review.</span><span>{base ? sourceLabels[mode] : 'Example data only'}</span></footer>
  </main></div>;
}

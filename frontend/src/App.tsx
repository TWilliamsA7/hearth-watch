import { useState } from 'react';
import type { Event } from './generated/contract';
import { exampleEvent, exampleObservation, type Preview } from './demoData';
import RoomView from './RoomView';

export default function App() {
  const [observation] = useState(exampleObservation);
  const [preview, setPreview] = useState<Preview>('standing');
  const [events, setEvents] = useState<Event[]>([]);
  const [acknowledged, setAcknowledged] = useState(false);
  const offline = preview === 'offline';
  function choosePreview(next: Preview) {
    setPreview(next);
    setAcknowledged(false);
    setEvents([exampleEvent(next === 'alert' ? 'possible_fall' : next === 'offline' ? 'source_offline' : 'person_visible')]);
  }
  function acknowledge() {
    setAcknowledged(true);
    setEvents(previous => [exampleEvent('alert_acknowledged'), ...previous]);
  }

  return <main>
    <header><div><p className="eyebrow">HEARTH / CARE MONITORING</p><h1>A clearer view of the room.</h1><p>One room. Observable activity. Events for human review.</p></div><span className="badge">Simulated · UI preview</span></header>
    <aside className="notice">Example data only. This preview is not connected to the camera, monitoring backend, or simulator.</aside>
    <div className="layout">
      <section className="card room"><div className="section-heading"><h2>Room 01</h2><span className={offline ? 'status unknown' : 'status'}>{offline ? 'Offline example · unknown' : 'Example source'}</span></div>
        <RoomView observation={observation} offline={offline} />
        <dl className="facts"><div><dt>Person</dt><dd>{offline ? 'Unknown' : 'Visible · example'}</dd></div><div><dt>Face detection</dt><dd>{offline ? 'Unknown' : 'Not run'}</dd></div><div><dt>Connection</dt><dd>Backend not connected</dd></div></dl>
      </section>
      <section className="card"><p className="eyebrow">REQUIRES ATTENTION</p><h2>Incident alerts</h2>
        <div aria-live="polite">{preview === 'alert' ? <div className="alert"><h3>Possible fall · example</h3><p>Example monitoring event for human review. The skeleton remains an example standing observation.</p><button onClick={acknowledge} disabled={acknowledged}>{acknowledged ? 'Acknowledged in preview' : 'Acknowledge example'}</button><p className="small">Acknowledgement does not resolve the incident.</p></div> : <p className="empty">{offline ? 'Observation unavailable. Room safety is unknown.' : 'No example alert selected.'}</p>}</div>
      </section>
      <section className="card"><h2>Explore the interface</h2><p>Choose an example to check how the dashboard responds.</p><div className="buttons">{(['standing', 'alert', 'offline'] as const).map(value => <button key={value} className={preview === value ? 'selected' : 'secondary'} aria-pressed={preview === value} onClick={() => choosePreview(value)}>{value === 'standing' ? 'Person visible' : value === 'alert' ? 'Possible fall alert' : 'Source offline'}</button>)}</div><p className="small">Simulator play, pause, reset, and replay controls will be connected once its API is available.</p></section>
      <section className="card"><h2>Event timeline</h2>{events.length === 0 ? <p className="empty">Select an example to add an event.</p> : <ol className="timeline">{events.map(event => <li key={event.event_id}><strong>{event.event_type.replaceAll('_', ' ')}</strong><span>{new Date(event.timestamp).toLocaleTimeString()} · UI example</span></li>)}</ol>}</section>
      <section className="card evaluation"><h2>Scenario evaluation</h2><p>Awaiting measured results from the evaluator.</p><div className="metrics">{['Detection delay', 'False alarms', 'Missed incidents', 'Test count'].map(label => <div key={label}><span>{label}</span><strong>—</strong></div>)}</div><p className="small">No accuracy or timing results have been measured by this preview.</p></section>
    </div><footer>Hearth · Possible safety events for human review.</footer>
  </main>;
}

import LiveCameraView from './LiveCameraView';
import CareResponsePanel from './CareResponsePanel';
import { useNurseSession } from './useNurseSession';
import NurseControls from './NurseControls';
import { useAppointments } from './useAppointments';
import { useEffect, useState } from 'react';
import { sourceKey, visibleObservations, type SourceMode } from './feedState';
import { INCIDENTS, SPEEDS, useLiveSimulation, type IncidentKind } from './useLiveSimulation';
import LiveSimulationView from './LiveSimulationView';
import RoomSettingsPanel from './RoomSettingsPanel';
import { useRoomFeed } from './useRoomFeed';
import RoomView from './RoomView';
import SimulatorPanel from './SimulatorPanel';
import ResidentPanel from './ResidentPanel';
import Icon from './Icon';
import AppointmentPanel from './AppointmentPanel';
import FeatureSpark from './FeatureSpark';
import IncidentList from './IncidentList';
import SourceHealth from './SourceHealth';
import { useRooms } from './useRooms';
import { DEFAULT_ROOM } from './api';
import { kindLabel, postureText, sourceTypeLabels } from './alertText';

// The Hearth server serves this page, so the API, WebSockets and Sim Studio share its origin.
const base = window.location.origin;
const sourceLabels = { edge: 'Live camera', simulation: 'Simulation', replay: 'Recorded replay' };
type View = 'caregiver' | 'resident';
const initialView = (): View => window.location.pathname === '/room' || window.location.hash === '#recognition' ? 'resident' : 'caregiver';

export default function App() {
  const [view, setView] = useState<View>(initialView);
  const [mode, setMode] = useState<SourceMode>('simulation');
  const [selectedSource, setSelectedSource] = useState('');
  const [incident, setIncident] = useState<IncidentKind>('fall');
  const rooms = useRooms(base);
  const [chosenRoom, setRoom] = useState(DEFAULT_ROOM);
  const room = rooms.room_ids.includes(chosenRoom) ? chosenRoom : rooms.room_ids[0];
  const feed = useRoomFeed(base, mode, room);
  const appointments = useAppointments(base);
  const nurseSession = useNurseSession(base);
  const simulation = useLiveSimulation(mode === 'simulation' ? base : null, room);
  // Keep the address in step with the view, so a reload or a shared link opens the same one.
  useEffect(() => {
    const path = view === 'resident' ? '/room' : '/';
    if (window.location.pathname !== path) window.history.replaceState(null, '', path + window.location.hash);
  }, [view]);
  useEffect(() => {
    const onHash = () => { if (window.location.hash === '#recognition') setView('resident'); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const allObservations = visibleObservations(feed.state, feed.now, feed.connected).filter(item => item.source_type === mode);
  const keys = [...new Set([...Object.keys(feed.state.sources), ...Object.values(feed.state.observations).map(item => sourceKey(item.data))])].filter(key => key.startsWith(`${mode}:`));
  const activeSource = keys.includes(selectedSource) ? selectedSource : keys[0];
  const observations = allObservations.filter(item => sourceKey(item) === activeSource);
  const status = feed.state.sources[activeSource];
  const offline = !observations.length;
  // With a nurse session, the nurse's coverage; before one, what this room's feed reports.
  const events = nurseSession.nurse ? nurseSession.events : feed.state.events;
  const alerts = Object.values(nurseSession.nurse ? nurseSession.alerts : feed.state.alerts).filter(alert => alert.status !== 'resolved');
  const healthSources = nurseSession.nurse ? Object.values(nurseSession.sources) : Object.values(feed.state.sources);
  const person = offline ? 'Unknown' : observations.some(o => o.person_detected) ? 'Visible' : 'Not visible';
  const face = offline ? 'Unknown' : observations.some(o => o.face_present === true) ? 'Visible' : observations.some(o => o.face_present === false) ? 'Not visible' : 'Not run';
  const features = observations.find(o => o.person_detected)?.features;
  const simulating = mode === 'simulation';
  const simulationIdle = !simulating || !simulation.ready || simulation.busy;
  const roomStatus = offline ? 'Observation unknown' : alerts.length ? `${alerts.length} active alert${alerts.length === 1 ? '' : 's'}` : 'Observations available';
  const sampleId = observations.find(o => o.person_detected)?.obs_id ?? 'unknown';
  return <main className="dashboard" id="overview">
    <a className="skip-link" href="#main-content">Skip to dashboard</a>
    <header className="page-header">
      <a className="brand" href="#overview"><span><h1 aria-label="HearthWatch">HearthWatc<span className="wordmark-tail" aria-hidden="true">h<svg className="heart-flourish" viewBox="0 0 36 36" fill="none"><path d="M 1 27 C 6 31 10 29 14 25 C 19 21 29 16 29 10 C 29 4 22 3 18 9 C 14 3 7 4 7 10 C 7 16 14 20 18 23" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" /></svg></span></h1></span></a>
      <div className="header-actions">
        <div className="view-toggle" role="group" aria-label="Dashboard view"><button aria-pressed={view === 'caregiver'} className={view === 'caregiver' ? 'selected' : ''} onClick={() => setView('caregiver')}>Caregiver</button><button aria-pressed={view === 'resident'} className={view === 'resident' ? 'selected' : ''} onClick={() => setView('resident')}>Patient</button></div>
        {rooms.room_ids.length > 1 && <label className="room-picker">Room<select value={room} onChange={e => { setRoom(e.target.value); setSelectedSource(''); }}>{rooms.room_ids.map(id => <option key={id} value={id}>{id}</option>)}</select></label>}
        <span className={`status ${offline ? 'unknown' : alerts.length ? 'warning' : ''}`}><span className="tiny-dot" />{roomStatus}</span>
        {view === 'caregiver' && <>
          <select aria-label="Incident to stage" value={incident} onChange={e => setIncident(e.target.value as IncidentKind)} disabled={!simulating}>{Object.entries(INCIDENTS).map(([kind, item]) => <option key={kind} value={kind}>{item.label}</option>)}</select>
          <button className="go" disabled={simulationIdle} title={simulating ? (simulation.selected ? `Stage it for ${simulation.selected}` : 'Click a person in the 3D room to choose who; otherwise the simulator picks someone suitable') : 'Choose Simulation as the main feed source'} onClick={() => void simulation.command(incident)}>Simulate {INCIDENTS[incident].label.toLowerCase()}</button>
          <button disabled={simulationIdle} onClick={() => void simulation.command('reset')}>Reset</button>
        </>}
        <details className="connection-panel"><summary aria-label="Observation details" title="Observation details"><Icon name="settings" /></summary><div className="connection"><div className="source-settings">
          <details className="observation-help" open><summary>Observation details & privacy</summary><dl className="facts"><div><dt>Person</dt><dd>{person}</dd></div><div><dt>Face detection</dt><dd>{face}</dd></div><div><dt>Source health</dt><dd>{!feed.connected ? 'Disconnected' : status?.online === false ? 'Offline' : status?.online ? 'Online' : 'Unknown'}</dd></div><div><dt>Posture</dt><dd>{offline || !status ? 'Unknown' : postureText(status)}</dd></div></dl><p className="small">Coordinates describe the camera image, not a floor map. Live camera shows the annotated feed when the camera client shares it. Dashed joints have unknown confidence.</p></details>
          <div className="source-tabs" role="group" aria-label="Observation source type">{(['edge','simulation','replay'] as const).map(value => <button type="button" key={value} className={mode === value ? 'selected' : ''} aria-pressed={mode === value} onClick={() => { setMode(value); setSelectedSource(''); }}>{sourceLabels[value]}</button>)}</div>
        </div></div></details>
        <details className="demo-tools"><summary aria-label="Scenario controls and evaluation" title="Scenario controls and evaluation"><Icon name="play" /></summary><div className="lower-layout"><SimulatorPanel base={simulating ? base : null} /></div></details>
        <details className="connection-panel developer-menu"><summary aria-label="Developer tools" title="Developer tools"><Icon name="signal" /></summary><div className="connection"><h3>Developer tools</h3><ul className="developer-links">
          <li><a href="/studio/" target="_blank" rel="noopener">Sim Studio</a><span className="small">Inspect residents, overlays, live-camera calibration</span></li>
          <li><a href="/studio/runs" target="_blank" rel="noopener">Recorded run player</a><span className="small">Replay saved sim runs against ground truth</span></li>
          <li><a href="/docs" target="_blank" rel="noopener">API reference</a><span className="small">Every monitoring, simulation and evaluation route</span></li>
        </ul></div></details>
      </div>
    </header>

    {(feed.error || simulation.error) && <p role="alert" className="error">{feed.error || simulation.error}</p>}
    <div id="main-content">
    <div className="resident-mount" hidden={view !== 'resident'}><ResidentPanel key={room} base={base} room={room} connected={feed.connected} alerts={feed.state.alerts} sources={Object.values(feed.state.sources)} gestureHoldSeconds={rooms.gesture_hold_seconds} appointments={appointments} /></div>
    {view === 'caregiver' && <>
      <div className="monitor-grid">
        <section className="card main-feed"><div className="section-heading"><h2>Main feed · {room} · {sourceLabels[mode]}</h2><span className="feed-mode">{(simulating ? simulation.client?.connected : feed.connected) ? 'Connected' : 'Disconnected'}</span></div>
          <div className="source-tabs" role="group" aria-label="Main feed source">{(['simulation','edge','replay'] as const).map(value => <button key={value} className={mode === value ? 'selected' : ''} aria-pressed={mode === value} onClick={() => {setMode(value); setSelectedSource('');}}>{sourceLabels[value]}</button>)}</div>
          {keys.length > 1 && <label>Observation source<select value={activeSource} onChange={e => setSelectedSource(e.target.value)}>{keys.map(key => <option key={key}>{key}</option>)}</select></label>}
          <div className="room-stage">{simulating
            ? <LiveSimulationView simulation={simulation} alertPersonIds={alerts.flatMap(alert => typeof alert.metadata.person_id === 'string' ? [alert.metadata.person_id] : [])} />
            : mode === 'edge' ? <LiveCameraView base={base} room={room} source={activeSource?.replace(/^edge:/,'') || 'fall-model-01'} />
            : <>{!keys.length && <div className="simulation-message" role="status"><p>No recorded replay is playing for {room}. Start one with <code>scripts/start.ps1 -Replay &lt;recording.jsonl&gt;</code>.</p></div>}<RoomView observations={observations} /></>}</div>
          {simulating && <div className="simulation-controls">
            <button disabled={!simulation.ready || simulation.busy} onClick={() => void simulation.command('toggle')}>{simulation.state === 'playing' ? 'Pause' : 'Play'}</button>
            <label>Speed<select value={simulation.speed} disabled={!simulation.ready || simulation.busy} onChange={e => void simulation.setSpeed(Number(e.target.value))}>{SPEEDS.map(value => <option key={value} value={value}>{value}×</option>)}</select></label>
            {simulation.selected && <button disabled={simulation.busy} onClick={() => void simulation.command('reset-person')}>Reset {simulation.selected}</button>}
            <RoomSettingsPanel simulation={simulation} />
            <span className="small">{simulation.room_id && simulation.room_id !== room ? `This simulated room reports as ${simulation.room_id}. Generate a room to simulate ${room}.` : 'Simulated world view · monitoring panels show camera observations'}</span>
          </div>}
        </section>
        <aside className="monitor-side">
          <section className="card skeleton-card"><div className="section-heading"><h2>Skeleton feed</h2><span className="small">pose estimation</span></div><RoomView observations={observations} skeleton /></section>
          <section className="card timeline-card">
            <NurseControls session={nurseSession} rooms={rooms.room_ids} />
            <IncidentList session={nurseSession} roomAlerts={feed.state.alerts} room={room} />
            <SourceHealth sources={healthSources} scope={nurseSession.nurse ? 'in your coverage' : `for ${room} · ${sourceLabels[mode]}`} />
            <div className="section-heading"><h2>Event log</h2><Icon name="clock" /></div>
            {!events.length ? <p className="empty">No events yet.</p> : <ol className="timeline">{events.slice(0,30).map(event => <li key={event.event_id}><time>{new Date(event.timestamp).toLocaleTimeString()}</time><strong>{kindLabel(event.event_type)}</strong><span>{event.room_id} · {sourceTypeLabels[event.source_type] ?? event.source_type} · {event.source}{typeof event.metadata?.nurse_name === 'string' ? ` · ${event.metadata.nurse_name}` : ''}</span></li>)}</ol>}
          </section>
        </aside>
      </div>
      <section className="posture-metrics" aria-label="Observation measurements">
        <div className="card"><h2 className="metric-heading"><span>Torso angle</span></h2><strong className="measurement">{features?.torso_angle_deg == null ? '—' : `${features.torso_angle_deg.toFixed(0)}°`}</strong><div className="bar"><i style={{ width: features?.torso_angle_deg == null ? '0%' : `${Math.min(100, features.torso_angle_deg / 180 * 100)}%` }} /></div><FeatureSpark key={`${mode}:${room}:${activeSource}`} value={features?.torso_angle_deg ?? null} sampleId={sampleId} max={180} preview={false} /><p className="small">Reported posture feature · 0° is upright</p></div>
        <div className="card"><h2 className="metric-heading"><span>Hip position</span><span className="metric-caption">image coordinate</span></h2><strong className="measurement">{features?.hip_y == null ? '—' : features.hip_y.toFixed(2)}</strong><FeatureSpark key={`${mode}:${room}:${activeSource}`} value={features?.hip_y ?? null} sampleId={sampleId} max={1} preview={false} /><p className="small">0 is the top of the image, 1 the bottom. Not a height in centimeters.</p></div>
        <CareResponsePanel session={nurseSession} />
        <AppointmentPanel compact scheduling feed={appointments} alerts={feed.state.alerts} room={room} />
      </section>
    </>}
    </div>
  </main>;
}

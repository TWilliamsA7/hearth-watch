import { useState } from 'react';
import { latestSources, sourceKey, visibleObservations, type SourceMode } from './feedState';
import { INCIDENTS, SPEEDS, type IncidentKind, type useLiveSimulation } from './useLiveSimulation';
import type { useRoomFeed } from './useRoomFeed';
import { useLiveLink } from './useLiveLink';
import LiveSimulationView from './LiveSimulationView';
import LiveCameraView from './LiveCameraView';
import RoomSettingsPanel from './RoomSettingsPanel';
import LiveLinkPanel from './LiveLinkPanel';
import RoomView from './RoomView';
import PersonInspector from './PersonInspector';
import AlertList from './AlertList';
import EventLog from './EventLog';
import CameraOverlay from './CameraOverlay';
import SourceHealth from './SourceHealth';
import Popover from './Popover';
import { peopleInView } from './people';
import { liveNames, useIdentities } from '@hearth/room-view';

export const sourceLabels: Record<SourceMode, string> = { simulation: 'Simulation', edge: 'Live camera', replay: 'Recorded replay' };

export default function MonitorView({ base, room, mode, setMode, feed, simulation }: {
  base: string; room: string; mode: SourceMode; setMode: (mode: SourceMode) => void;
  feed: ReturnType<typeof useRoomFeed>; simulation: ReturnType<typeof useLiveSimulation>;
}) {
  const [selectedSource, setSelectedSource] = useState('');
  const [selectedPerson, setSelectedPerson] = useState<string | null>(null);
  const [incident, setIncident] = useState<IncidentKind>('fall');
  const simulating = mode === 'simulation';
  const live = useLiveLink(simulating ? base : null);
  const identities = useIdentities(base, room);
  const observations = visibleObservations(feed.state, feed.now, feed.connected).filter(item => item.source_type === mode);
  const sources = latestSources(feed.state);
  const keys = [...new Set([...sources.map(item => sourceKey(item)), ...observations.map(item => sourceKey(item))])].filter(key => key.startsWith(`${mode}:`));
  const activeSource = keys.includes(selectedSource) ? selectedSource : keys[0];
  const shown = observations.filter(item => sourceKey(item) === activeSource);
  const alerts = Object.values(feed.state.alerts);
  const people = peopleInView(shown);
  const cameraSources = sources.filter(s => s.source_type === 'edge' || s.source_type === 'replay').map(s => s.source);
  const idle = !simulation.ready || simulation.busy;
  const choose = (next: SourceMode) => { setMode(next); setSelectedSource(''); setSelectedPerson(null); };
  const connected = simulating ? simulation.client?.connected : feed.connected;
  return <div className="monitor">
    <section className="card main-feed">
      <div className="feed-bar">
        <div className="source-tabs" role="tablist" aria-label="What the main view shows">{(['simulation', 'edge', 'replay'] as const).map(value =>
          <button key={value} role="tab" aria-selected={mode === value} className={mode === value ? 'selected' : ''} onClick={() => choose(value)}>{sourceLabels[value]}</button>)}</div>
        {keys.length > 1 && <select aria-label="Observation source" value={activeSource} onChange={e => { setSelectedSource(e.target.value); setSelectedPerson(null); }}>{keys.map(key => <option key={key} value={key}>{key.split(':')[1]}</option>)}</select>}
        <span className={`feed-mode ${connected ? '' : 'off'}`}>{room} · {connected ? 'Connected' : 'Disconnected'}</span>
      </div>
      <div className="room-stage">{simulating
        ? <LiveSimulationView simulation={simulation} liveNames={liveNames(live.link)}
            alertPersonIds={alerts.flatMap(alert => alert.status !== 'resolved' && typeof alert.metadata.person_id === 'string'
              // a linked camera's person appears in the room as live-<source>-<track>
              ? [alert.source === live.link?.source && live.link?.enabled ? `live-${alert.source}-${alert.metadata.person_id}` : alert.metadata.person_id] : [])} />
        : mode === 'edge' ? <LiveCameraView base={base} room={room} source={activeSource?.replace(/^edge:/, '') ?? ''} skeleton={<RoomView observations={shown} identities={identities} />} hasPeople={people.length > 0}
            overlay={activeSource?.includes('fall-model') ? undefined : (w, h) => <CameraOverlay width={w} height={h} observations={shown} alerts={alerts} identities={identities} />} />
        : <>{!keys.length && <div className="simulation-message" role="status"><p>No recorded replay is playing. Start one with <code>scripts/start.ps1 -Replay</code>.</p></div>}<RoomView observations={shown} identities={identities} /></>}</div>
      {simulating && <div className="simulation-controls">
        <button disabled={idle} onClick={() => void simulation.command('toggle')}>{simulation.state === 'playing' ? 'Pause' : 'Play'}</button>
        <label>Speed<select value={simulation.speed} disabled={idle} onChange={e => void simulation.setSpeed(Number(e.target.value))}>{SPEEDS.map(value => <option key={value} value={value}>{value}×</option>)}</select></label>
        <span className="control-group">
          <select aria-label="Incident to stage" value={incident} onChange={e => setIncident(e.target.value as IncidentKind)}>{Object.entries(INCIDENTS).map(([kind, item]) => <option key={kind} value={kind}>{item.label}</option>)}</select>
          <button className="go" disabled={idle} title={simulation.selected ? `Stage it for ${simulation.selected.toUpperCase()}` : 'Click a resident to choose who; otherwise the simulator picks someone suitable'} onClick={() => void simulation.command(incident)}>
            Stage {INCIDENTS[incident].label.toLowerCase()}{simulation.selected ? ` · ${simulation.selected.toUpperCase()}` : ''}
          </button>
        </span>
        {simulation.selected && <button disabled={idle} onClick={() => void simulation.command('reset-person')}>Reset {simulation.selected.toUpperCase()}</button>}
        <button disabled={idle} onClick={() => void simulation.command('reset')}>Reset room</button>
        <span className="spacer" />
        <Popover label={`Live people${live.link?.enabled ? ` · ${live.link.state}` : ''}`} title="Live people" align="up">{() => <LiveLinkPanel live={live} sources={cameraSources} />}</Popover>
        <RoomSettingsPanel simulation={simulation} />
      </div>}
    </section>
    <aside className="monitor-side">
      <div className="card">
        <PersonInspector base={base} mode={mode} simulation={simulation} link={live.link} people={people} identities={identities}
          selectedKey={selectedPerson} onSelectKey={setSelectedPerson} alerts={alerts}
          observationFor={personId => observations.find(item => 'person_id' in item && item.person_id === personId)} />
      </div>
      <div className="card alerts-card"><AlertList base={base} alerts={feed.state.alerts} connected={feed.connected} /></div>
      <div className="card log-card">
        <SourceHealth sources={sources} />
        <EventLog events={feed.state.events} notable={feed.state.notable} />
      </div>
    </aside>
  </div>;
}

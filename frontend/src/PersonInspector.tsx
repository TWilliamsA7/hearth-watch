import { useEffect, useState } from 'react';
import { isLivePerson, liveName, livePersonName, modelText, type ActorDetail, type LinkStatus } from '@hearth/room-view';
import type { Alert } from './generated/contract';
import type { RoomObservation } from './protocol';
import type { SourceMode } from './feedState';
import type { useLiveSimulation } from './useLiveSimulation';
import { request } from './api';
import { alertTitle } from './alertText';
import { personLabel, type TrackedPerson } from './people';

type Simulation = ReturnType<typeof useLiveSimulation>;
const pct = (value: number | null | undefined) => value == null ? '—' : `${Math.round(value * 100)}%`;
const num = (value: number | null | undefined, digits = 2) => value == null ? '—' : value.toFixed(digits);

// One person at a time: whoever was clicked in the room (simulation) or in the list (camera, replay).
export default function PersonInspector({ base, mode, simulation, link, people, selectedKey, onSelectKey, alerts, observationFor }: {
  base: string; mode: SourceMode; simulation: Simulation; link: LinkStatus | null;
  people: TrackedPerson[]; selectedKey: string | null; onSelectKey: (key: string | null) => void;
  alerts: Alert[]; observationFor: (personId: string) => RoomObservation | undefined;
}) {
  const openAlert = (personId: string | null | undefined) =>
    personId ? alerts.find(alert => alert.status !== 'resolved' && alert.metadata.person_id === personId) : undefined;

  if (mode === 'simulation') {
    const picked = simulation.picked;
    if (!picked) return <Empty text="Click a person in the room to inspect them." />;
    if (isLivePerson(picked)) {
      const me = link?.people.find(person => person.person_id === picked);
      return <section className="inspector" aria-label="Selected person">
        <Heading name={me ? livePersonName(me, link?.source) : liveName(picked, link?.source)} onClear={() => simulation.setSelected(null)} />
        <dl className="facts">
          <Fact term="Seen by" value={`${link?.source ?? 'live camera'} · a real person, not simulated`} />
          <Fact term="Identity" value={me?.name ? `Recognized: ${me.name}` : me?.recognized ? 'Recognized (not named: no consent)' : 'Not recognized'} />
          <Fact term="Monitor says" value={me ? `${me.label ?? '—'} (${me.posture ?? 'unknown'})` : '—'} />
          <Fact term="Camera's model" value={modelText(me?.model)} />
          <Fact term="Feed" value={me?.state ?? link?.state ?? '—'} />
          <Fact term="Alert" value={me?.alert_open ? 'Possible fall open' : '—'} warn={me?.alert_open} />
        </dl>
      </section>;
    }
    return <Resident base={base} simulation={simulation} personId={picked} observation={observationFor(picked)} alert={openAlert(picked)} />;
  }

  const person = people.find(item => item.key === selectedKey);
  return <section className="inspector" aria-label="People in view">
    <div className="section-heading"><h2>People in view · {people.length}</h2></div>
    {people.length ? <div className="people-list" role="listbox" aria-label="People in view">
      {people.map(item => <button key={item.key} role="option" aria-selected={item.key === selectedKey} className={item.key === selectedKey ? 'selected' : ''}
        onClick={() => onSelectKey(item.key === selectedKey ? null : item.key)}>{personLabel(item)}</button>)}
    </div> : <p className="small">Nobody is in view right now.</p>}
    {person ? <ObservationFacts observation={person.observation} alert={openAlert(person.personId)} />
      : people.length > 0 && <p className="small">Choose a person to see what the monitor sees.</p>}
  </section>;
}

function Resident({ base, simulation, personId, observation, alert }: {
  base: string; simulation: Simulation; personId: string; observation?: RoomObservation; alert?: Alert;
}) {
  const [detail, setDetail] = useState<ActorDetail | null>(null);
  const runId = simulation.client?.world?.run_id;
  useEffect(() => {
    setDetail(null);
    let alive = true;
    const poll = () => request<ActorDetail>(base, `/sim/live/actors/${encodeURIComponent(personId)}`)
      .then(value => { if (alive) setDetail(value); }).catch(() => { if (alive) setDetail(null); });
    poll();
    const timer = setInterval(poll, 600);
    return () => { alive = false; clearInterval(timer); };
  }, [base, personId, runId]);
  const now = simulation.client?.latest?.people.find(person => person.person_id === personId);
  return <section className="inspector" aria-label="Selected resident">
    <Heading name={personId.toUpperCase()} detail={detail?.info.archetype} onClear={() => simulation.setSelected(null)} />
    <dl className="facts">
      <Fact term="Doing" value={now?.activity ?? '—'} />
      <Fact term="Posture" value={now?.posture ?? '—'} />
      <Fact term="Camera sees" value={now ? `${Math.round(now.visible * 100)}% of joints` : '—'} />
      <Fact term="Incident" value={now?.incident ?? '—'} warn={!!now?.incident} />
      <Fact term="Next" value={detail?.plan.length ? detail.plan.slice(0, 4).join(' → ') : '—'} />
      <Fact term="Height · pace" value={detail ? `${detail.info.height.toFixed(2)} m · ${detail.info.speed.toFixed(2)} m/s` : '—'} />
    </dl>
    <h3>What the monitor sees</h3>
    {observation ? <ObservationFacts observation={observation} alert={alert} /> : <p className="small">Not in the camera's view right now.</p>}
  </section>;
}

function ObservationFacts({ observation, alert }: { observation: RoomObservation; alert?: Alert }) {
  const prediction = 'posture_prediction' in observation ? observation.posture_prediction : null;
  const features = observation.features;
  return <dl className="facts">
    <Fact term="Posture (model)" value={prediction ? `${prediction.label} · ${pct(prediction.confidence)}` : 'Pose rules only'} />
    <Fact term="Torso angle" value={features?.torso_angle_deg == null ? '—' : `${features.torso_angle_deg.toFixed(0)}° from upright`} />
    <Fact term="Head · hip height" value={`${num(features?.head_y)} · ${num(features?.hip_y)} (image, 0 = top)`} />
    <Fact term="Face" value={observation.face_present == null ? 'Not checked' : observation.face_present ? 'Visible' : 'Not visible'} />
    <Fact term="Alert" value={alert ? alertTitle(alert) : '—'} warn={!!alert} />
  </dl>;
}

function Heading({ name, detail, onClear }: { name: string; detail?: string; onClear: () => void }) {
  return <div className="section-heading"><h2>{name}{detail ? <span className="small"> · {detail}</span> : null}</h2><button className="link-button" onClick={onClear}>Clear</button></div>;
}
function Fact({ term, value, warn = false }: { term: string; value: string; warn?: boolean }) {
  return <div><dt>{term}</dt><dd className={warn ? 'warn' : ''}>{value}</dd></div>;
}
function Empty({ text }: { text: string }) {
  return <section className="inspector empty-inspector"><p className="small">{text}</p></section>;
}

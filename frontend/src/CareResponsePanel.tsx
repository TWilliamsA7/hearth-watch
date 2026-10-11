import type { NurseSession } from './useNurseSession';

export default function CareResponsePanel({session}: {session: NurseSession}) {
  const alerts = Object.values(session.alerts).filter(a => a.status !== 'resolved' && ['possible_fall','help_requested'].includes(a.kind))
    .sort((a,b) => Number(!!a.accepted_by)-Number(!!b.accepted_by) || Date.parse(a.opened_at)-Date.parse(b.opened_at));
  return <section className="card care-response" aria-label="Caregiver response">
    <h2>Caregiver response · {alerts.length}</h2>
    {!session.nurse ? <p>Start a nurse session and select Help requests and Possible falls to receive alerts here.</p> : !session.connected ? <p role="status">Disconnected. Waiting for current requests.</p> : !alerts.length ? <p>No active requests in your coverage.</p> : null}
    <div className="response-list" aria-live="polite">{alerts.map(alert => {
      const mine = alert.accepted_by === session.nurse?.nurse_id;
      const label = alert.kind === 'possible_fall' ? 'Possible fall' : alert.metadata.method === 'button' ? 'Button request' : 'Hand signal';
      return <article className="alert" key={alert.alert_id}>
        <strong>{label}</strong><p className="small">{alert.room_id} · {alert.source_type === 'simulation' ? 'Simulation' : alert.source_type === 'replay' ? 'Recorded replay' : alert.source_type === 'edge' ? 'Camera' : 'Patient button'}</p>
        {alert.metadata.identity_demo === true && typeof alert.metadata.resident_label === 'string' && <p>{alert.metadata.resident_label}</p>}
        <p>{alert.accepted_by_name ? `${alert.accepted_by_name} is responding` : 'Awaiting caregiver response'}</p>
        <button disabled={!session.connected || session.pending || (!mine && (!!alert.accepted_by || !session.nurse?.on_duty))} onClick={() => void session.action(alert.alert_id,mine ? 'resolve':'accept')}>{mine ? 'Resolve' : alert.accepted_by ? 'Being handled' : "Received — I’m on it"}</button>
      </article>;
    })}</div>
    {session.error && <p role="alert">{session.error}</p>}
  </section>;
}

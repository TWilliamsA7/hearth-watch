import { useEffect, useRef, useState } from 'react';
import type { Alert } from './generated/contract';
import type { NurseSession } from './useNurseSession';
import { alertReason, alertTitle, identityLine, recoveryText, sourceTypeLabels } from './alertText';

// Every alert in the nurse's coverage, or — before a session starts — this room's alerts,
// which can be read but not accepted.
export default function IncidentList({ session, roomAlerts, room }: { session: NurseSession; roomAlerts: Record<string, Alert>; room: string }) {
  const [showResolved, setShowResolved] = useState(false);
  const all = Object.values(session.nurse ? session.alerts : roomAlerts).sort((a, b) => Date.parse(b.opened_at) - Date.parse(a.opened_at));
  const open = all.filter(a => a.status === 'open').length;
  const accepted = all.filter(a => a.status === 'acknowledged').length;
  const resolved = all.filter(a => a.status === 'resolved').length;
  const visible = all.filter(a => showResolved || a.status !== 'resolved');
  // Stays as the nurse left it, but opens again when a new alert needs attention.
  const [expanded, setExpanded] = useState(true);
  const previousOpen = useRef(open);
  useEffect(() => { if (open > previousOpen.current) setExpanded(true); previousOpen.current = open; }, [open]);
  return <details className="alerts-card" id="alerts" open={expanded} onToggle={e => setExpanded(e.currentTarget.open)}>
    <summary>Incident alerts · {open + accepted}</summary>
    <div className="section-heading"><h2>Incident alerts</h2><span className="count-chip">{open + accepted}</span></div>
    <p className="small">{open} need attention · {accepted} accepted · {resolved} resolved{session.nurse ? ' · your coverage' : ` · ${room} only`}
      {' '}<label><input type="checkbox" checked={showResolved} onChange={e => setShowResolved(e.target.checked)} /> Show resolved</label></p>
    <div aria-live="polite">
      {visible.length ? visible.map(alert => <AlertCard key={alert.alert_id} alert={alert} session={session} />)
        : <p className="empty">{!session.nurse ? 'No alerts in this room. No alerts does not confirm room safety.' : session.nurse.on_duty ? 'No requests need attention.' : 'You are off duty. Go on duty and save coverage to receive new requests.'}</p>}
      <p className="small">{session.nurse ? 'Accepting assigns the request to you. Resolve it after checking on the patient.' : 'Start a nurse session to accept and resolve requests.'}</p>
    </div>
  </details>;
}

function AlertCard({ alert, session }: { alert: Alert; session: NurseSession }) {
  const mine = !!session.nurse && alert.accepted_by === session.nurse.nurse_id;
  const identity = identityLine(alert);
  const recovery = recoveryText(alert);
  return <div className="alert">
    <span className="alert-label">{alert.room_id} · {sourceTypeLabels[alert.source_type] ?? alert.source_type} · {alert.status === 'acknowledged' ? 'accepted' : alert.status}</span>
    <h3>{alertTitle(alert)}</h3>
    {identity && <p className="small">{identity}</p>}
    <p>{alertReason(alert)}</p>
    {recovery && <p className="small">{recovery}</p>}
    <p className="small">Opened {new Date(alert.opened_at).toLocaleTimeString()} · {alert.source}</p>
    <p>{alert.accepted_by_name ? `Accepted by ${alert.accepted_by_name}` : alert.status === 'acknowledged' ? 'Acknowledged without an assigned nurse' : alert.status === 'resolved' ? 'Resolved' : 'No nurse has accepted yet.'}</p>
    {session.nurse && alert.status !== 'resolved' && <button
      disabled={!session.connected || session.pending || (!mine && (!!alert.accepted_by || !session.nurse.on_duty))}
      onClick={() => void session.action(alert.alert_id, mine ? 'resolve' : 'accept')}>{mine ? 'Resolve' : alert.accepted_by ? 'Assigned' : 'Accept request'}</button>}
  </div>;
}

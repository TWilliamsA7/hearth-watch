import type { AppointmentFeed } from './useAppointments';
import { useRef, useState } from 'react';
import { requestHelp } from './api';
import type { Alert } from './generated/contract';
import AppointmentPanel from './AppointmentPanel';

export default function ResidentPanel({ base, connected, alerts, appointments }: { base: string | null; connected: boolean; alerts: Record<string, Alert>; appointments: AppointmentFeed }) {
  const [request, setRequest] = useState<Alert | null>(null);
  const [previewSent, setPreviewSent] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  // Keep this id on retry so a network failure cannot duplicate a help request.
  const requestId = useRef(crypto.randomUUID());
  const roomRequests = Object.values(alerts).filter(a => a.room_id === 'room-01' && a.kind === 'help_requested');
  const currentRequest = request ? alerts[request.alert_id] ?? request : roomRequests.find(a => a.metadata.method === 'button' && a.status !== 'resolved') ?? null;
  async function send() {
    if (!base) { setPreviewSent(true); return; }
    if (currentRequest?.status === 'resolved') { requestId.current = crypto.randomUUID(); setRequest(null); }
    setPending(true); setError('');
    try { setRequest(await requestHelp(base, requestId.current)); }
    catch (e) { setError((e as Error).message); }
    finally { setPending(false); }
  }
  const sent = (!!currentRequest && currentRequest.status !== 'resolved') || previewSent;
  return <section className="resident-layout">
    <div className="card resident-help"><p className="eyebrow">Room 01</p><h2>Hello{!base ? ', Margaret' : ''}</h2><p>{!base ? 'Patient view · example only' : 'Room 01'}</p>
      <button aria-label="Request help" className={`help-button ${sent ? 'sent' : ''}`} onClick={send} disabled={pending || sent || (!!base && !connected)}>{pending ? 'Sending…' : sent ? 'Request received' : 'SOS'}</button>
      <div aria-live="polite"><p>{!base ? previewSent ? 'Example only. No request was sent.' : 'Preview only. This button demonstrates a help request.' : currentRequest?.status === 'resolved' ? 'Monitoring marked your previous request as resolved. You can request help again.' : currentRequest?.accepted_by_name ? `${currentRequest.accepted_by_name} has accepted your request.` : currentRequest ? 'Your request has been received. Waiting for a caregiver to accept.' : !connected ? 'Monitoring is disconnected. Help requests cannot be sent here.' : 'Your request will be sent to the monitoring system.'}</p></div>
      {roomRequests.filter(a => a.metadata.method !== 'button' && a.status !== 'resolved').map(a => <p key={a.alert_id} role="status">Camera help signal received. {a.accepted_by_name ? `${a.accepted_by_name} has accepted your request.` : 'Waiting for a caregiver.'}</p>)}
      {error && <p role="alert" className="error">{error} Your request has not been confirmed. Try again.</p>}
    </div>
    <AppointmentPanel preview={!base} feed={appointments} alerts={alerts} />
  </section>;
}

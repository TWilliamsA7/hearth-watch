import type { AppointmentFeed } from './useAppointments';
import { useEffect, useRef, useState } from 'react';
import { requestHelp } from './api';
import type { Alert } from './generated/contract';
import type { RoomSource } from './protocol';
import AppointmentPanel from './AppointmentPanel';
import RecognitionPanel from './RecognitionPanel';
import { postureText, sourceHealth } from './alertText';

export default function ResidentPanel({ base, room, connected, alerts, sources, gestureHoldSeconds, appointments }: {
  base: string; room: string; connected: boolean; alerts: Record<string, Alert>; sources: RoomSource[];
  gestureHoldSeconds: number | null; appointments: AppointmentFeed;
}) {
  const [request, setRequest] = useState<Alert | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  // Keep this id on retry so a network failure cannot duplicate a help request.
  const requestId = useRef(crypto.randomUUID());
  useEffect(() => { setRequest(null); setError(''); requestId.current = crypto.randomUUID(); }, [room]);
  const roomRequests = Object.values(alerts).filter(a => a.room_id === room && a.kind === 'help_requested');
  const currentRequest = request ? alerts[request.alert_id] ?? request : roomRequests.find(a => a.metadata.method === 'button' && a.status !== 'resolved') ?? null;
  async function send() {
    if (currentRequest?.status === 'resolved') { requestId.current = crypto.randomUUID(); setRequest(null); }
    setPending(true); setError('');
    try { setRequest(await requestHelp(base, requestId.current, room)); requestId.current = crypto.randomUUID(); }
    catch (e) { setError((e as Error).message); }
    finally { setPending(false); }
  }
  const sent = !!currentRequest && currentRequest.status !== 'resolved';
  return <section className="resident-layout">
    <div className="card resident-help"><p className="eyebrow">{room}</p><h2>Hello</h2>
      <button aria-label="Request help" className={`help-button ${sent ? 'sent' : ''}`} onClick={send} disabled={pending || sent || !connected}>{pending ? 'Sending…' : sent ? 'Request received' : 'SOS'}</button>
      <div aria-live="polite"><p>{currentRequest?.status === 'resolved' ? 'Monitoring marked your previous request as resolved. You can request help again.' : currentRequest?.accepted_by_name ? `${currentRequest.accepted_by_name} has accepted your request.` : currentRequest ? 'Your request has been received. Waiting for a caregiver to accept.' : !connected ? 'Monitoring is disconnected. Help requests cannot be sent here.' : 'Your request will be sent to the care team.'}</p></div>
      {roomRequests.filter(a => a.metadata.method !== 'button' && a.status !== 'resolved').map(a => <p key={a.alert_id} role="status">Camera help signal received. {a.accepted_by_name ? `${a.accepted_by_name} has accepted your request.` : 'Waiting for a caregiver.'}</p>)}
      {error && <p role="alert" className="error">{error} Your request has not been confirmed. Try again.</p>}
      {gestureHoldSeconds != null && <p className="small">No button nearby? Hold one hand up toward the camera for {gestureHoldSeconds} seconds.</p>}
      <p className="small">{sources.length ? sources.map(s => `${s.source}: ${sourceHealth(s).toLowerCase()} · ${postureText(s)}`).join(' · ') : 'No camera observations yet. The help button works without the camera.'}</p>
    </div>
    <div className="resident-side">
      <AppointmentPanel feed={appointments} alerts={alerts} room={room} scheduling />
      <RecognitionPanel room={room} appointments={appointments.items} onCheckedIn={appointments.reload} />
    </div>
  </section>;
}

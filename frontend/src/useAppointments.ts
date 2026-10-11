import { useEffect, useRef, useState } from 'react';

export type Appointment = {
  appointment_id: string; resident_label: string; room_id: string;
  scheduled_at: string; status: 'scheduled' | 'overdue' | 'arrived' | 'cancelled';
  check_in_method: string | null; alert_id: string | null;
};
export function useAppointments(base: string | null) {
  const session = useRef(0);
  const revision = useRef(0);
  const [items, setItems] = useState<Appointment[]>([]);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const generation = ++session.current;
    setItems([]); setError(''); setPending(false);
    if (!base) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      const version = revision.current;
      try {
        const response = await fetch(new URL('/appointments', base!), {signal: AbortSignal.any([abort.signal, AbortSignal.timeout(5000)])});
        if (!response.ok) throw new Error(`Schedule unavailable (${response.status}).`);
        const data: unknown = await response.json();
        if (!Array.isArray(data) || !data.every(item => item && typeof item.appointment_id === 'string' && typeof item.resident_label === 'string' && typeof item.room_id === 'string' && Number.isFinite(Date.parse(item.scheduled_at)) && ['scheduled','overdue','arrived','cancelled'].includes(item.status))) throw new Error('Invalid appointment response.');
        if (!abort.signal.aborted && version === revision.current) { setItems(data.map(({appointment_id, resident_label, room_id, scheduled_at, status, check_in_method, alert_id}) => ({appointment_id, resident_label, room_id, scheduled_at, status, check_in_method, alert_id}))); setError(''); }
      } catch (e) { if (!abort.signal.aborted) setError((e as Error).message); }
      finally { if (!abort.signal.aborted) timer = setTimeout(poll, 5000); }
    }
    void poll();
    return () => { if (session.current === generation) session.current++; abort.abort(); clearTimeout(timer); };
  }, [base, refresh]);
  async function checkIn(id: string) {
    if (!base || pending || error) return;
    const generation = session.current;
    revision.current++;
    setPending(true);
    try {
      const response = await fetch(new URL(`/appointments/${encodeURIComponent(id)}/check-in`, base), {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({method:'manual'}), signal:AbortSignal.timeout(5000)});
      if (!response.ok) throw new Error(`Check-in not confirmed (${response.status}).`);
      const item = await response.json();
      if (item.appointment_id !== id || item.status !== 'arrived') throw new Error('Arrival not confirmed.');
      if (session.current === generation) setRefresh(value => value + 1);
    } catch (e) { if (session.current === generation) setError((e as Error).message); }
    finally { if (session.current === generation) setPending(false); }
  }
  async function create(resident_label: string, scheduled_at: string, room_id: string) {
    if (!base || pending) return false;
    const generation = session.current; revision.current++; setPending(true); setError('');
    try {
      const response = await fetch(new URL('/appointments',base), {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({resident_label,room_id,scheduled_at,grace_seconds:900,demo:false}),signal:AbortSignal.timeout(5000)});
      if (!response.ok) throw new Error(`Appointment not saved (${response.status}).`);
      if (session.current !== generation) return false;
      setRefresh(value => value+1); return true;
    } catch (e) { if (session.current === generation) setError((e as Error).message); return false; }
    finally { if (session.current === generation) setPending(false); }
  }
  async function cancel(id: string) {
    if (!base || pending) return;
    const generation = session.current; revision.current++; setPending(true); setError('');
    try {
      const response = await fetch(new URL(`/appointments/${encodeURIComponent(id)}/cancel`,base), {method:'POST',signal:AbortSignal.timeout(5000)});
      if (!response.ok) throw new Error(`Appointment cancellation failed (${response.status}).`);
      const item = await response.json();
      if (item.appointment_id !== id || item.status !== 'cancelled') throw new Error('Cancellation was not confirmed.');
      if (session.current === generation) setRefresh(value => value+1);
    } catch (e) { if (session.current === generation) setError((e as Error).message); }
    finally { if (session.current === generation) setPending(false); }
  }
  return {items, error, pending, checkIn, create, cancel, reload: () => setRefresh(value => value + 1)};
}
export type AppointmentFeed = ReturnType<typeof useAppointments>;

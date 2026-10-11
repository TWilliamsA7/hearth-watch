import { useState } from 'react';
import type { AppointmentFeed } from './useAppointments';
import type { Alert } from './generated/contract';

export default function AppointmentPanel({compact = false, feed, scheduling = false, alerts = {}, room}: {compact?: boolean; feed: AppointmentFeed; scheduling?: boolean; alerts?: Record<string, Alert>; room: string}) {
  const [selected, setSelected] = useState('');
  const [name,setName] = useState('');
  const [date,setDate] = useState('');
  const [saved,setSaved] = useState(false);
  const [page, setPage] = useState(0);
  const inRoom = feed.items.filter(item => item.room_id === room && item.status !== 'cancelled').sort((a,b) => Date.parse(a.scheduled_at)-Date.parse(b.scheduled_at));
  const today = inRoom.filter(item => new Date(item.scheduled_at).toDateString() === new Date().toDateString());
  const open = (item: typeof inRoom[number]) => ['scheduled','overdue'].includes(item.status);
  const next = today.find(open) ?? today.find(item => item.status === 'arrived');
  const pageCount = Math.max(1, Math.ceil(inRoom.length / 4));
  const currentPage = Math.min(page, pageCount - 1);
  const shown = inRoom.slice(currentPage * 4, (currentPage + 1) * 4);
  const chosen = inRoom.find(item => item.appointment_id === selected && open(item));
  const time = (value: string) => new Date(value).toLocaleTimeString([], {hour:'numeric',minute:'2-digit'});
  const nurse = next?.alert_id ? alerts[next.alert_id]?.accepted_by_name : null;
  return <section className={`card appointment-card ${compact ? 'appointment-summary' : ''}`}>
    <div className="section-heading"><h2>{compact ? "Today's appointment" : 'My appointments'}</h2>{!compact && <span className="appointment-status">{feed.error ? 'Unavailable' : `${today.length} today`}</span>}</div>
    {compact ? <dl className="appointment-facts"><div><dt>Check-in time</dt><dd>{next ? time(next.scheduled_at) : '—'}</dd></div><div><dt>Room</dt><dd>{next?.room_id ?? room}</dd></div><div><dt>Status</dt><dd><span className="chip wait">{feed.error ? 'Unavailable' : next?.status ?? 'No appointment'}</span></dd></div><div><dt>Nurse</dt><dd>{nurse ?? '—'}</dd></div></dl> : <>
      {inRoom.length ? <ul className="appts">{shown.map(item => <li key={item.appointment_id} className={selected === item.appointment_id ? 'next' : ''} role="button" tabIndex={0} aria-pressed={selected === item.appointment_id} onClick={() => setSelected(item.appointment_id)} onKeyDown={event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); setSelected(item.appointment_id);}}}><div><b>{item.resident_label}</b><small>{new Date(item.scheduled_at).toLocaleDateString()} · {item.room_id} · {item.status}{['face_recognition','face_recognition_demo'].includes(item.check_in_method ?? '') ? ' · Recognized arrival' : ''}</small></div><span className="when">{time(item.scheduled_at)}</span></li>)}</ul> : <div className="appointment-empty"><span className="appointment-symbol" aria-hidden="true">▦</span><h3>Upcoming appointments</h3><p>{feed.error || 'No appointments scheduled for this room.'}</p></div>}
      <button className="appointment-checkin" disabled={!chosen || feed.pending || !!feed.error} onClick={() => chosen && void feed.checkIn(chosen.appointment_id)}>{feed.pending ? 'Confirming…' : chosen ? `Check in for ${time(chosen.scheduled_at)}` : 'Select an appointment to check in'}</button>
    </>}
    {scheduling && <details className="observation-help"><summary>Schedule an appointment</summary><form onSubmit={async event => {event.preventDefault(); setSaved(false); if(await feed.create(name.trim(),new Date(date).toISOString(),room)) {setSaved(true); setName(''); setDate('');}}}>
      <label>Patient name <input required maxLength={80} value={name} onChange={e => setName(e.target.value)}/></label>
      <label>Date and time <input required type="datetime-local" value={date} onChange={e => setDate(e.target.value)}/></label>
      <p className="small">{room} · Notify subscribed nurses 15 minutes after the scheduled time if arrival is unconfirmed.</p>
      <button disabled={feed.pending || !name.trim() || !date}>Save appointment</button>{saved && <p role="status">Appointment saved.</p>}
    </form></details>}
    {scheduling && <details className="observation-help"><summary>Manage scheduled appointments ({inRoom.filter(open).length})</summary>
      {inRoom.filter(open).length ? <ul className="appointment-management">{inRoom.filter(open).map(item => <li key={item.appointment_id}><span><b>{item.resident_label}</b><br/>{new Date(item.scheduled_at).toLocaleString()} · {item.status}</span><button disabled={feed.pending} onClick={() => void feed.cancel(item.appointment_id)}>Cancel appointment</button></li>)}</ul> : <p>No scheduled appointments to clear.</p>}
      <p className="small">Cancelling removes the appointment from the active schedule and closes its overdue reminder.</p>
    </details>}
    <p className="small">{feed.error || `${today.length} today · ${today.filter(item => item.status === 'arrived').length} arrived · ${today.filter(item => item.status === 'overdue').length} overdue`}{!compact && pageCount > 1 && <> · <button disabled={currentPage === 0} onClick={() => setPage(currentPage-1)}>Previous</button> {currentPage+1}/{pageCount} <button disabled={currentPage+1 === pageCount} onClick={() => setPage(currentPage+1)}>Next</button></>}{!compact && <> · <a href="#recognition">Check in with face recognition</a></>}</p>
  </section>;
}

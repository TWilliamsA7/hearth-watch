export default function AppointmentPanel({ compact = false, preview = false }: { compact?: boolean; preview?: boolean }) {
  return <section className={`card appointment-card ${compact ? 'appointment-summary' : ''}`}>
    <div className="section-heading"><h2>{compact ? "Today's appointment" : 'My appointments'}</h2>{!compact && <span className="appointment-status">{preview ? 'Example schedule' : 'Not connected'}</span>}</div>
    {compact ? <dl className="appointment-facts"><div><dt>Check-in time</dt><dd>{preview ? '2:30 PM' : '—'}</dd></div><div><dt>Room</dt><dd>{preview ? '204' : '—'}</dd></div><div><dt>Status</dt><dd><span className="chip wait">{preview ? 'Example only' : 'Unavailable'}</span></dd></div><div><dt>Nurse</dt><dd>{preview ? 'Nurse Carter' : '—'}</dd></div></dl> : preview ? <>
      <ul className="appts"><li className="next"><div><b>Check-up with Nurse Carter</b><small>Today · Room 204</small></div><span className="when">2:30 PM</span></li><li><div><b>Physical therapy</b><small>Today · Room 110</small></div><span className="when">4:00 PM</span></li><li><div><b>Blood pressure check</b><small>Tomorrow · Room 204</small></div><span className="when">9:15 AM</span></li><li><div><b>Eye exam</b><small>Friday · Room 305</small></div><span className="when">11:00 AM</span></li></ul>
      <button className="appointment-checkin" disabled>Check in for 2:30 PM · example</button>
    </> : <>
      <div className="appointment-empty"><span className="appointment-symbol" aria-hidden="true">▦</span><h3>Upcoming appointments</h3><p>Appointments will appear here when a scheduling service is connected.</p></div>
      <button className="appointment-checkin" disabled>Check-in unavailable</button>
    </>}
    <p className="small">{preview ? 'Example schedule · no appointment or check-in is connected.' : compact ? 'Scheduling is not connected.' : 'No appointment or check-in has been confirmed.'}</p>
  </section>;
}

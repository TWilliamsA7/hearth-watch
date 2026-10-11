import { useEffect, useRef, useState } from 'react';
import type { Appointment } from './useAppointments';

// Optional, opt-in patient recognition: enroll from three camera views, then confirm an
// arrival with three consistent matches. Frames go only to this computer's server, which
// accepts them only from a page it served itself (loopback, same origin, explicit header).
type Status = { enabled: boolean; message: string };
type Profile = { profile_id: string; alias: string; expires_at: number; appointment_ids?: string[] };
type Recognition = { status: string; matches?: number; checked_in?: boolean; profile?: { alias: string } };

async function api<T>(path: string, method = 'GET', data?: unknown): Promise<T> {
  const response = await fetch(path, { method, headers: { 'Content-Type': 'application/json', 'X-Hearth-Face-Demo': '1' },
    body: data === undefined ? undefined : JSON.stringify(data) });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : `Request failed (${response.status}).`);
  return value as T;
}
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const when = (value: string) => new Date(value).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' });

export default function RecognitionPanel({ room, appointments, onCheckedIn }: { room: string; appointments: Appointment[]; onCheckedIn: () => void }) {
  const [status, setStatus] = useState<Status | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [alias, setAlias] = useState('');
  const [appointment, setAppointment] = useState('');
  const [consent, setConsent] = useState(false);
  const [identifyAlerts, setIdentifyAlerts] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState('Camera is off. Nothing is being captured.');
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef<string | null>(null);
  // Bumped by every start/stop so a stale capture loop stops on its own.
  const generation = useRef(0);
  const ready = !!status?.enabled;
  const openAppointments = appointments.filter(a => a.room_id === room && ['scheduled', 'overdue'].includes(a.status));

  async function refresh() {
    setProfiles(await api<Profile[]>('/face-demo/profiles'));
  }
  function stop() {
    generation.current++;
    if (session.current) { const id = session.current; session.current = null; api(`/face-demo/sessions/${id}`, 'DELETE').catch(() => {}); }
    stream.current?.getTracks().forEach(track => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setStreaming(false); setBusy(false);
    setResult('Camera is off. Nothing is being captured.');
  }
  useEffect(() => {
    api<Status>('/face-demo/status').then(value => { setStatus(value); if (value.enabled) refresh().catch(e => setResult((e as Error).message)); })
      .catch(e => setStatus({ enabled: false, message: (e as Error).message }));
    const hide = () => { if (document.hidden) stop(); };
    window.addEventListener('pagehide', stop);
    document.addEventListener('visibilitychange', hide);
    return () => { window.removeEventListener('pagehide', stop); document.removeEventListener('visibilitychange', hide); stop(); };
  }, []);
  useEffect(() => { if (appointment && !openAppointments.some(a => a.appointment_id === appointment)) setAppointment(''); }, [room, appointments]);

  function frame(): string {
    const el = video.current;
    if (!stream.current || !el?.videoWidth) throw new Error('Start the camera and wait for a preview.');
    const canvas = document.createElement('canvas');
    canvas.width = 640; canvas.height = Math.round(el.videoHeight * 640 / el.videoWidth);
    canvas.getContext('2d')!.drawImage(el, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', .85).split(',')[1];
  }
  async function open() {
    const request = ++generation.current;
    try {
      const opened = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      if (request !== generation.current) { opened.getTracks().forEach(track => track.stop()); return; }
      stream.current = opened;
      if (video.current) { video.current.srcObject = opened; await video.current.play(); }
      setStreaming(true); setResult('Preview is local. Enroll yourself or start recognition when ready.');
    } catch (e) { setResult(`Camera unavailable: ${(e as Error).message}`); }
  }
  async function enroll() {
    const name = alias.trim();
    if (!name || !consent) { setResult('Enter your preferred name and confirm your consent.'); return; }
    setBusy(true); const active = ++generation.current;
    const frames: string[] = [];
    try {
      for (const direction of ['Look straight ahead.', 'Turn slightly left.', 'Turn slightly right.']) {
        setResult(direction); await delay(1400);
        if (active !== generation.current) return;
        frames.push(frame());
      }
      await api('/face-demo/enroll', 'POST', { alias: name, consent, identify_alerts: identifyAlerts, frames, appointment_id: appointment || null });
      setConsent(false); setIdentifyAlerts(false);
      await refresh(); setResult(`${name} enrolled. You can now check in using recognition.`);
    } catch (e) { setResult((e as Error).message); }
    finally { frames.length = 0; if (active === generation.current) setBusy(false); }
  }
  async function recognize() {
    setBusy(true); const active = ++generation.current;
    let own: string | null = null;
    try {
      own = (await api<{ session_id: string }>('/face-demo/sessions', 'POST', { room_id: room })).session_id;
      if (active !== generation.current) return;
      session.current = own;
      while (active === generation.current && stream.current) {
        try {
          const value = await api<Recognition>('/face-demo/recognize', 'POST', { session_id: own, frame: frame() });
          if (active !== generation.current) break;
          if (value.status === 'recognized') {
            setResult(`${value.profile?.alias ?? 'Patient'} recognized.${value.checked_in ? ' Appointment checked in.' : ' Recognition confirmed.'}`);
            if (value.checked_in) onCheckedIn();
            await refresh(); break;
          }
          setResult(value.status === 'confirming' ? `Confirming match: ${value.matches}/3` : 'No confirmed match. Face the camera in good light.');
        } catch (e) { setResult((e as Error).message); }
        await delay(650);
      }
    } catch (e) { setResult((e as Error).message); }
    finally {
      if (own) await api(`/face-demo/sessions/${own}`, 'DELETE').catch(() => {});
      if (session.current === own) session.current = null;
      if (active === generation.current) setBusy(false);
    }
  }
  async function act(fn: () => Promise<unknown>, done: string) {
    try { await fn(); await refresh(); setResult(done); } catch (e) { setResult((e as Error).message); }
  }

  return <section className="card recognition-card" id="recognition" aria-label="Patient recognition">
    <div className="section-heading"><h2>Face recognition check-in</h2><span className="appointment-status">Optional</span></div>
    <p role="status" className="small">{status ? status.message : 'Checking setup…'}</p>
    {ready && <div className="recognition-grid">
      <div>
        <label>Patient name <input maxLength={40} value={alias} placeholder="Your preferred name" onChange={e => setAlias(e.target.value)} /></label>
        <label>Link an appointment in {room}<select value={appointment} onChange={e => setAppointment(e.target.value)}>
          <option value="">Recognition only</option>
          {openAppointments.map(a => <option key={a.appointment_id} value={a.appointment_id}>{a.resident_label} · {when(a.scheduled_at)}</option>)}
        </select></label>
        <label className="consent"><input type="checkbox" checked={consent} onChange={e => setConsent(e.target.checked)} /> I am the person in front of the camera and opt in to face recognition. Only appointments I explicitly link may be checked in.</label>
        <label className="consent"><input type="checkbox" checked={identifyAlerts} onChange={e => setIdentifyAlerts(e.target.checked)} /> Also show my name on possible-fall and camera help-signal alerts. Alerts still work without it.</label>
        <p className="small">Camera frames go to this computer's server for matching and are not saved. Enrollments are encrypted at rest and expire after 24 hours. Withdraw below at any time.</p>
      </div>
      <div>
        <video ref={video} className="recognition-video" autoPlay muted playsInline aria-label="Local camera preview" />
        <div className="buttons">
          <button disabled={streaming || busy} onClick={() => void open()}>Start camera</button>
          <button disabled={!streaming || busy || !consent} onClick={() => void enroll()}>Capture 3 views and enroll</button>
          <button disabled={!streaming || busy || !profiles.length} onClick={() => void recognize()}>Check in with recognition</button>
          <button disabled={!streaming} onClick={stop}>Stop camera</button>
        </div>
        <p role="status" aria-live="polite">{result}</p>
        <p className="small">Three consistent matches are required. Recognition can be wrong; a caregiver can confirm arrival manually. Stop this camera before starting room monitoring on the same webcam.</p>
      </div>
    </div>}
    {ready && <details className="observation-help"><summary>Current enrollments ({profiles.length})</summary>
      {profiles.length ? <ul className="appointment-management">{profiles.map(profile => <li key={profile.profile_id}>
        <span><b>{profile.alias}</b><br />expires {new Date(profile.expires_at * 1000).toLocaleString()}</span>
        <span className="buttons">
          <button disabled={!appointment} title={appointment ? '' : 'Select an appointment above first'} onClick={() => void act(() => api(`/face-demo/profiles/${profile.profile_id}/appointments/${appointment}`, 'POST'), `Appointment linked to ${profile.alias}. No new enrollment needed.`)}>Link selected appointment</button>
          {(profile.appointment_ids ?? []).map(id => { const item = appointments.find(a => a.appointment_id === id); return <button key={id} onClick={() => void act(() => api(`/face-demo/profiles/${profile.profile_id}/appointments/${id}`, 'DELETE'), 'Recognition check-in link removed. The appointment and enrollment remain.')}>{item ? `Unlink ${item.room_id} · ${when(item.scheduled_at)}` : 'Remove appointment link'}</button>; })}
          <button onClick={() => { stop(); void act(() => api(`/face-demo/profiles/${profile.profile_id}`, 'DELETE'), 'Enrollment deleted. Recognition has stopped.'); }}>Withdraw &amp; delete</button>
        </span>
      </li>)}</ul> : <p className="small">No patients enrolled.</p>}
    </details>}
  </section>;
}

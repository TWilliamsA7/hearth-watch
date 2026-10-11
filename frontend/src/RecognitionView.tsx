import { useEffect, useRef, useState } from 'react';

// Staff-run, opt-in resident recognition: enroll a consenting resident from three camera views,
// then check that the camera recognizes them. Named alerts need a second, separate opt-in.
// Frames go only to this computer's server, which accepts them only from a page it served.
type Status = { enabled: boolean; message: string };
type Profile = { profile_id: string; alias: string; expires_at: number; identify_alerts?: boolean };
type Recognition = { status: string; matches?: number; profile?: { alias: string } };

async function api<T>(path: string, method = 'GET', data?: unknown): Promise<T> {
  const response = await fetch(path, { method, headers: { 'Content-Type': 'application/json', 'X-Hearth-Face-Demo': '1' },
    body: data === undefined ? undefined : JSON.stringify(data) });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : `Request failed (${response.status}).`);
  return value as T;
}
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export default function RecognitionView({ room }: { room: string }) {
  const [status, setStatus] = useState<Status | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [name, setName] = useState('');
  const [consent, setConsent] = useState(false);
  const [identifyAlerts, setIdentifyAlerts] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState('Camera is off. Nothing is being captured.');
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef<string | null>(null);
  // Bumped by every start and stop, so a stale capture loop ends on its own.
  const generation = useRef(0);
  const ready = !!status?.enabled;

  async function refresh() { setProfiles(await api<Profile[]>('/face-demo/profiles')); }
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
      setStreaming(true); setResult('Preview is local. Enroll or test recognition when ready.');
    } catch (e) { setResult(`Camera unavailable: ${(e as Error).message}`); }
  }
  async function enroll() {
    const alias = name.trim();
    if (!alias || !consent) { setResult("Enter the resident's name and confirm their consent."); return; }
    setBusy(true); const active = ++generation.current;
    const frames: string[] = [];
    try {
      for (const direction of ['Look straight ahead.', 'Turn slightly left.', 'Turn slightly right.']) {
        setResult(direction); await delay(1400);
        if (active !== generation.current) return;
        frames.push(frame());
      }
      await api('/face-demo/enroll', 'POST', { alias, consent, identify_alerts: identifyAlerts, frames });
      setConsent(false); setIdentifyAlerts(false); setName('');
      await refresh(); setResult(`${alias} enrolled.`);
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
          if (value.status === 'recognized') { setResult(`${value.profile?.alias ?? 'Resident'} recognized.`); break; }
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
  async function withdraw(profile: Profile) {
    stop();
    try { await api(`/face-demo/profiles/${profile.profile_id}`, 'DELETE'); await refresh(); setResult(`${profile.alias}'s enrollment was deleted.`); }
    catch (e) { setResult((e as Error).message); }
  }

  return <div className="page recognition-page">
    <section className="card">
      <div className="section-heading"><h2>Resident recognition</h2><span className={`status ${ready ? '' : 'unknown'}`}><span className="tiny-dot" />{ready ? 'Ready' : 'Unavailable'}</span></div>
      <p>Optional. Enroll here on this laptop; the Jetson camera then recognizes enrolled people (three matching face checks) and, if they agreed to be named, shows their name on their alerts and on their figure in the simulated room (Live people). Alerts work the same without it.</p>
      {status && !ready && <p role="alert" className="error">{status.message}</p>}
      {!status && <p className="small">Checking setup…</p>}
    </section>
    {ready && <div className="recognition-grid">
      <section className="card">
        <h2>Camera</h2>
        <video ref={video} className="recognition-video" autoPlay muted playsInline aria-label="Local camera preview" />
        <div className="buttons">
          <button disabled={streaming || busy} onClick={() => void open()}>Start camera</button>
          <button disabled={!streaming || busy || !profiles.length} onClick={() => void recognize()}>Test recognition</button>
          <button disabled={!streaming} onClick={stop}>Stop camera</button>
        </div>
        <p role="status" aria-live="polite">{result}</p>
        <p className="small">Three consistent matches are required, and recognition can be wrong. Stop this camera before room monitoring needs the same webcam.</p>
      </section>
      <section className="card">
        <h2>Enroll a resident</h2>
        <label>Resident's name<input maxLength={40} value={name} onChange={e => setName(e.target.value)} /></label>
        <label className="consent"><input type="checkbox" checked={consent} onChange={e => setConsent(e.target.checked)} /> The resident in front of the camera has agreed to face recognition.</label>
        <label className="consent"><input type="checkbox" checked={identifyAlerts} onChange={e => setIdentifyAlerts(e.target.checked)} /> They have also agreed to be named: on their alerts, and in the room view when the Jetson recognizes them.</label>
        <button className="go" disabled={!streaming || busy || !consent || !name.trim()} onClick={() => void enroll()}>Capture 3 views and enroll</button>
        <p className="small">Frames are processed in memory and not saved. Enrollments are encrypted and expire after 24 hours.</p>
        <h2>Enrolled · {profiles.length}</h2>
        {profiles.length ? <ul className="enrollments">{profiles.map(profile => <li key={profile.profile_id}>
          <span><b>{profile.alias}</b><br /><span className="small">{profile.identify_alerts ? 'Named on alerts' : 'Not named on alerts'} · expires {new Date(profile.expires_at * 1000).toLocaleString()}</span></span>
          <button onClick={() => void withdraw(profile)}>Withdraw &amp; delete</button>
        </li>)}</ul> : <p className="small">Nobody is enrolled.</p>}
      </section>
    </div>}
  </div>;
}

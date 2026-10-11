import { useEffect, useRef, useState } from 'react';
import type { Alert } from './generated/contract';
import { acknowledgeAlert, resolveAlert } from './api';
import { alertReason, alertTitle, identityLine, recoveryText, sourceTypeLabels } from './alertText';

// Every alert in the room, from any source. Acknowledge means someone has seen it;
// resolve it after checking on the person.
export default function AlertList({ base, alerts, connected }: { base: string; alerts: Record<string, Alert>; connected: boolean }) {
  const [showResolved, setShowResolved] = useState(false);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState('');
  const sound = useAlertSound(alerts, connected);
  const all = Object.values(alerts).sort((a, b) => Number(a.status === 'resolved') - Number(b.status === 'resolved') || Date.parse(b.opened_at) - Date.parse(a.opened_at));
  const open = all.filter(a => a.status === 'open').length;
  const acknowledged = all.filter(a => a.status === 'acknowledged').length;
  const visible = all.filter(a => showResolved || a.status !== 'resolved');
  async function act(alert: Alert) {
    setPending(alert.alert_id); setError('');
    try { await (alert.status === 'open' ? acknowledgeAlert : resolveAlert)(base, alert.alert_id); }
    catch (e) { setError((e as Error).message); }
    finally { setPending(null); }
  }
  return <section className="alerts" aria-label="Alerts">
    <div className="section-heading">
      <h2>Alerts · {open + acknowledged}</h2>
      <button className="link-button" onClick={sound.toggle}>{sound.on ? 'Sound on' : 'Sound off'}</button>
    </div>
    <p className="small">{open} new · {acknowledged} acknowledged{' '}
      <label className="inline-check"><input type="checkbox" checked={showResolved} onChange={e => setShowResolved(e.target.checked)} /> Show resolved</label></p>
    <div className="alert-list" aria-live="polite">
      {visible.length ? visible.map(alert => {
        const identity = identityLine(alert), recovery = recoveryText(alert);
        return <article key={alert.alert_id} className={`alert ${alert.status}`}>
          <span className="alert-label">{sourceTypeLabels[alert.source_type] ?? alert.source_type} · {alert.source} · {new Date(alert.opened_at).toLocaleTimeString()}</span>
          <h3>{alertTitle(alert)}</h3>
          <p>{alertReason(alert)}</p>
          {identity && <p className="small">{identity}</p>}
          {recovery && <p className="small">{recovery}</p>}
          {alert.status === 'resolved' ? <p className="small">Resolved</p>
            : <button disabled={!connected || pending === alert.alert_id} onClick={() => void act(alert)}>{alert.status === 'open' ? 'Acknowledge' : 'Resolve'}</button>}
        </article>;
      }) : <p className="empty">{connected ? 'No alerts. No alerts does not confirm the room is safe.' : 'Disconnected: alerts are unavailable.'}</p>}
    </div>
    {error && <p role="alert" className="error">{error}</p>}
  </section>;
}

// A short tone for each new open alert once enabled (browsers need a click before playing sound).
function useAlertSound(alerts: Record<string, Alert>, connected: boolean) {
  const [on, setOn] = useState(false);
  const audio = useRef<AudioContext | null>(null);
  const seen = useRef<Set<string> | null>(null);
  useEffect(() => { if (!connected) seen.current = null; }, [connected]);
  useEffect(() => {
    const ids = Object.values(alerts).filter(a => a.status === 'open').map(a => a.alert_id);
    // The first batch after (re)connecting is the server's snapshot, not news.
    if (!seen.current) { seen.current = new Set(ids); return; }
    const fresh = ids.filter(id => !seen.current!.has(id));
    fresh.forEach(id => seen.current!.add(id));
    const ctx = audio.current;
    if (fresh.length && on && ctx?.state === 'running') {
      const oscillator = ctx.createOscillator(), gain = ctx.createGain();
      oscillator.connect(gain); gain.connect(ctx.destination); oscillator.frequency.value = 660;
      gain.gain.setValueAtTime(.08, ctx.currentTime); gain.gain.exponentialRampToValueAtTime(.001, ctx.currentTime + .3);
      oscillator.start(); oscillator.stop(ctx.currentTime + .3);
    }
  }, [alerts, on]);
  useEffect(() => () => { void audio.current?.close(); }, []);
  async function toggle() {
    if (on) { setOn(false); return; }
    try { audio.current ??= new AudioContext(); await audio.current.resume(); setOn(true); } catch { setOn(false); }
  }
  return { on, toggle };
}

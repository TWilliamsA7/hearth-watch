import { useEffect, useRef, useState } from 'react';
import type { Alert, Event, NursePreferences, NurseProfile, NurseSnapshot } from './generated/contract';
import { request } from './api';
import { parseMessage, type FeedMessage, type RoomSource } from './protocol';
type NurseEvent = Extract<FeedMessage, {type:'event'}>['data'];
// One session per tab, kept across reloads (sessionStorage is per tab).
const STORED = 'hearth-nurse';
const sourceId = (s: RoomSource) => `${s.room_id}:${s.source_type}:${s.source}`;
function remember(id: string | null) {
  try { if (id) sessionStorage.setItem(STORED, id); else sessionStorage.removeItem(STORED); } catch { /* Sessions still work without storage. */ }
}

export function useNurseSession(base: string | null) {
  const [nurse, setNurse] = useState<NurseProfile | null>(null);
  const [alerts, setAlerts] = useState<Record<string, Alert>>({});
  const [events, setEvents] = useState<NurseEvent[]>([]);
  const [sources, setSources] = useState<Record<string, RoomSource>>({});
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [sound, setSound] = useState(false);
  const audio = useRef<AudioContext | null>(null);
  const generation = useRef(0);
  const seen = useRef(new Set<string>());
  const initialized = useRef(false);
  function snapshot(data: NurseSnapshot) {
    setNurse(data.nurse); setAlerts(Object.fromEntries(data.alerts.map(a => [a.alert_id, a]))); setEvents(data.events);
    setSources(Object.fromEntries(data.sources.map(s => [sourceId(s), s]))); remember(data.nurse.nurse_id);
    seen.current = new Set(data.alerts.map(a => a.alert_id)); initialized.current = true;
  }
  useEffect(() => {
    const version = ++generation.current; setNurse(null); setAlerts({}); setEvents([]); setSources({}); setError(''); setPending(false);
    let stored: string | null = null;
    try { stored = sessionStorage.getItem(STORED); } catch { /* No storage: start a new session. */ }
    if (base && stored) {
      request<NurseSnapshot>(base, `/nurse-sessions/${encodeURIComponent(stored)}/snapshot`)
        .then(data => { if (version === generation.current) snapshot(data); })
        .catch(() => { if (version === generation.current) { remember(null); setError('Your previous nurse session ended when the server restarted. Start a new session.'); } });
    }
    return () => { generation.current++; };
  }, [base]);
  useEffect(() => () => { void audio.current?.close(); }, []);
  useEffect(() => {
    setConnected(false);
    if (!base || !nurse) return;
    let stopped = false;
    let socket: WebSocket;
    let retry: ReturnType<typeof setTimeout>;
    const id = nurse.nurse_id;
    function connect() {
      if (stopped) return;
      initialized.current = false;
      const url = new URL(`/ws/nurse/${encodeURIComponent(id)}`, base!); url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
      socket = new WebSocket(url);
      socket.onmessage = event => {
        if (stopped) return;
        try {
          const message = parseMessage(JSON.parse(event.data));
          if (!message) throw new Error('Unsupported nurse update.');
          if (message.type === 'nurse_snapshot') { snapshot(message.data); setConnected(true); setError(''); }
          if (message.type === 'alert') {
            const item = message.data;
            if (initialized.current && !seen.current.has(item.alert_id) && item.status === 'open' && audio.current?.state === 'running') {
              const ctx = audio.current, oscillator = ctx.createOscillator(), gain = ctx.createGain();
              oscillator.connect(gain); gain.connect(ctx.destination); oscillator.frequency.value = 660;
              gain.gain.setValueAtTime(.08,ctx.currentTime); gain.gain.exponentialRampToValueAtTime(.001,ctx.currentTime+.3);
              oscillator.start(); oscillator.stop(ctx.currentTime+.3);
            }
            seen.current.add(item.alert_id); setAlerts(previous => ({...previous,[item.alert_id]:item}));
          }
          if (message.type === 'source_status') { const s = message.data; setSources(previous => ({...previous,[sourceId(s)]:s})); }
          if (message.type === 'event') setEvents(previous => [message.data, ...previous.filter(e => e.event_id !== message.data.event_id)].slice(0,100));
        } catch (e) { setError((e as Error).message); }
      };
      socket.onclose = async () => {
        if (stopped) return;
        setConnected(false); setError('Nurse connection interrupted. Reconnecting…');
        try {
          const response = await fetch(new URL(`/nurse-sessions/${id}/snapshot`,base!), {signal:AbortSignal.timeout(4000)});
          if (stopped) return;
          if (response.status === 404) { remember(null); setNurse(null); setAlerts({}); setEvents([]); setSources({}); setError('Session expired. Start a new session.'); return; }
        } catch { /* Retry when the API returns. */ }
        if (!stopped) retry = setTimeout(connect,1000);
      };
    }
    connect();
    return () => { stopped = true; clearTimeout(retry); socket?.close(); };
  }, [base,nurse?.nurse_id]);
  async function save(preferences: NursePreferences) {
    if (!base || pending) return;
    const version = generation.current; setPending(true); setError('');
    try {
      const profile = await request<NurseProfile>(base,nurse ? `/nurse-sessions/${nurse.nurse_id}` : '/nurse-sessions', {method:nurse ? 'PUT':'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(preferences)});
      const data = await request<NurseSnapshot>(base,`/nurse-sessions/${profile.nurse_id}/snapshot`);
      if (version === generation.current) snapshot(data);
    } catch (e) { if (version === generation.current) setError((e as Error).message); }
    finally { if (version === generation.current) setPending(false); }
  }
  async function action(id: string, verb: 'accept'|'resolve') {
    if (!base || !nurse || !connected || pending) return;
    const version = generation.current; setPending(true); setError('');
    try {
      const item = await request<Alert>(base,`/alerts/${encodeURIComponent(id)}/${verb}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({nurse_id:nurse.nurse_id})});
      if (version === generation.current) setAlerts(previous => ({...previous,[id]:item}));
    } catch (e) {
      if (version !== generation.current) return;
      setError((e as Error).message);
      try { const data = await request<NurseSnapshot>(base,`/nurse-sessions/${nurse.nurse_id}/snapshot`); if (version === generation.current) snapshot(data); } catch {}
    } finally { if (version === generation.current) setPending(false); }
  }
  async function enableSound() {
    try { audio.current ??= new AudioContext(); await audio.current.resume(); setSound(true); }
    catch { setError('Audio unavailable. Visual alerts remain active.'); }
  }
  return {nurse,alerts,events,sources,connected,error,pending,sound,save,action,enableSound};
}
export type NurseSession = ReturnType<typeof useNurseSession>;

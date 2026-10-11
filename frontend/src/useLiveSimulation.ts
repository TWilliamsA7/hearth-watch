import { useEffect, useState } from 'react';
import { LiveClient } from '@hearth/room-view';
import { ensureLiveRoom, generateLiveRoom, liveCommand, liveSocketUrl, type RoomSettings } from './liveSimulation';

// The same staged incidents the Sim Studio offers; the engine picks a suitable person when none is selected.
export const INCIDENTS = {
  fall: { label: 'Fall', body: { kind: 'fall', duration: 30, recover: true } },
  inactivity: { label: 'Inactivity', body: { kind: 'inactivity', duration: 120 } },
  pacing: { label: 'Pacing', body: { kind: 'pacing', repeats: 6 } },
  exit_loiter: { label: 'Exit loitering', body: { kind: 'exit_loiter', repeats: 3 } },
} as const;
export type IncidentKind = keyof typeof INCIDENTS;
export const SPEEDS = [0.5, 1, 2, 4, 8] as const;

// `room` is the monitored room newly generated simulated rooms report as.
export function useLiveSimulation(base: string | null, room?: string) {
  const [session, setSession] = useState<{ base: string; client: LiveClient } | null>(null);
  const [, refresh] = useState(0);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    setError(''); setSelected(null); setBusy(false);
    if (!base) { setSession(null); return; }
    let disposed = false;
    let connected = false;
    const client = new LiveClient(liveSocketUrl(base));
    setSession({ base, client });
    const initialize = () => {
      void ensureLiveRoom(base).then(() => { if (!disposed) setError(''); })
        .catch(e => { if (!disposed) setError((e as Error).message); });
    };
    initialize();
    const unsubscribe = client.subscribe(() => {
      if (disposed) return;
      refresh(value => value + 1);
      if (client.connected && !connected) {
        initialize();
      }
      connected = client.connected;
    });
    return () => { disposed = true; unsubscribe(); client.close(); };
  }, [base, retry]);
  const client = session?.base === base ? session.client : null;
  const state = client?.latest?.state ?? (client?.world ? 'paused' : 'idle');
  const ready = !!client?.connected && !!client.world && state !== 'idle';
  const actor = client?.world?.actors.some(person => person.person_id === selected) ? selected : null;
  async function command(action: IncidentKind | 'reset' | 'toggle' | 'reset-person') {
    if (!base || !ready || busy) return;
    setBusy(true); setError('');
    try {
      if (action in INCIDENTS) {
        await liveCommand(base, 'incidents', { ...INCIDENTS[action as IncidentKind].body, actor });
        if (state !== 'playing') await liveCommand(base, 'play');
      } else if (action === 'reset-person') {
        if (actor) await liveCommand(base, `actors/${encodeURIComponent(actor)}/reset`);
      } else if (action === 'reset') {
        await liveCommand(base, 'reset');
        await liveCommand(base, 'play');
        setSelected(null);
      } else await liveCommand(base, state === 'playing' ? 'pause' : 'play');
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function setSpeed(speed: number) {
    if (!base || !ready || busy) return;
    setBusy(true); setError('');
    try { await liveCommand(base, 'speed', { speed }); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function generate(settings: RoomSettings, play: boolean): Promise<boolean> {
    if (!base || !client?.connected || busy) return false;
    setBusy(true); setError('');
    try {
      const spec = client.world?.spec;
      await generateLiveRoom(base, settings, play, spec && room ? { ...spec, room_id: room } : spec);
      setSelected(null);
      return true;
    } catch (e) { setError((e as Error).message); return false; }
    finally { setBusy(false); }
  }
  const speed = client?.latest?.speed ?? 1;
  const room_id = client?.world?.spec.room_id ?? null;
  return { client, ready, state, speed, room_id, error, busy, selected: actor, picked: selected, setSelected, command, setSpeed, generate, retry: () => setRetry(value => value + 1) };
}

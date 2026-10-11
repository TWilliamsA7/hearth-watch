import { useEffect, useState } from 'react';
import { LiveClient } from '@hearth/room-view';
import { ensureLiveRoom, liveCommand, liveSocketUrl } from './liveSimulation';

export function useLiveSimulation(base: string | null) {
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
  async function command(action: 'fall' | 'reset' | 'toggle') {
    if (!base || !ready || busy) return;
    setBusy(true); setError('');
    try {
      if (action === 'fall') {
        const actor = client?.world?.actors.some(person => person.person_id === selected) ? selected : null;
        await liveCommand(base, 'incidents', { kind: 'fall', actor, duration: 30, recover: true });
        if (state !== 'playing') await liveCommand(base, 'play');
      } else if (action === 'reset') {
        await liveCommand(base, 'reset');
        await liveCommand(base, 'play');
        setSelected(null);
      } else await liveCommand(base, state === 'playing' ? 'pause' : 'play');
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return { client, ready, state, error, busy, selected, setSelected, command, retry: () => setRetry(value => value + 1) };
}

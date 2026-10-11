import { useEffect, useState } from 'react';
import type { DisplayIdentity } from './identity';

/** Camera labels work even when the simulation is unlinked or its person limit is full. */
export function useIdentities(base: string, roomId: string): DisplayIdentity[] {
  const [snapshot, setSnapshot] = useState<{ key: string; people: DisplayIdentity[] } | null>(null);
  const key = `${base}\n${roomId}`;
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController;
    const poll = async () => {
      controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 3000);
      try {
        const response = await fetch(`${base.replace(/\/$/, '')}/rooms/${encodeURIComponent(roomId)}/identities`,
          { signal: controller.signal });
        if (!response.ok) throw new Error('Identity lookup unavailable');
        const people = await response.json() as DisplayIdentity[];
        if (alive) setSnapshot({ key, people });
      } catch {
        if (alive) setSnapshot({ key, people: [] });
      } finally {
        clearTimeout(timeout);
        if (alive) timer = setTimeout(poll, 1000);
      }
    };
    void poll();
    return () => { alive = false; clearTimeout(timer); controller?.abort(); };
  }, [base, roomId, key]);
  return snapshot?.key === key ? snapshot.people : [];
}

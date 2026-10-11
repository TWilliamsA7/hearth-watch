import { useEffect, useState } from 'react';
import type { CalibrateResult, LinkStatus } from '@hearth/room-view';
import { request } from './api';

// The live camera link (/sim/live/link): real people a camera sees, played out in the simulated room.
export function useLiveLink(base: string | null) {
  const [link, setLink] = useState<LinkStatus | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [calibrating, setCalibrating] = useState<'walk' | 'centre' | null>(null);
  const [calibration, setCalibration] = useState<CalibrateResult | null>(null);
  useEffect(() => {
    setLink(null);
    if (!base) return;
    let alive = true;
    const poll = () => request<LinkStatus>(base, '/sim/live/link').then(value => { if (alive) setLink(value); }).catch(() => { if (alive) setLink(null); });
    poll();
    const timer = setInterval(poll, 1000);
    return () => { alive = false; clearInterval(timer); };
  }, [base]);
  const post = <T,>(path: string, body: unknown, timeout = 5000) => request<T>(base!, path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }, timeout);
  async function setLinked(enabled: boolean, source: string, height: number, maxPeople?: number) {
    if (!base || busy) return;
    setBusy(true); setError('');
    try { setLink(await post<LinkStatus>('/sim/live/link', { enabled, source, height, max_people: maxPeople })); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  // Step "walk" (about 12 s) finds the real camera's height and tilt; "centre" (3 s) its direction.
  async function calibrate(step: 'walk' | 'centre', height: number) {
    if (!base || calibrating) return;
    setCalibrating(step); setCalibration(null); setError('');
    try { setCalibration(await post<CalibrateResult>('/sim/live/link/calibrate', { step, height }, 30000)); }
    catch (e) { setError((e as Error).message); }
    finally { setCalibrating(null); }
  }
  return { link, error, busy, calibrating, calibration, setLinked, calibrate };
}
export type LiveLink = ReturnType<typeof useLiveLink>;

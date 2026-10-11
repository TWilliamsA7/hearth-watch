import { request } from './api.ts';
import type { LiveStatus, LiveWorld } from '@hearth/room-view';

export function liveSocketUrl(base: string): string {
  const url = new URL('/sim/live/ws', base);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

export function liveCommand<T = LiveStatus>(base: string, action: string, body: unknown = {}): Promise<T> {
  return request<T>(base, `/sim/live/${action}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
}

// Share initialization across StrictMode mounts so a new room is generated only once.
const starting = new Map<string, Promise<void>>();
export function ensureLiveRoom(base: string): Promise<void> {
  const pending = starting.get(base);
  if (pending) return pending;
  const task = (async () => {
    let status: LiveStatus;
    try { status = await request<LiveStatus>(base, '/sim/live/status'); }
    catch (e) {
      if ((e as Error).message.includes('(404)')) {
        throw new Error('This server does not support the live 3D simulation. In Connection settings, connect to the standard monitoring backend. If it is running on port 8001, use http://127.0.0.1:8001.');
      }
      throw e;
    }
    if (status.state !== 'idle') return;
    await liveCommand<LiveWorld>(base, 'new', {
      seed: 7, play: true,
      spec: {
        actors: 6, room: { width: 7, depth: 6, furniture_density: 0.6 },
        camera: { preset: 'corner_left', height: 2.4, tilt: 12, fov: 62 },
        lighting: 'day', activity_frequency: 1, social_frequency: 1,
      },
    });
    await liveCommand(base, 'speed', { speed: 1 });
  })();
  starting.set(base, task);
  void task.finally(() => starting.delete(base)).catch(() => {});
  return task;
}

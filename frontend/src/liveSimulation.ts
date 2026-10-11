import { request } from './api.ts';
import type { EngineSpec, LiveStatus, LiveWorld } from '@hearth/room-view';

export type RoomSettings = {
  seed: number; actors: number; width: number; depth: number; furniture: number;
  camera: EngineSpec['camera']['preset']; cameraHeight: number; tilt: number; fov: number;
  lighting: EngineSpec['lighting']; activity: number; social: number;
};
export const DEFAULT_ROOM_SETTINGS: RoomSettings = {
  seed: 7, actors: 6, width: 7, depth: 6, furniture: 0.6,
  camera: 'corner_left', cameraHeight: 2.4, tilt: 12, fov: 62,
  lighting: 'day', activity: 1, social: 1,
};

export function roomSettings(world: LiveWorld): RoomSettings {
  const spec = world.spec;
  return {
    seed: world.seed, actors: spec.actors, width: spec.room.width ?? DEFAULT_ROOM_SETTINGS.width,
    depth: spec.room.depth ?? DEFAULT_ROOM_SETTINGS.depth, furniture: spec.room.furniture_density,
    camera: spec.camera.preset, cameraHeight: spec.camera.height, tilt: spec.camera.tilt, fov: spec.camera.fov,
    lighting: spec.lighting, activity: spec.activity_frequency, social: spec.social_frequency,
  };
}

export function generateLiveRoom(base: string, settings: RoomSettings, play: boolean, currentSpec?: EngineSpec) {
  return liveCommand<LiveWorld>(base, 'new', {
    seed: settings.seed, play,
    spec: {
      ...currentSpec, actors: settings.actors,
      room: { ...currentSpec?.room, width: settings.width, depth: settings.depth, furniture_density: settings.furniture },
      camera: { ...currentSpec?.camera, preset: settings.camera, height: settings.cameraHeight, tilt: settings.tilt, fov: settings.fov },
      lighting: settings.lighting, activity_frequency: settings.activity, social_frequency: settings.social,
    },
  });
}

export function liveSocketUrl(base: string): string {
  const url = new URL('/sim/live/ws', base);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

export function liveCommand<T = LiveStatus>(base: string, action: string, body: unknown = {}): Promise<T> {
  return request<T>(base, `/sim/live/${action}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }, action === 'new' || action === 'reset' ? 30000 : 5000);
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
    await generateLiveRoom(base, DEFAULT_ROOM_SETTINGS, true);
    await liveCommand(base, 'speed', { speed: 1 });
  })();
  starting.set(base, task);
  void task.finally(() => starting.delete(base)).catch(() => {});
  return task;
}

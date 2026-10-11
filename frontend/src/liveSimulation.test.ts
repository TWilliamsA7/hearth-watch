import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_ROOM_SETTINGS, ensureLiveRoom, generateLiveRoom, liveCommand, liveSocketUrl } from './liveSimulation.ts';

test('live socket uses the selected backend and supports HTTPS', () => {
  assert.equal(liveSocketUrl('http://127.0.0.1:8000'), 'ws://127.0.0.1:8000/sim/live/ws');
  assert.equal(liveSocketUrl('https://monitor.example'), 'wss://monitor.example/sim/live/ws');
});

test('startup preserves existing playback and creates only one room for simultaneous mounts', async () => {
  const original = globalThis.fetch;
  const calls: { path: string; body: any }[] = [];
  let state = 'paused';
  globalThis.fetch = async (input, options) => {
    const path = new URL(String(input)).pathname;
    calls.push({ path, body: options?.body ? JSON.parse(String(options.body)) : null });
    return new Response(JSON.stringify({ state }), { status: 200 });
  };
  try {
    await ensureLiveRoom('http://127.0.0.1:8000');
    assert.deepEqual(calls.map(call => call.path), ['/sim/live/status']);
    calls.length = 0;
    state = 'idle';
    await Promise.all([ensureLiveRoom('http://127.0.0.1:8000'), ensureLiveRoom('http://127.0.0.1:8000')]);
    assert.deepEqual(calls.map(call => call.path), ['/sim/live/status', '/sim/live/new', '/sim/live/speed']);
    assert.equal(calls[1].body.play, true);
    assert.equal(calls[1].body.seed, 7);
    assert.equal(calls[1].body.spec.actors, 6);
  } finally { globalThis.fetch = original; }
});

test('failed startup can be retried and commands report backend errors', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response('', { status: 404 });
  try {
    await assert.rejects(ensureLiveRoom('http://127.0.0.1:8000'), /does not support the live 3D simulation/);
    await assert.rejects(liveCommand('http://127.0.0.1:8000', 'incidents', { kind: 'fall' }), /404/);
    globalThis.fetch = async () => new Response(JSON.stringify({ state: 'playing' }));
    await ensureLiveRoom('http://127.0.0.1:8000');
  } finally { globalThis.fetch = original; }
});

test('room generation sends edited environment settings and preserves paused versus playing intent', async () => {
  const original = globalThis.fetch;
  const calls: { url: string; body: any }[] = [];
  globalThis.fetch = async (input, options) => {
    calls.push({ url: String(input), body: JSON.parse(String(options?.body)) });
    return new Response('{}');
  };
  try {
    const settings = {
      ...DEFAULT_ROOM_SETTINGS, seed: 42, actors: 3, width: 8, depth: 5, furniture: 0.25,
      camera: 'corner_right' as const, cameraHeight: 2.8, tilt: 20, fov: 70,
      lighting: 'evening' as const, activity: 2, social: 0.5,
    };
    await generateLiveRoom('http://127.0.0.1:8001', settings, false);
    await generateLiveRoom('http://127.0.0.1:8001', settings, true);
    assert.equal(calls[0].url, 'http://127.0.0.1:8001/sim/live/new');
    assert.equal(calls[0].body.play, false);
    assert.equal(calls[1].body.play, true);
    assert.equal(calls[0].body.seed, 42);
    assert.deepEqual(calls[0].body.spec, {
      actors: 3, room: { width: 8, depth: 5, furniture_density: 0.25 },
      camera: { preset: 'corner_right', height: 2.8, tilt: 20, fov: 70 },
      lighting: 'evening', activity_frequency: 2, social_frequency: 0.5,
    });
  } finally { globalThis.fetch = original; }
});

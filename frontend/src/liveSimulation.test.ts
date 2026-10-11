import test from 'node:test';
import assert from 'node:assert/strict';
import { ensureLiveRoom, liveCommand, liveSocketUrl } from './liveSimulation.ts';

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
    await assert.rejects(ensureLiveRoom('http://127.0.0.1:8000'), /404/);
    await assert.rejects(liveCommand('http://127.0.0.1:8000', 'incidents', { kind: 'fall' }), /404/);
    globalThis.fetch = async () => new Response(JSON.stringify({ state: 'playing' }));
    await ensureLiveRoom('http://127.0.0.1:8000');
  } finally { globalThis.fetch = original; }
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseMessage, validateApi } from './protocol.ts';
import { emptyFeed, latestSources, reduceFeed, visibleObservations } from './feedState.ts';
import { acknowledgeAlert, requestHelp, simCommand, websocketUrl } from './api.ts';

const fixture = JSON.parse(readFileSync(new URL('../contracts/edge.fixture.json', import.meta.url), 'utf8'));
const schema = JSON.parse(readFileSync(new URL('../contracts/monitoring.schema.json', import.meta.url), 'utf8'));
// Serialize a branch fixture as the backend does: absent defaults are included in output.
function defaults(name: string, values: Record<string, unknown>) {
  const output = { ...values };
  for (const [key, field] of Object.entries(schema.$defs[name].properties) as [string, { default?: unknown }][]) {
    if (!(key in output) && 'default' in field) output[key] = field.default;
  }
  return output;
}
const observation = defaults('Observation', fixture.observations[0]);
observation.features = defaults('PostureFeatures', observation.features as Record<string, unknown>);
const obs = parseMessage({ type: 'observation', data: observation })!;

test('actual edge fixture matches monitoring output schema and reaches the room view', () => {
  assert.ok(obs);
  const state = reduceFeed(emptyFeed(), obs, 1000, 'edge');
  assert.equal(visibleObservations(state, 1200, true).length, 1);
});
test('arrival clock expires observations despite old or future simulation record timestamps', () => {
  const state = reduceFeed(emptyFeed(), obs, 1000, 'edge');
  assert.equal(visibleObservations(state, 4000, true).length, 0);
  assert.equal(visibleObservations(state, 1200, false).length, 0);
});
test('a stale reconnect snapshot stays unknown even when received now', () => {
  let state = reduceFeed(emptyFeed(), obs, 1000, 'edge');
  const status = parseMessage({ type: 'source_status', data: defaults('SourceStatus', {
    source: observation.source, source_type: 'edge', room_id: 'room-01', online: true, observation_stale: true,
  }) });
  assert.ok(status);
  state = reduceFeed(state, status, 1100, 'edge');
  assert.equal(visibleObservations(state, 1200, true).length, 0);
});
test('live, simulation, and other rooms cannot contaminate the selected feed', () => {
  assert.deepEqual(reduceFeed(emptyFeed(), obs, 1000, 'simulation'), emptyFeed());
  const other = parseMessage({ type: 'observation', data: { ...observation, room_id: 'room-02' } })!;
  assert.deepEqual(reduceFeed(emptyFeed(), other, 1000, 'edge'), emptyFeed());
});
test('simulation person tracks and null face results remain distinct', () => {
  const { run_id, posture_prediction, hand_gesture_prediction, ...old } = observation;
  const one = parseMessage({ type: 'observation', data: { ...old, source_type: 'simulation', person_id: 'p1', face_present: null } })!;
  const two = parseMessage({ type: 'observation', data: { ...old, source_type: 'simulation', person_id: 'p2', obs_id: 'second', face_present: false } })!;
  assert.ok(one); assert.ok(two);
  const state = reduceFeed(reduceFeed(emptyFeed(), one, 1000, 'simulation'), two, 1100, 'simulation');
  const visible = visibleObservations(state, 1200, true);
  assert.equal(visible.length, 2);
  assert.deepEqual(visible.map(o => o.face_present), [null, false]);
});
const alert = defaults('Alert', { alert_id: 'incident-1', event_id: 'event-1', source: 'jetson-01', source_type: 'edge', room_id: 'room-01', opened_at: '2026-10-10T00:00:00Z', metadata: {} });
test('backend alert transitions replace existing alerts without duplication', () => {
  let state = emptyFeed();
  for (const status of ['open','acknowledged','resolved']) {
    const message = parseMessage({ type: 'alert', data: { ...alert, status } });
    assert.ok(message);
    state = reduceFeed(state, message, 1000, 'edge');
  }
  assert.equal(Object.keys(state.alerts).length, 1);
  assert.equal(state.alerts['incident-1'].status, 'resolved');
});
test('malformed messages are rejected instead of crashing the dashboard', () => {
  for (const value of [null, {}, { type: 'observation', data: {} }, { type: 'event', data: { event_type: 'fall_confirmed' } }]) assert.equal(parseMessage(value), null);
});
test('HTTPS backends use secure WebSockets', () => {
  assert.equal(websocketUrl('https://example.com'), 'wss://example.com/ws/room/room-01');
});
test('acknowledgement and scenario load use the actual branch endpoint and body', async () => {
  const original = globalThis.fetch;
  const calls: { url: string; options?: RequestInit }[] = [];
  const simSchema = JSON.parse(readFileSync(new URL('../contracts/simulation-api.schema.json', import.meta.url), 'utf8'));
  const status: Record<string, unknown> = { state: 'loaded', truth: [] };
  for (const [key, field] of Object.entries(simSchema.$defs.SimStatus.properties) as [string, { default?: unknown }][]) if ('default' in field) status[key] = field.default;
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), options });
    return new Response(JSON.stringify(calls.length === 1 ? { ...alert, status: 'acknowledged' } : status), { status: 200 });
  };
  try {
    await acknowledgeAlert('http://localhost:8000', 'incident-1');
    await simCommand('http://localhost:8000', 'load', 'possible-fall-01');
    assert.equal(calls[0].url, 'http://localhost:8000/alerts/incident-1/ack');
    assert.equal(calls[0].options?.method, 'POST');
    assert.equal(calls[1].url, 'http://localhost:8000/sim/load');
    assert.deepEqual(JSON.parse(String(calls[1].options?.body)), { scenario_id: 'possible-fall-01' });
  } finally { globalThis.fetch = original; }
});
test('missing simulator endpoints return a useful error', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response('', { status: 404 });
  try { await assert.rejects(simCommand('http://localhost:8000', 'play'), /not be mounted/); }
  finally { globalThis.fetch = original; }
});
test('actual monitoring WebSocket, alert and stale-health API output passes dashboard validation', () => {
  const messages = JSON.parse(readFileSync(new URL('../contracts/monitoring.integration.fixture.json', import.meta.url), 'utf8'));
  let state = emptyFeed();
  for (const value of messages) {
    const message = parseMessage(value);
    assert.ok(message, `Invalid production payload: ${value.type}`);
    state = reduceFeed(state, message, 1000, 'edge');
  }
  assert.ok(Object.values(state.alerts).some(alert => alert.kind === 'possible_fall' && alert.status === 'acknowledged'));
  assert.equal(visibleObservations(state, 1200, true).length, 0);
});
test('actual simulator router responses and after-action reports pass dashboard validation', () => {
  const data = JSON.parse(readFileSync(new URL('../contracts/simulation.integration.fixture.json', import.meta.url), 'utf8'));
  for (const value of data.catalog) validateApi('simApi#/$defs/ScenarioInfo', value);
  validateApi('simApi#/$defs/SimStatus', data.loaded);
  validateApi('simApi#/$defs/SimStatus', data.status);
  validateApi('simApi#/$defs/EvaluationReport', data.report);
  assert.ok(parseMessage({ type: 'observation', data: data.observation }));
});
test('manual resident help requests remain visible alongside each observation source type', () => {
  const message = parseMessage({ type: 'alert', data: { ...alert, kind: 'help_requested', source_type: 'manual', source: 'room-panel' } });
  assert.ok(message);
  for (const mode of ['edge', 'simulation', 'replay'] as const) {
    const state = reduceFeed(emptyFeed(), message, 1000, mode);
    assert.equal(state.alerts['incident-1'].kind, 'help_requested');
  }
});
test('resident help uses the real endpoint, room and retry-safe request id', async () => {
  const original = globalThis.fetch;
  const calls: { url: string; body: unknown }[] = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), body: JSON.parse(String(options?.body)) });
    return new Response(JSON.stringify({ ...alert, kind: 'help_requested', source_type: 'manual', source: 'room-panel' }));
  };
  try {
    const result = await requestHelp('http://localhost:8000', 'same-id-on-retry');
    assert.equal(result.kind, 'help_requested');
    assert.equal(calls[0].url, 'http://localhost:8000/help-requests');
    assert.deepEqual(calls[0].body, { request_id: 'same-id-on-retry', room_id: 'room-01' });
  } finally { globalThis.fetch = original; }
});
test('the room feed follows the chosen room', () => {
  const other = parseMessage({ type: 'observation', data: { ...observation, room_id: 'room-02' } })!;
  assert.deepEqual(reduceFeed(emptyFeed(), other, 1000, 'edge'), emptyFeed());
  assert.equal(visibleObservations(reduceFeed(emptyFeed(), other, 1000, 'edge', 'room-02'), 1200, true).length, 1);
  assert.equal(websocketUrl('http://127.0.0.1:8000', 'room 2'), 'ws://127.0.0.1:8000/ws/room/room%202');
});
test('alerts from any source stay in the list whichever feed tab is showing', () => {
  const live = parseMessage({ type: 'alert', data: { ...alert, status: 'open' } })!;
  assert.equal(reduceFeed(emptyFeed(), live, 1000, 'simulation').alerts['incident-1'].source_type, 'edge');
  const otherRoom = parseMessage({ type: 'alert', data: { ...alert, room_id: 'room-02', status: 'open' } })!;
  assert.deepEqual(reduceFeed(emptyFeed(), otherRoom, 1000, 'edge'), emptyFeed());
});
test("another run's stale health does not hide this run's observations", () => {
  const status = (changes: Record<string, unknown>) => parseMessage({ type: 'source_status', data: defaults('SourceStatus', {
    source: observation.source, source_type: 'edge', room_id: 'room-01', online: true, observation_stale: false,
    scenario_id: observation.scenario_id ?? null, run_id: observation.run_id ?? null, ...changes }) })!;
  let state = reduceFeed(emptyFeed(), obs, 1000, 'edge');
  state = reduceFeed(state, status({}), 1000, 'edge');
  // The same source name, but a different (ended) run: stale.
  state = reduceFeed(state, status({ scenario_id: 'other', run_id: 'old-run', online: false, observation_stale: true }), 1100, 'edge');
  assert.equal(visibleObservations(state, 1200, true).length, 1);
  assert.equal(latestSources(state).length, 1);
  assert.equal(latestSources(state)[0].online, true);
});

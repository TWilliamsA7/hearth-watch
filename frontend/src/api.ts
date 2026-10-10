import { validateApi } from './protocol.ts';
import type { Alert } from './generated/contract';
import type { ScenarioInfo, SimStatus, EvaluationReport, EvaluationSummary } from './generated/simulation';

export function websocketUrl(base: string): string {
  const url = new URL('/ws/room/room-01', base);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}
export async function request<T>(base: string, path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(new URL(path, base), { ...options, signal: AbortSignal.timeout(5000) });
  if (!response.ok) throw new Error(`Backend request failed (${response.status}).${response.status === 404 ? ' This endpoint may not be mounted yet.' : ''}`);
  return response.json() as Promise<T>;
}
export async function acknowledgeAlert(base: string, id: string): Promise<Alert> {
  return validateApi('monitoring#/$defs/Alert', await request(base, `/alerts/${encodeURIComponent(id)}/ack`, { method: 'POST' }));
}
export async function requestHelp(base: string, requestId: string): Promise<Alert> {
  return validateApi('monitoring#/$defs/Alert', await request(base, '/help-requests', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: requestId, room_id: 'room-01' }),
  }));
}
export async function scenarios(base: string): Promise<ScenarioInfo[]> {
  const values = await request<unknown>(base, '/sim/scenarios');
  if (!Array.isArray(values)) throw new Error('Invalid scenario list.');
  return values.map(value => validateApi<ScenarioInfo>('simApi#/$defs/ScenarioInfo', value));
}
export async function simStatus(base: string): Promise<SimStatus> {
  return validateApi('simApi#/$defs/SimStatus', await request(base, '/sim/status'));
}
export async function simCommand(base: string, action: 'load' | 'play' | 'pause' | 'reset', scenarioId?: string): Promise<SimStatus> {
  const options: RequestInit = { method: 'POST' };
  if (action === 'load') {
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify({ scenario_id: scenarioId });
  }
  return validateApi('simApi#/$defs/SimStatus', await request(base, `/sim/${action}`, options));
}
export async function evaluation(base: string, runId?: string): Promise<EvaluationReport | EvaluationSummary> {
  const path = runId ? `/eval/runs/${encodeURIComponent(runId)}` : '/eval/summary';
  return validateApi(`simApi#/$defs/${runId ? 'EvaluationReport' : 'EvaluationSummary'}`, await request(base, path));
}

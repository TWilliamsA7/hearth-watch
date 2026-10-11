import type { Alert, Event } from './generated/contract';
import type { Event as SimEvent } from './generated/simulationContract';
import type { FeedMessage, RoomObservation, RoomSource } from './protocol.ts';
import { DEFAULT_ROOM } from './api.ts';

export type SourceMode = 'edge' | 'simulation' | 'replay';
export type FeedState = {
  observations: Record<string, { data: RoomObservation; receivedAt: number }>;
  sources: Record<string, RoomSource>;
  events: (Event | SimEvent)[];
  // Events someone might act on, kept apart so the stream of face and visibility changes
  // (several a second per person) can't push them out.
  notable: (Event | SimEvent)[];
  alerts: Record<string, Alert>;
};
export const NOTABLE_EVENTS = new Set(['possible_fall', 'help_requested', 'appointment_overdue', 'recovered',
  'alert_acknowledged', 'alert_resolved', 'source_offline', 'source_online']);
export function emptyFeed(): FeedState { return { observations: {}, sources: {}, events: [], notable: [], alerts: {} }; }
const newestFirst = <T extends { timestamp: string }>(items: T[]) => items.sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp));
export function sourceKey(data: { source_type: string; source: string }): string { return `${data.source_type}:${data.source}`; }
// The server keeps health per source, scenario and run (one source can have several: a live
// room, an ended scripted run, a heartbeat-only entry). Match an observation to its own.
type Scoped = { source_type: string; source: string; scenario_id?: string | null; run_id?: string | null };
export function statusKey(data: Scoped): string {
  return `${sourceKey(data)}:${data.scenario_id ?? ''}:${data.run_id ?? ''}`;
}
// One entry per source for display: its most recently heard-from run.
export function latestSources(state: FeedState): RoomSource[] {
  const latest = new Map<string, RoomSource>();
  for (const status of Object.values(state.sources)) {
    const key = sourceKey(status), previous = latest.get(key);
    const seen = (s: RoomSource) => Date.parse(('last_seen' in s && s.last_seen) || '') || 0;
    if (!previous || Number(status.online) > Number(previous.online) || (status.online === previous.online && seen(status) >= seen(previous))) latest.set(key, status);
  }
  return [...latest.values()];
}
export function reduceFeed(state: FeedState, message: FeedMessage, now: number, mode: SourceMode, room = DEFAULT_ROOM): FeedState {
  if (message.type === 'nurse_snapshot') return state;
  const data = message.data;
  if (data.room_id !== room) return state;
  // Alerts, events and source health count every source; only the people on screen follow the tab.
  if (message.type === 'observation' && data.source_type !== mode) return state;
  const key = sourceKey(data);
  switch (message.type) {
    case 'observation': {
      const person = 'person_id' in message.data ? message.data.person_id : null;
      return { ...state, observations: { ...state.observations, [`${key}:${person ?? ''}`]: { data: message.data, receivedAt: now } } };
    }
    case 'source_status': return { ...state, sources: { ...state.sources, [statusKey(message.data)]: message.data } };
    case 'event': {
      const event = message.data, others = <T extends { event_id: string }>(list: T[]) => list.filter(e => e.event_id !== event.event_id);
      return { ...state, events: [event, ...others(state.events)].slice(0, 100),
        notable: NOTABLE_EVENTS.has(event.event_type) ? newestFirst([event, ...others(state.notable)]).slice(0, 200) : state.notable };
    }
    case 'alert': return { ...state, alerts: { ...state.alerts, [message.data.alert_id]: message.data } };
  }
}
// Use arrival time: simulation timestamps can intentionally differ from wall time.
// SourceStatus also invalidates old observations sent in a reconnect snapshot.
export function visibleObservations(state: FeedState, now: number, connected: boolean): RoomObservation[] {
  if (!connected) return [];
  return Object.values(state.observations).filter(({ data, receivedAt }) => {
    const status = state.sources[statusKey(data)];
    return now - receivedAt < 3000 && status?.online !== false &&
      !(status && 'observation_stale' in status && status.observation_stale);
  }).map(item => item.data);
}

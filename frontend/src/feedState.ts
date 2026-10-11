import type { Alert, Event } from './generated/contract';
import type { Event as SimEvent } from './generated/simulationContract';
import type { FeedMessage, RoomObservation, RoomSource } from './protocol.ts';
import { DEFAULT_ROOM } from './api.ts';

export type SourceMode = 'edge' | 'simulation' | 'replay';
export type FeedState = {
  observations: Record<string, { data: RoomObservation; receivedAt: number }>;
  sources: Record<string, RoomSource>;
  events: (Event | SimEvent)[];
  alerts: Record<string, Alert>;
};
export function emptyFeed(): FeedState { return { observations: {}, sources: {}, events: [], alerts: {} }; }
export function sourceKey(data: { source_type: string; source: string }): string { return `${data.source_type}:${data.source}`; }
export function reduceFeed(state: FeedState, message: FeedMessage, now: number, mode: SourceMode, room = DEFAULT_ROOM): FeedState {
  if (message.type === 'nurse_snapshot') return state;
  const data = message.data;
  const manualIncident = (data.source_type === 'manual' || data.source === 'appointments' || (message.type === 'alert' && message.data.kind === 'help_requested') || (message.type === 'event' && message.data.event_type === 'help_requested')) && (message.type === 'alert' || message.type === 'event');
  if (data.room_id !== room || (data.source_type !== mode && !manualIncident)) return state;
  const key = sourceKey(data);
  switch (message.type) {
    case 'observation': {
      const person = 'person_id' in message.data ? message.data.person_id : null;
      return { ...state, observations: { ...state.observations, [`${key}:${person ?? ''}`]: { data: message.data, receivedAt: now } } };
    }
    case 'source_status': return { ...state, sources: { ...state.sources, [key]: message.data } };
    case 'event': return { ...state, events: [message.data, ...state.events.filter(e => e.event_id !== message.data.event_id)].slice(0, 100) };
    case 'alert': return { ...state, alerts: { ...state.alerts, [message.data.alert_id]: message.data } };
  }
}
// Use arrival time: simulation timestamps can intentionally differ from wall time.
// SourceStatus also invalidates old observations sent in a reconnect snapshot.
export function visibleObservations(state: FeedState, now: number, connected: boolean): RoomObservation[] {
  if (!connected) return [];
  return Object.values(state.observations).filter(({ data, receivedAt }) => {
    const status = state.sources[sourceKey(data)];
    return now - receivedAt < 3000 && status?.online !== false &&
      !(status && 'observation_stale' in status && status.observation_stale);
  }).map(item => item.data);
}

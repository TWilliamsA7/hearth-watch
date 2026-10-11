import { useState } from 'react';
import type { FeedState } from './feedState';
import { sourceTypeLabels } from './alertText';

type FeedEvent = FeedState['events'][number];
type Tone = 'alert' | 'ok' | 'warn' | 'note';

// What happened that someone might act on, in words. Face and visibility changes arrive several
// times a second per person; they stay behind "All events".
const NOTABLE: Record<string, { tone: Tone; text: (who: string, e: FeedEvent) => string }> = {
  possible_fall: { tone: 'alert', text: who => `${who} may have fallen` },
  help_requested: { tone: 'alert', text: (who, e) => `${who} asked for help${e.metadata?.method === 'button' ? ' (button)' : e.metadata?.method === 'hand_gesture' ? ' (hand signal)' : ' (raised arm)'}` },
  appointment_overdue: { tone: 'warn', text: (who, e) => `${typeof e.metadata?.resident_label === 'string' ? e.metadata.resident_label : 'A resident'} has not checked in for their appointment` },
  recovered: { tone: 'ok', text: who => `${who} is upright again` },
  alert_acknowledged: { tone: 'note', text: () => 'Alert acknowledged' },
  alert_resolved: { tone: 'note', text: () => 'Alert resolved' },
  source_offline: { tone: 'warn', text: (_, e) => `${e.source} stopped sending` },
  source_online: { tone: 'ok', text: (_, e) => `${e.source} is sending again` },
};

function who(event: FeedEvent): string {
  const name = typeof event.metadata?.resident_label === 'string' ? event.metadata.resident_label : null;
  const person = 'person_id' in event && event.person_id ? event.person_id.toUpperCase() : null;
  return name ?? (person ? `${person}` : 'Someone');
}

export default function EventLog({ events }: { events: FeedEvent[] }) {
  const [all, setAll] = useState(false);
  const shown = (all ? events : events.filter(e => e.event_type in NOTABLE)).slice(0, 40);
  return <section className="event-log" aria-label="Event log">
    <div className="section-heading">
      <h2>What happened</h2>
      <label className="inline-check"><input type="checkbox" checked={all} onChange={e => setAll(e.target.checked)} /> All events</label>
    </div>
    {!shown.length ? <p className="empty">{all ? 'No events yet.' : 'Nothing notable yet. Falls, help requests, recoveries and camera outages appear here.'}</p>
      : <ol className="events">{shown.map(event => {
        const notable = NOTABLE[event.event_type];
        return <li key={event.event_id} className={notable ? `tone-${notable.tone}` : 'tone-quiet'}>
          <time>{new Date(event.timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}</time>
          <span className="what">{notable ? notable.text(who(event), event) : `${event.event_type.replaceAll('_', ' ')}${'person_id' in event && event.person_id ? ` · ${event.person_id.toUpperCase()}` : ''}`}</span>
          <span className="where">{sourceTypeLabels[event.source_type] ?? event.source_type} · {event.source}</span>
        </li>;
      })}</ol>}
  </section>;
}

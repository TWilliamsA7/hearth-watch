import type { RoomSource } from './protocol';
import { postureText, sourceHealth, sourceTypeLabels } from './alertText';

// Every camera, simulator or replay feeding the rooms in view: whether it is sending and what it sees.
export default function SourceHealth({ sources, scope }: { sources: RoomSource[]; scope: string }) {
  return <details className="observation-help">
    <summary>Source health · {sources.filter(s => s.online).length}/{sources.length} online</summary>
    {sources.length ? <ul className="timeline">{sources.map(s => <li key={`${s.room_id}:${s.source_type}:${s.source}`}>
      <strong>{s.room_id} · {s.source}</strong><span>{sourceTypeLabels[s.source_type] ?? s.source_type} · {sourceHealth(s)} · {postureText(s)}</span>
    </li>)}</ul> : <p className="small">No source reports {scope}.</p>}
  </details>;
}

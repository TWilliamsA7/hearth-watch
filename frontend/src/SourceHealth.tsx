import type { RoomSource } from './protocol';
import { postureText, sourceHealth, sourceTypeLabels } from './alertText';

// Every camera, simulator or replay reporting for this room: whether it is sending and what it sees.
export default function SourceHealth({ sources }: { sources: RoomSource[] }) {
  return <details className="source-health">
    <summary>Sources · {sources.filter(s => s.online).length}/{sources.length} online</summary>
    {sources.length ? <ul className="timeline">{sources.map(s => <li key={`${s.source_type}:${s.source}`}>
      <strong>{s.source}</strong><span>{sourceTypeLabels[s.source_type] ?? s.source_type} · {sourceHealth(s)} · {postureText(s)}</span>
    </li>)}</ul> : <p className="small">No source has reported yet.</p>}
  </details>;
}

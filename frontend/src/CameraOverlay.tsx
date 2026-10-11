import type { Alert } from './generated/contract';
import type { RoomObservation } from './protocol';
import { bones } from './RoomView';

// What the camera's pipeline sees, drawn over its video in the dashboard's colors: each person's
// skeleton, their posture-model call, and red when the monitor has an open alert for them.
// Sized like the <img> under it (object-fit: contain), so normalized keypoints line up.
const postureColor: Record<string, string> = { standing: 'var(--ok)', upright: 'var(--ok)', sitting: 'var(--amber)', fallen: 'var(--ember)', floor: 'var(--ember)' };
const postureName: Record<string, string> = { standing: 'Standing', upright: 'Upright', sitting: 'Sitting', fallen: 'On the floor', floor: 'On the floor' };

export default function CameraOverlay({ width, height, observations, alerts }: {
  width: number; height: number; observations: RoomObservation[]; alerts: Alert[];
}) {
  const scale = width / 640;
  return <svg className="camera-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
    {observations.filter(o => o.person_detected).map(observation => {
      const personId = 'person_id' in observation ? observation.person_id ?? null : null;
      const points = (observation.keypoints ?? []).filter(p => p.confidence === null || p.confidence >= .3);
      const prediction = 'posture_prediction' in observation ? observation.posture_prediction : null;
      const alert = alerts.find(a => a.status !== 'resolved' && a.source === observation.source && (a.metadata.person_id ?? null) === personId);
      const color = alert ? 'var(--ember)' : prediction ? postureColor[prediction.label] ?? 'var(--amber)' : 'var(--amber)';
      const top = points.length ? points.reduce((a, b) => (b.y < a.y ? b : a)) : null;
      const label = `${personId ? personId.toUpperCase() + ' · ' : ''}${prediction ? `${postureName[prediction.label] ?? prediction.label} ${prediction.confidence == null ? '' : Math.round(prediction.confidence * 100) + '%'}` : 'Pose only'}${alert ? ' · ALERT' : ''}`;
      const x = (v: number) => v * width, y = (v: number) => v * height;
      return <g key={`${observation.source}:${personId}`} stroke={color} strokeLinecap="round" strokeLinejoin="round">
        {bones.map(([a, b]) => {
          const start = points.find(p => p.name === a), end = points.find(p => p.name === b);
          return start && end ? <line key={`${a}-${b}`} x1={x(start.x)} y1={y(start.y)} x2={x(end.x)} y2={y(end.y)} strokeWidth={4 * scale} strokeOpacity=".9" /> : null;
        })}
        {points.map(p => <circle key={p.name} cx={x(p.x)} cy={y(p.y)} r={4 * scale} fill="var(--card)" strokeWidth={2.5 * scale} />)}
        {top && <g transform={`translate(${x(top.x)}, ${Math.max(22 * scale, y(top.y) - 18 * scale)})`} stroke="none">
          <rect x={-label.length * 4.4 * scale - 10 * scale} y={-17 * scale} width={label.length * 8.8 * scale + 20 * scale} height={24 * scale} rx={12 * scale} fill={color} />
          <text textAnchor="middle" y={0} fontSize={14 * scale} fontWeight="700" fill="var(--text)" style={{ fontFamily: 'Commissioner, system-ui, sans-serif' }}>{label}</text>
        </g>}
      </g>;
    })}
  </svg>;
}

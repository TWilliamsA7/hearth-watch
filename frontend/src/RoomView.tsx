import { useId } from 'react';
import type { RoomObservation } from './protocol';

const bones = [['nose','neck'],['neck','left_shoulder'],['neck','right_shoulder'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],['left_shoulder','left_hip'],['right_shoulder','right_hip'],['left_hip','right_hip'],['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle']];

export default function RoomView({ observations }: { observations: RoomObservation[] }) {
  const gridId = useId().replaceAll(':', '');
  const people = observations.filter(o => o.person_detected);
  return <svg viewBox="0 0 600 380" role="img" aria-label="Room 01 observation view">
    <defs><pattern id={gridId} width="25" height="25" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".9" fill="#cdd8c7" /></pattern></defs>
    <rect x="20" y="20" width="560" height="340" rx="18" fill="#eef3ee" stroke="#bacbc1" strokeWidth="2" />
    <rect x="20" y="20" width="560" height="340" rx="18" fill={`url(#${gridId})`} />
    {!observations.length ? <text x="300" y="190" textAnchor="middle" fill="#52645b" fontSize="26">No current observation · unknown</text> : !people.length ? <text x="300" y="190" textAnchor="middle" fill="#52645b" fontSize="26">Person not visible</text> : people.map((observation, index) => {
      const points = (observation.keypoints ?? []).filter(p => p.confidence === null || p.confidence >= .5);
      const prediction = 'posture_prediction' in observation ? observation.posture_prediction : null;
      const box = prediction?.bbox;
      return <g key={observation.obs_id} stroke={index % 2 ? '#a96536' : '#246b58'} strokeWidth="4" strokeLinecap="round">
        {box && <rect x={box.x1 * 600} y={box.y1 * 380} width={(box.x2 - box.x1) * 600} height={(box.y2 - box.y1) * 380} fill="none" strokeDasharray="8 5" />}
        {bones.map(([a,b]) => {
          const start = points.find(p => p.name === a), end = points.find(p => p.name === b);
          return start && end ? <line key={`${a}-${b}`} strokeDasharray={start.confidence === null || end.confidence === null ? '4 4' : undefined} x1={start.x * 600} y1={start.y * 380} x2={end.x * 600} y2={end.y * 380} /> : null;
        })}
        {points.map(p => <circle key={p.name} cx={p.x * 600} cy={p.y * 380} r="5" fill={p.confidence === null ? '#bacbc1' : '#fbfaf6'} />)}
        {!points.length && !box && <text x="300" y={160 + index * 30} textAnchor="middle" stroke="none" fill="#52645b" fontSize="24">Person detected · pose unavailable</text>}
      </g>;
    })}
  </svg>;
}

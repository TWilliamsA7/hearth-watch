import type { Observation } from './generated/contract';

const bones = [
  ['nose', 'neck'], ['neck', 'left_shoulder'], ['neck', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
];

export default function RoomView({ observation, offline }: { observation: Observation; offline: boolean }) {
  const points = observation.keypoints ?? [];
  return <svg viewBox="0 0 600 380" role="img" aria-label={offline ? 'Room observations unavailable' : 'Example person skeleton in one room'}>
    <rect x="20" y="20" width="560" height="340" rx="18" fill="#eef3ee" stroke="#bacbc1" strokeWidth="2" />
    <text x="40" y="50" fill="#61766c" fontSize="13">ROOM 01 · OBSERVATION VIEW</text>
    {offline ? <text x="300" y="190" textAnchor="middle" fill="#61766c">No current observation</text> : <g stroke="#246b58" strokeWidth="5" strokeLinecap="round">
      {bones.map(([a, b]) => {
        const start = points.find(p => p.name === a);
        const end = points.find(p => p.name === b);
        return start && end ? <line key={`${a}-${b}`} x1={start.x * 600} y1={start.y * 380} x2={end.x * 600} y2={end.y * 380} /> : null;
      })}
      {points.map(p => <circle key={p.name} cx={p.x * 600} cy={p.y * 380} r="5" fill="#fff" />)}
    </g>}
  </svg>;
}

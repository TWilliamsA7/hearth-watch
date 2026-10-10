import { useId } from 'react';
import type { RoomObservation } from './protocol';

const bones = [['nose','neck'],['neck','left_shoulder'],['neck','right_shoulder'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],['left_shoulder','left_hip'],['right_shoulder','right_hip'],['left_hip','right_hip'],['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle']];

export default function RoomView({ observations, illustrated = false, skeleton = false }: { observations: RoomObservation[]; illustrated?: boolean; skeleton?: boolean }) {
  const gridId = useId().replaceAll(':', '');
  const people = observations.filter(o => o.person_detected);
  const width = skeleton ? 200 : illustrated ? 640 : 600;
  const height = skeleton ? 240 : illustrated ? 360 : 380;
  const px = (x: number) => illustrated ? 230 + x * 220 : x * width;
  const py = (y: number) => illustrated ? 62 + y * 264 : y * height;
  return <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Room 01 observation view">
    <rect width={width} height={height} fill={'var(--room)'} />
    {illustrated ? <g aria-hidden="true">
      <rect x="0" y="305" width="640" height="55" fill="var(--floor)" />
      <rect x="440" y="50" width="130" height="120" rx="6" fill="color-mix(in srgb,var(--amber) 22%,var(--room))" stroke="var(--line)" strokeWidth="4" />
      <path d="M505 50v120M440 110h130" stroke="var(--line)" strokeWidth="3" />
      <rect x="30" y="235" width="170" height="75" rx="14" fill="color-mix(in srgb,var(--ember) 35%,var(--room))" />
      <rect x="22" y="215" width="40" height="95" rx="12" fill="color-mix(in srgb,var(--ember) 45%,var(--room))" />
      <ellipse cx="330" cy="330" rx="150" ry="14" fill="color-mix(in srgb,var(--amber) 25%,var(--floor))" />
      <rect x="580" y="250" width="22" height="60" rx="4" fill="color-mix(in srgb,var(--mute) 50%,var(--room))" />
      <circle cx="591" cy="238" r="26" fill="color-mix(in srgb,var(--ok) 55%,var(--room))" />
      <text x="14" y="24" fontSize="14" fill="var(--mute)">CAM 01 · LIVING ROOM · PREVIEW</text>
    </g> : <><defs><pattern id={gridId} width={width / 5} height={height / 6} patternUnits="userSpaceOnUse"><path d={`M${width / 5} 0H0V${height / 6}`} fill="none" stroke="var(--line)" strokeOpacity=".18" strokeWidth=".5" /></pattern></defs><rect width={width} height={height} fill={`url(#${gridId})`} /><line x1="0" x2={width} y1={height * .896} y2={height * .896} stroke="var(--floor)" strokeWidth="1.5" /></>}
    {!observations.length ? <text x={width / 2} y={height / 2} textAnchor="middle" fill="var(--mute)" fontSize={skeleton ? 14 : 20}>No current observation · unknown</text> : !people.length ? <text x={width / 2} y={height / 2} textAnchor="middle" fill="var(--mute)" fontSize={skeleton ? 14 : 20}>Person not visible</text> : people.map((observation, index) => {
      const points = (observation.keypoints ?? []).filter(p => p.confidence === null || p.confidence >= .5);
      const prediction = 'posture_prediction' in observation ? observation.posture_prediction : null;
      const box = prediction?.bbox;
      const previewBounds = illustrated && points.length ? {
        x: Math.min(...points.map(p => px(p.x))) - 11,
        y: Math.min(...points.map(p => py(p.y))) - 15,
        width: Math.max(...points.map(p => px(p.x))) - Math.min(...points.map(p => px(p.x))) + 22,
        height: Math.max(...points.map(p => py(p.y))) - Math.min(...points.map(p => py(p.y))) + 24,
      } : null;
      return <g key={observation.obs_id} stroke={index % 2 ? 'var(--ember)' : 'var(--amber)'} strokeWidth={skeleton ? 2.5 : 3} strokeLinecap="round">
        {previewBounds && <><rect {...previewBounds} fill="none" strokeWidth="1.5" strokeDasharray="5 3" /><text x={previewBounds.x} y={previewBounds.y - 4} fontSize="12" fill="var(--mute)" stroke="none">example person · {observation.features?.torso_angle_deg === 90 ? 'down' : 'upright'}</text></>}
        {box && <rect x={px(box.x1)} y={py(box.y1)} width={px(box.x2) - px(box.x1)} height={py(box.y2) - py(box.y1)} fill="none" strokeDasharray="8 5" />}
        {bones.map(([a,b]) => {
          const start = points.find(p => p.name === a), end = points.find(p => p.name === b);
          return start && end ? <line key={`${a}-${b}`} strokeDasharray={start.confidence === null || end.confidence === null ? '4 4' : undefined} x1={px(start.x)} y1={py(start.y)} x2={px(end.x)} y2={py(end.y)} /> : null;
        })}
        {points.map(p => <circle key={p.name} cx={px(p.x)} cy={py(p.y)} r={p.name === 'nose' ? 9 : 3} fill={p.name === 'nose' ? 'none' : p.confidence === null ? 'var(--mute)' : 'var(--amber)'} />)}
        {!points.length && !box && <text x={width / 2} y={height / 2 + index * 20} textAnchor="middle" stroke="none" fill="var(--mute)" fontSize={skeleton ? 14 : 20}>Person detected · pose unavailable</text>}
      </g>;
    })}
  </svg>;
}

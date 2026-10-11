import type { Observation, Event } from './generated/contract';

export type Preview = 'standing' | 'alert' | 'offline';

// These are UI examples, not simulator output or measured detection results.
export function exampleObservation(preview: Preview = 'standing'): Observation {
  const observation: Observation = {
    schema_version: '1', obs_id: 'preview-observation', timestamp: new Date().toISOString(),
    source: 'dashboard-preview', source_type: 'simulation', room_id: 'room-01', scenario_id: 'ui-preview',
    run_id: null, posture_prediction: null, hand_gesture_prediction: null,
    person_detected: preview !== 'offline', face_present: null, face_confidence: null,
    features: preview === 'offline' ? null : { torso_angle_deg: preview === 'alert' ? 90 : 0, head_y: preview === 'alert' ? .8 : .22, hip_y: preview === 'alert' ? .8 : .55, bbox_aspect: null, vertical_velocity: null },
    keypoints: [
      { name: 'nose', x: .5, y: .22, confidence: 1 },
      { name: 'neck', x: .5, y: .32, confidence: 1 },
      { name: 'left_shoulder', x: .42, y: .34, confidence: 1 },
      { name: 'right_shoulder', x: .58, y: .34, confidence: 1 },
      { name: 'left_elbow', x: .38, y: .46, confidence: 1 },
      { name: 'right_elbow', x: .62, y: .46, confidence: 1 },
      { name: 'left_wrist', x: .35, y: .56, confidence: 1 },
      { name: 'right_wrist', x: .65, y: .56, confidence: 1 },
      { name: 'left_hip', x: .45, y: .55, confidence: 1 },
      { name: 'right_hip', x: .55, y: .55, confidence: 1 },
      { name: 'left_knee', x: .43, y: .7, confidence: 1 },
      { name: 'right_knee', x: .57, y: .7, confidence: 1 },
      { name: 'left_ankle', x: .41, y: .84, confidence: 1 },
      { name: 'right_ankle', x: .59, y: .84, confidence: 1 }
    ]
  };
  if (preview === 'alert') observation.keypoints = observation.keypoints!.map(point => ({ ...point, x: .5 + (point.y - .55) * .8, y: .8 - (point.x - .5) * .7 }));
  if (preview === 'offline') observation.keypoints = null;
  return observation;
}

export function exampleEvent(event_type: Event['event_type']): Event {
  return {
    schema_version: '1', event_id: crypto.randomUUID(), timestamp: new Date().toISOString(),
    source: 'dashboard-preview', source_type: 'simulation', room_id: 'room-01',
    scenario_id: 'ui-preview', run_id: null, event_type, confidence: null, metadata: { ui_preview: true }
  };
}

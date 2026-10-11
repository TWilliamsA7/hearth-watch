import type { Alert } from './generated/contract';
import type { RoomSource } from './protocol';

// Wording shared by every alert list, so nurses read the same explanation everywhere.
const kindLabels: Record<string, string> = {
  possible_fall: 'Possible fall', help_requested: 'Help requested', appointment_overdue: 'Appointment check-in overdue',
  source_offline: 'Source offline', source_online: 'Source reconnected', recovered: 'Upright posture observed',
  alert_acknowledged: 'Accepted', alert_resolved: 'Resolved',
};
export const sourceTypeLabels: Record<string, string> = {
  edge: 'Live camera', simulation: 'Simulation', replay: 'Recorded replay', manual: 'Patient button',
};

export function kindLabel(kind: string): string {
  return kindLabels[kind] ?? kind.replaceAll('_', ' ');
}

function clock(value: unknown): string {
  return typeof value === 'string' ? new Date(value).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : 'the scheduled time';
}

// A name is shown only when the patient opted in to identifying their alerts.
export function residentName(alert: Alert): string | null {
  return alert.metadata.identity_demo === true && typeof alert.metadata.resident_label === 'string' ? alert.metadata.resident_label : null;
}

export function alertTitle(alert: Alert): string {
  if (alert.kind === 'help_requested') {
    const how = alert.metadata.method === 'button' ? 'Button request' : alert.metadata.method === 'hand_gesture' ? 'Hand signal' : 'Raised-arm signal';
    return `${how}${residentName(alert) ? ` · ${residentName(alert)}` : ''}`;
  }
  return `${kindLabel(alert.kind)}${residentName(alert) ? ` · ${residentName(alert)}` : ''}`;
}

// Only camera-raised alerts can carry an identity; say how sure it is.
export function identityLine(alert: Alert): string | null {
  if (alert.kind !== 'possible_fall' && !(alert.kind === 'help_requested' && alert.metadata.method !== 'button')) return null;
  if (!residentName(alert)) return 'Identity unconfirmed';
  return alert.metadata.identity_method === 'face_match' ? 'Face match confirmed' : 'Recent face match associated with this person';
}

export function alertReason(alert: Alert): string {
  switch (alert.kind) {
    case 'help_requested':
      return alert.metadata.method === 'button' ? 'A person pressed the room help button.'
        : alert.metadata.method === 'hand_gesture' ? 'A hand help signal was held toward the camera.'
        : 'A raised arm was held long enough to request assistance.';
    case 'possible_fall':
      return alert.metadata.reason === 'sustained_model_fallen' ? 'Sustained lying posture indicates a possible fall. Please check in.'
        : 'Sustained floor-level posture was observed. Please check in.';
    case 'appointment_overdue':
      return `${typeof alert.metadata.resident_label === 'string' ? alert.metadata.resident_label : 'A patient'}: arrival not confirmed for the appointment at ${clock(alert.metadata.scheduled_at)}. Please follow up.`;
    case 'source_offline':
      return 'This source stopped sending. The room cannot be verified through this source.';
    default:
      return kindLabel(alert.kind);
  }
}

export function recoveryText(alert: Alert): string | null {
  if (!alert.recovered_at) return null;
  return alert.kind === 'appointment_overdue' ? 'Arrival subsequently checked in. Review before resolving.'
    : alert.kind === 'source_offline' ? 'Source reconnected. Review before resolving.'
    : 'Upright posture observed afterward. Review before resolving.';
}

// What the source reports about the person, and whether it came from the posture model.
export function postureText(source: RoomSource): string {
  if (!source.online || ('observation_stale' in source && source.observation_stale) || !('posture' in source) || source.posture === 'unknown') return 'Posture unknown';
  if (!('posture_model' in source) || !source.posture_model) return `${source.posture === 'standing' ? 'Upright' : 'Floor-level posture'} (pose)`;
  const label = ({ standing: 'Standing', sitting: 'Sitting', fallen: 'Lying' } as Record<string, string>)[source.posture] ?? 'Unknown';
  const confidence = 'posture_confidence' in source && typeof source.posture_confidence === 'number' ? `, ${Math.round(source.posture_confidence * 100)}%` : '';
  return `${label} (model${confidence})`;
}

export function sourceHealth(source: RoomSource): string {
  return !source.online ? 'Offline' : 'observation_stale' in source && source.observation_stale ? 'Connected; observations stale' : 'Observations arriving';
}

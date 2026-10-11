import test from 'node:test';
import assert from 'node:assert/strict';
import type { Alert } from './generated/contract';
import { alertReason, alertTitle, identityLine, postureText, recoveryText } from './alertText.ts';

const alert = (kind: string, metadata: Record<string, unknown> = {}, extra: Partial<Alert> = {}) => ({
  alert_id: 'a', event_id: 'e', kind, source: 'jetson-01', source_type: 'edge', room_id: 'room-01', status: 'open',
  opened_at: '2026-10-10T12:00:00Z', recovered_at: null, metadata, ...extra,
}) as unknown as Alert;

test('help requests say how they were raised', () => {
  assert.equal(alertTitle(alert('help_requested', { method: 'button' })), 'Button request');
  assert.equal(alertTitle(alert('help_requested', { method: 'hand_gesture' })), 'Hand signal');
  assert.match(alertReason(alert('help_requested', { method: 'raised_arm' })), /raised arm/);
  assert.equal(identityLine(alert('help_requested', { method: 'button' })), null);
});

test('a name appears only with an opted-in identity', () => {
  assert.equal(alertTitle(alert('possible_fall', { resident_label: 'Alex' })), 'Possible fall');
  assert.equal(identityLine(alert('possible_fall')), 'Identity unconfirmed');
  const named = alert('possible_fall', { identity_demo: true, resident_label: 'Alex', identity_method: 'face_match' });
  assert.equal(alertTitle(named), 'Possible fall · Alex');
  assert.equal(identityLine(named), 'Face match confirmed');
});

test('overdue appointments name the patient and recoveries depend on the kind', () => {
  assert.match(alertReason(alert('appointment_overdue', { resident_label: 'Alex', scheduled_at: '2026-10-10T12:00:00Z' })), /^Alex: arrival not confirmed/);
  assert.equal(recoveryText(alert('possible_fall')), null);
  assert.match(recoveryText(alert('appointment_overdue', {}, { recovered_at: '2026-10-10T12:05:00Z' }))!, /checked in/);
  assert.match(recoveryText(alert('source_offline', {}, { recovered_at: '2026-10-10T12:05:00Z' }))!, /reconnected/);
});

test('posture text reports the model and its confidence', () => {
  const source = { source: 's', source_type: 'edge', room_id: 'room-01', online: true, observation_stale: false, posture: 'fallen', posture_confidence: 0.91, posture_model: 'hearth/posture-mlp' };
  assert.equal(postureText(source as never), 'Lying (model, 91%)');
  assert.equal(postureText({ ...source, posture_model: null, posture: 'standing' } as never), 'Upright (pose)');
  assert.equal(postureText({ ...source, online: false } as never), 'Posture unknown');
});

import { describe, expect, it, vi } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { identityFor, recognizedName, recordedName, type DisplayIdentity } from './identity';
import { liveDisplayName, liveNames, livePersonName, linkStateText, type LinkStatus } from './link';
import { drawRoom, type FrameState } from './draw';
import CameraOverlay from '../../src/CameraOverlay';
import PersonInspector from '../../src/PersonInspector';

const identity: DisplayIdentity = {
  room_id: 'room-01', source_type: 'edge', source: 'jetson-01', run_id: null, scenario_id: null,
  person_id: 'p1', timestamp: '2026-10-11T12:00:00Z', recognized: true, name: 'Alex',
};
const person = { person_id: 'live-jetson-01-p1', track_id: 'p1', state: 'live' as const,
  posture: 'standing', label: 'standing', model: null, alert_open: false, recognized: true, name: 'Alex' };
const link: LinkStatus = { enabled: true, source: 'jetson-01', max_people: 3, state: 'live',
  person_id: person.person_id, last_seen: identity.timestamp, posture: 'standing', label: 'standing',
  alert_open: false, camera: null, people: [person] };

describe('recognized display names', () => {
  it('uses the same name for avatar tags, roster, inspector and link status', () => {
    expect(liveNames(link)[person.person_id]).toBe('Alex');
    expect(livePersonName(person, link.source)).toBe('Alex');
    expect(liveDisplayName(person.person_id, link)).toBe('Alex');
    expect(linkStateText(link)).toBe('In the room: Alex standing');
  });

  it.each(['room_id', 'source', 'source_type', 'scenario_id', 'run_id', 'person_id'] as const)(
    'cannot transfer a name across %s', field => {
      expect(identityFor({ ...identity, [field]: 'different' }, [identity])).toBeUndefined();
    });

  it('supports omitted null fields without confusing anonymous and numbered tracks', () => {
    expect(recognizedName(identityFor({ ...identity, run_id: undefined }, [identity]))).toBe('Alex');
    expect(identityFor({ ...identity, person_id: null }, [identity])).toBeUndefined();
  });

  it('falls back to a track label when unnamed, expired or recognition is withdrawn', () => {
    for (const update of [{ name: null }, { recognized: false }, { name: '  ' }]) {
      const current = { ...link, people: [{ ...person, ...update }] };
      expect(liveDisplayName(person.person_id, current)).toBe('LIVE P1');
      expect(liveNames(current)).toEqual({});
    }
    expect(recognizedName(identityFor(identity, []))).toBeNull();
  });

  it('uses only the recorded, consent-filtered name for historical events', () => {
    expect(recordedName({ resident_label: 'Alex', identity_demo: true })).toBe('Alex');
    expect(recordedName({ resident_label: 'Alex' })).toBeNull();
    expect(recordedName({ person_id: 'p1' })).toBeNull();
  });

  it('renders named camera overlays and people buttons, with anonymous fallback', () => {
    const observation = { ...identity, obs_id: 'o1', person_detected: true, face_present: true,
      features: null, posture_prediction: null,
      keypoints: [{ name: 'nose', x: 0.5, y: 0.3, confidence: null }] };
    for (const identities of [[identity], [{ ...identity, name: null }], []]) {
      const expected = identities[0]?.name ? 'Alex' : 'P1';
      const overlay = renderToStaticMarkup(createElement(CameraOverlay, {
        width: 640, height: 480, observations: [observation], alerts: [], identities,
      } as never));
      expect(overlay).toContain(`${expected} · Pose only`);
      const inspector = renderToStaticMarkup(createElement(PersonInspector, {
        base: '', mode: 'edge', simulation: {}, link: null, identities, selectedKey: 'jetson-01:p1',
        people: [{ key: 'jetson-01:p1', personId: 'p1', source: 'jetson-01', observation }],
        alerts: [], onSelectKey: () => {}, observationFor: () => undefined,
      } as never));
      expect(inspector).toContain(`>${expected}</button>`);
    }
  });

  it('renders approved names on the Studio camera canvas and updates when withdrawn', () => {
    const text = vi.fn();
    const ctx = new Proxy({ fillText: text, measureText: () => ({ width: 30 }) }, {
      get: (target, key) => key in target ? target[key as keyof typeof target] : () => {},
    }) as unknown as CanvasRenderingContext2D;
    const frame: FrameState = { status: 'ok', sourceType: 'edge', personDetected: true, alertActive: false,
      people: [{ personId: 'p1', track: identity, alert: false, facePresent: false,
        keypoints: [{ name: 'nose', x: .5, y: .3, confidence: null }] }] };
    drawRoom(ctx, 640, 480, frame, undefined, undefined, [identity]);
    expect(text.mock.calls.some(call => call[0] === 'Alex')).toBe(true);
    text.mockClear();
    drawRoom(ctx, 640, 480, frame, undefined, undefined, []);
    expect(text.mock.calls.some(call => call[0] === 'Alex')).toBe(false);
    expect(text.mock.calls.some(call => call[0] === 'P1')).toBe(true);
  });
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { peopleInView, personLabel } from './people.ts';
import type { RoomObservation } from './protocol.ts';

const seen = (source: string, person: string | null, timestamp: string, detected = true) =>
  ({ source, source_type: 'edge', room_id: 'room-01', person_id: person, timestamp, person_detected: detected }) as unknown as RoomObservation;

test('one entry per person and source, using their newest observation', () => {
  const people = peopleInView([
    seen('jetson-01', 'p1', '2026-10-10T00:00:01Z'), seen('jetson-01', 'p1', '2026-10-10T00:00:02Z'),
    seen('jetson-01', 'p2', '2026-10-10T00:00:01Z'), seen('jetson-01', 'p3', '2026-10-10T00:00:01Z', false),
    seen('fall-model-01', null, '2026-10-10T00:00:01Z'),
  ]);
  assert.deepEqual(people.map(p => p.key), ['fall-model-01:', 'jetson-01:p1', 'jetson-01:p2']);
  assert.equal(people[1].observation.timestamp, '2026-10-10T00:00:02Z');
  assert.deepEqual(people.map(personLabel), ['fall-model-01', 'P1', 'P2']);
});

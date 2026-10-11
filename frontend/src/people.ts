import type { RoomObservation } from './protocol';

// One entry per person a camera or replay currently reports, for the person inspector.
export type TrackedPerson = { key: string; personId: string | null; source: string; observation: RoomObservation };

export function personKey(observation: RoomObservation): string {
  const personId = 'person_id' in observation ? observation.person_id ?? null : null;
  return `${observation.source}:${personId ?? ''}`;
}

export function peopleInView(observations: RoomObservation[]): TrackedPerson[] {
  const latest = new Map<string, RoomObservation>();
  for (const observation of observations) {
    if (!observation.person_detected) continue;
    const key = personKey(observation);
    const previous = latest.get(key);
    if (!previous || Date.parse(observation.timestamp) >= Date.parse(previous.timestamp)) latest.set(key, observation);
  }
  return [...latest.entries()]
    .map(([key, observation]) => ({ key, personId: 'person_id' in observation ? observation.person_id ?? null : null, source: observation.source, observation }))
    .sort((a, b) => a.key.localeCompare(b.key));
}

export function personLabel(person: TrackedPerson): string {
  return person.personId ? person.personId.toUpperCase() : person.source;
}

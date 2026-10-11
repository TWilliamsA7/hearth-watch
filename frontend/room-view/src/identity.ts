/** Identity comes from the backend's consent-filtered lookup, never from a track ID. */
export interface IdentityTrack {
  room_id: string;
  source_type: string;
  source: string;
  scenario_id?: string | null;
  run_id?: string | null;
  person_id?: string | null;
}

export interface DisplayIdentity extends IdentityTrack {
  timestamp: string;
  recognized: boolean;
  name: string | null;
}

export const identityKey = (track: IdentityTrack): string => JSON.stringify([
  track.room_id, track.source_type, track.source, track.scenario_id ?? null,
  track.run_id ?? null, track.person_id ?? null,
]);

export const identityFor = (track: IdentityTrack, identities: readonly DisplayIdentity[]) =>
  identities.find(identity => identity.recognized && identityKey(identity) === identityKey(track));

export const recognizedName = (identity: { recognized?: boolean; name?: string | null } | undefined): string | null =>
  identity?.recognized ? identity.name?.trim() || null : null;

/** Historical events retain the identity recorded then; never relabel from today's track. */
export const recordedName = (metadata: Record<string, unknown> | undefined | null): string | null =>
  metadata?.identity_demo === true && typeof metadata.resident_label === 'string'
    ? metadata.resident_label.trim() || null : null;

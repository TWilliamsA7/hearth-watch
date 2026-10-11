import type { FeedStatus, FrameState, PersonFrame } from "./draw";
import { interpolateKeypoints } from "./skeleton";
import type { Observation, ServerMessage, SourceType } from "./types";

/** The last two sightings of one person (for smoothing). Keyed by person_id, or "" for a single-person source. */
export interface PersonTrack {
  prev: Observation | null;
  last: Observation;
  /** Feed clock (ms) when `last` arrived. */
  receivedAt: number;
}

/** What the view remembers between messages. Pure data; updated by reduce(). */
export interface ViewModel {
  people: Record<string, PersonTrack>;
  /** Feed clock (ms) of the latest observation of any kind: the source's pulse. */
  lastAt: number | null;
  /** Whether the latest observation saw anyone at all. */
  anyoneVisible: boolean;
  source: string | null;
  sourceType: SourceType | null;
  /** From the backend's source_status: false = offline, null = no word yet. */
  online: boolean | null;
}

export const INITIAL_VIEW: ViewModel = {
  people: {},
  lastAt: null,
  anyoneVisible: false,
  source: null,
  sourceType: null,
  online: null,
};

export interface ViewFilter {
  roomId?: string;
  /** Only show this source (e.g. "sim" or "jetson-01"); default: any source in the room. */
  source?: string;
}

const key = (obs: Observation) => obs.person_id ?? "";

export function reduce(vm: ViewModel, msg: ServerMessage, receivedAt: number, filter: ViewFilter = {}): ViewModel {
  const { roomId, source } = filter;
  if (msg.type === "nurse_snapshot") return vm; // a care-desk message, not about the camera
  if (roomId && msg.data.room_id !== roomId) return vm;
  if (source && msg.data.source !== source) return vm;

  if (msg.type === "observation") {
    const obs = msg.data;
    // An arriving observation is proof of life; let staleness decide from here on.
    const base = { ...vm, lastAt: receivedAt, source: obs.source, sourceType: obs.source_type, online: null };
    if (!obs.person_detected || !obs.keypoints?.length) {
      return { ...base, people: {}, anyoneVisible: false }; // the camera sees nobody
    }
    const before = vm.people[key(obs)];
    return {
      ...base,
      anyoneVisible: true,
      people: { ...vm.people, [key(obs)]: { prev: before?.last ?? null, last: obs, receivedAt } },
    };
  }
  if (msg.type === "source_status") {
    if (vm.source && msg.data.source !== vm.source) return vm;
    return { ...vm, online: msg.data.online ? null : false };
  }
  return vm;
}

export interface FrameOptions {
  staleAfterMs: number;
  alertActive: boolean;
  /** person_ids with an open alert. */
  alertPersonIds?: readonly string[];
  smoothing: boolean;
  /** Drop a person not reported for this long while the source keeps sending (they left or are hidden). */
  personTimeoutMs?: number;
}

export function statusAt(vm: ViewModel, now: number, staleAfterMs: number): FeedStatus {
  if (vm.online === false) return "offline";
  if (vm.lastAt === null) return "waiting";
  return now - vm.lastAt > staleAfterMs ? "stale" : "ok";
}

function personFrame(track: PersonTrack, now: number, opts: FrameOptions): PersonFrame {
  const { last, prev } = track;
  let keypoints = last.keypoints ?? [];
  if (opts.smoothing && prev?.person_detected) {
    const interval = Date.parse(last.timestamp) - Date.parse(prev.timestamp);
    if (interval > 0 && interval <= 500) {
      keypoints = interpolateKeypoints(prev.keypoints, last.keypoints, (now - track.receivedAt) / interval) ?? [];
    }
  }
  return {
    personId: last.person_id ?? null, // older producers omit the field entirely
    keypoints,
    facePresent: last.face_present,
    alert: last.person_id != null && (opts.alertPersonIds ?? []).includes(last.person_id),
  };
}

/** The frame to draw at feed time `now`, smoothing each person between their last two sightings. */
export function frameAt(vm: ViewModel, now: number, opts: FrameOptions): FrameState {
  const timeout = opts.personTimeoutMs ?? 600;
  // Measured against the source's latest pulse, not `now`: if the whole feed goes quiet the
  // last people stay on screen (faded by the stale status) instead of vanishing.
  const people =
    vm.lastAt === null
      ? []
      : Object.values(vm.people)
          .filter((t) => vm.lastAt! - t.receivedAt <= timeout)
          .map((t) => personFrame(t, now, opts));
  return {
    people,
    personDetected: vm.lastAt === null ? null : vm.anyoneVisible || people.length > 0,
    sourceType: vm.sourceType,
    status: statusAt(vm, now, opts.staleAfterMs),
    alertActive: opts.alertActive,
  };
}

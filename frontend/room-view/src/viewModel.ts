import type { FeedStatus, FrameState } from "./draw";
import { interpolateKeypoints } from "./skeleton";
import type { Observation, ServerMessage } from "./types";

/** What the view remembers between messages. Pure data; updated by reduce(). */
export interface ViewModel {
  prev: Observation | null;
  last: Observation | null;
  /** Feed clock (ms) when `last` arrived. */
  receivedAt: number | null;
  /** From the backend's source_status: false = offline, null = no word yet. */
  online: boolean | null;
}

export const INITIAL_VIEW: ViewModel = { prev: null, last: null, receivedAt: null, online: null };

export interface ViewFilter {
  roomId?: string;
  /** Only show this source (e.g. "sim" or "jetson-01"); default: any source in the room. */
  source?: string;
}

export function reduce(vm: ViewModel, msg: ServerMessage, receivedAt: number, filter: ViewFilter = {}): ViewModel {
  const { roomId, source } = filter;
  if (roomId && msg.data.room_id !== roomId) return vm;
  if (source && msg.data.source !== source) return vm;

  if (msg.type === "observation") {
    // An arriving observation is proof of life; let staleness decide from here on.
    return { prev: vm.last, last: msg.data, receivedAt, online: null };
  }
  if (msg.type === "source_status") {
    if (vm.last && msg.data.source !== vm.last.source) return vm;
    return { ...vm, online: msg.data.online ? null : false };
  }
  return vm;
}

export interface FrameOptions {
  staleAfterMs: number;
  alertActive: boolean;
  smoothing: boolean;
}

export function statusAt(vm: ViewModel, now: number, staleAfterMs: number): FeedStatus {
  if (vm.online === false) return "offline";
  if (vm.receivedAt === null) return "waiting";
  return now - vm.receivedAt > staleAfterMs ? "stale" : "ok";
}

/** The frame to draw at feed time `now`, smoothing between the last two observations. */
export function frameAt(vm: ViewModel, now: number, opts: FrameOptions): FrameState {
  const { last, prev } = vm;
  let keypoints = last?.keypoints ?? null;
  if (opts.smoothing && last && prev && vm.receivedAt !== null && prev.person_detected && last.person_detected) {
    const interval = Date.parse(last.timestamp) - Date.parse(prev.timestamp);
    if (interval > 0 && interval <= 500) {
      keypoints = interpolateKeypoints(prev.keypoints, last.keypoints, (now - vm.receivedAt) / interval);
    }
  }
  return {
    keypoints,
    personDetected: last ? last.person_detected : null,
    facePresent: last ? last.face_present : null,
    sourceType: last?.source_type ?? null,
    status: statusAt(vm, now, opts.staleAfterMs),
    alertActive: opts.alertActive,
  };
}

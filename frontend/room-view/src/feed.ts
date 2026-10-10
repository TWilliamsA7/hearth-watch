import type { Heartbeat, Observation, ServerMessage } from "./types";

/**
 * Anything that delivers dashboard messages. The live backend WebSocket and a recorded
 * run both implement it, so the view cannot tell them apart except by the source_type
 * the records themselves carry.
 *
 * Times are on the feed's own clock, in epoch milliseconds: wall time for a live socket,
 * playback time for a recording. receivedAt is when the message arrived on that clock
 * (for a recording, when it was originally due, even if delivered late by a seek), and
 * now() is the current time. The view measures staleness as now() - receivedAt.
 */
export type FeedListener = (msg: ServerMessage, receivedAt: number) => void;

export interface FeedSource {
  subscribe(listener: FeedListener): () => void;
  now(): number;
}

class Listeners {
  private set = new Set<FeedListener>();
  add(listener: FeedListener): () => void {
    this.set.add(listener);
    return () => this.set.delete(listener);
  }
  emit(msg: ServerMessage, receivedAt: number): void {
    for (const l of this.set) l(msg, receivedAt);
  }
}

// --- live: WS /ws/room/{room_id} ---

export interface WebSocketFeed extends FeedSource {
  close(): void;
}

/** Connects to the backend feed and reconnects with backoff. Malformed frames are dropped. */
export function websocketFeed(url: string, opts: { maxBackoffMs?: number } = {}): WebSocketFeed {
  const listeners = new Listeners();
  const maxBackoff = opts.maxBackoffMs ?? 5000;
  let socket: WebSocket | null = null;
  let backoff = 250;
  let closed = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const connect = () => {
    socket = new WebSocket(url);
    socket.onopen = () => {
      backoff = 250;
    };
    socket.onmessage = (ev) => {
      try {
        const msg = JSON.parse(String(ev.data)) as ServerMessage;
        if (msg && typeof msg === "object" && "type" in msg && "data" in msg) listeners.emit(msg, Date.now());
      } catch {
        /* ignore malformed frames */
      }
    };
    socket.onclose = () => {
      if (closed) return;
      timer = setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, maxBackoff);
    };
  };
  connect();

  return {
    subscribe: (l) => listeners.add(l),
    now: () => Date.now(),
    close() {
      closed = true;
      clearTimeout(timer);
      socket?.close();
    },
  };
}

// --- recorded: <run>.stream.jsonl from `python -m hearth.sim run` ---

export type StreamRecord = { type: "observation"; data: Observation } | { type: "heartbeat"; data: Heartbeat };

/** Parse a stream file. Lines are {"type": "observation" | "heartbeat", "data": ...}. */
export function parseStream(text: string): StreamRecord[] {
  return text
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as StreamRecord)
    .filter((r) => r.type === "observation" || r.type === "heartbeat");
}

export interface PlaybackState {
  playing: boolean;
  speed: number;
  /** Seconds since the start of the recording. */
  t: number;
  duration: number;
}

/**
 * Plays a recorded stream as if it were arriving live. Observations are emitted as
 * ServerMessages. Heartbeats are not: on the live path the backend turns them into
 * source_status, and the view already treats silence as unknown.
 *
 * Time only moves through tick(), which the play loop calls; tests call it directly.
 */
export class RecordedFeed implements FeedSource {
  private listeners = new Listeners();
  private stateListeners = new Set<(s: PlaybackState) => void>();
  private records: { at: number; msg: ServerMessage | null }[];
  private cursor = 0;
  private startMs: number;
  private elapsedMs = 0;
  private playing = false;
  private speed = 1;
  private raf: number | ReturnType<typeof setTimeout> | null = null;
  private lastFrame: number | null = null;
  readonly duration: number;

  constructor(records: StreamRecord[], opts: { duration?: number; start?: string } = {}) {
    const times = records.map((r) => Date.parse(r.data.timestamp));
    this.startMs = opts.start ? Date.parse(opts.start) : Math.min(...times);
    this.records = records
      .map((r, i) => ({
        at: times[i] - this.startMs,
        msg: r.type === "observation" ? ({ type: "observation", data: r.data } as ServerMessage) : null,
      }))
      .sort((a, b) => a.at - b.at);
    const last = this.records.length ? this.records[this.records.length - 1].at : 0;
    this.duration = opts.duration ?? last / 1000;
  }

  subscribe(listener: FeedListener): () => void {
    return this.listeners.add(listener);
  }

  onState(listener: (s: PlaybackState) => void): () => void {
    this.stateListeners.add(listener);
    listener(this.state);
    return () => this.stateListeners.delete(listener);
  }

  now(): number {
    return this.startMs + this.elapsedMs;
  }

  get state(): PlaybackState {
    return { playing: this.playing, speed: this.speed, t: this.elapsedMs / 1000, duration: this.duration };
  }

  /** Advance playback by real milliseconds (scaled by speed) and emit what is now due. */
  tick(realMs: number): void {
    this.advanceTo(this.elapsedMs + realMs * this.speed);
  }

  play(): void {
    if (this.playing) return;
    if (this.elapsedMs >= this.duration * 1000) this.seek(0);
    this.playing = true;
    this.lastFrame = null;
    this.loop();
    this.notify();
  }

  pause(): void {
    this.playing = false;
    this.cancel();
    this.notify();
  }

  setSpeed(speed: number): void {
    this.speed = speed;
    this.notify();
  }

  /** Jump to t seconds and re-emit the most recent observation at or before it. */
  seek(t: number): void {
    const target = Math.min(Math.max(t, 0), this.duration) * 1000;
    this.cursor = 0;
    let latest: { at: number; msg: ServerMessage | null } | null = null;
    while (this.cursor < this.records.length && this.records[this.cursor].at <= target) {
      if (this.records[this.cursor].msg) latest = this.records[this.cursor];
      this.cursor++;
    }
    this.elapsedMs = target;
    if (latest?.msg) this.listeners.emit(latest.msg, this.startMs + latest.at);
    this.notify();
  }

  dispose(): void {
    this.cancel();
    this.stateListeners.clear();
  }

  private advanceTo(ms: number): void {
    const end = this.duration * 1000;
    this.elapsedMs = Math.min(ms, end);
    while (this.cursor < this.records.length && this.records[this.cursor].at <= this.elapsedMs) {
      const { at, msg } = this.records[this.cursor++];
      if (msg) this.listeners.emit(msg, this.startMs + at);
    }
    if (this.elapsedMs >= end && this.playing) {
      this.playing = false;
      this.cancel();
    }
    this.notify();
  }

  private loop = (): void => {
    if (!this.playing) return;
    const frame = (ts: number) => {
      if (this.lastFrame !== null) this.tick(ts - this.lastFrame);
      this.lastFrame = ts;
      this.loop();
    };
    this.raf =
      typeof requestAnimationFrame === "function"
        ? requestAnimationFrame(frame)
        : setTimeout(() => frame(performance.now()), 16);
  };

  private cancel(): void {
    if (this.raf === null) return;
    if (typeof cancelAnimationFrame === "function" && typeof this.raf === "number") cancelAnimationFrame(this.raf);
    else clearTimeout(this.raf as ReturnType<typeof setTimeout>);
    this.raf = null;
  }

  private notify(): void {
    const s = this.state;
    for (const l of this.stateListeners) l(s);
  }
}

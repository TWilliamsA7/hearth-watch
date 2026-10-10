import type { LiveFrame, LivePerson, LiveWorld } from "./generated/sim";
import type { WorldTrack } from "./types";
import { sampleWorld } from "./world";

/**
 * Client for the live engine's WebSocket (WS /sim/live/ws): the room once, then frames
 * about 15 times a second. It keeps a short buffer of frames and renders a little behind
 * the newest one, blending each person between the frames around the render time, so
 * movement is smooth whatever the frame rate and network jitter.
 *
 * Everything it carries is simulator-only ground truth for people watching (where each
 * person really is, what they're doing); none of it is what the camera reports.
 */

export interface LiveSample {
  t: number;
  people: LivePerson[];
  frame: LiveFrame | null;
}

type Listener = () => void;

const BUFFER = 64;
const DELAY_S = 0.12; // render this far behind the newest frame (scenario seconds at 1x)

export class LiveClient {
  world: LiveWorld | null = null;
  frames: LiveFrame[] = [];
  connected = false;
  private socket: WebSocket | null = null;
  private closed = false;
  private backoff = 250;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private listeners = new Set<Listener>();
  private clock: { t: number; at: number } | null = null; // render clock, smoothed
  private worldRev = 0;

  constructor(private url: string) {
    this.connect();
  }

  /** Called on a new room, on connection changes and on every frame. */
  subscribe(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  /** Bumps when the room changes (a new world), for views that rebuild on it. */
  get revision(): number {
    return this.worldRev;
  }

  get latest(): LiveFrame | null {
    return this.frames.length ? this.frames[this.frames.length - 1] : null;
  }

  /** Feed a message by hand (tests, or another transport). */
  receive(msg: { type: string; data: unknown }, now = performance.now()): void {
    if (msg.type === "world") {
      this.world = msg.data as LiveWorld;
      this.frames = [];
      this.clock = null;
      this.worldRev++;
    } else if (msg.type === "frame") {
      const f = msg.data as LiveFrame;
      const last = this.latest;
      if (last && f.t < last.t - 1e-6) {
        this.frames = []; // time went back: a reset
        this.clock = null;
      }
      this.frames.push(f);
      if (this.frames.length > BUFFER) this.frames.shift();
      if (!this.clock) this.clock = { t: f.t, at: now };
    }
    for (const l of this.listeners) l();
  }

  /** The scenario time to draw now: just behind the newest frame, advancing smoothly. */
  renderTime(now = performance.now()): number {
    const last = this.latest;
    if (!last || !this.clock) return 0;
    const playing = last.state === "playing";
    const speed = last.speed || 1;
    let t = this.clock.t + (playing ? ((now - this.clock.at) / 1000) * speed : 0);
    const target = last.t - (playing ? DELAY_S * speed : 0);
    // Ease towards the target so the clock neither stalls nor jumps.
    t += (target - t) * (Math.abs(target - t) > 1.5 ? 1 : 0.08);
    t = Math.min(t, last.t);
    this.clock = { t, at: now };
    return t;
  }

  /** Everyone at scenario time t, blended between the frames around it. */
  sample(t: number): LiveSample {
    const frames = this.frames;
    if (!frames.length) return { t, people: [], frame: null };
    const track = { frames } as unknown as WorldTrack;
    const blended = sampleWorld(track, t);
    // Who's doing what: from the newest frame at or before t.
    let k = frames.length - 1;
    while (k > 0 && frames[k].t > t) k--;
    const meta = new Map(frames[k].people.map((p) => [p.person_id, p]));
    const later = new Map(frames[Math.min(k + 1, frames.length - 1)].people.map((p) => [p.person_id, p]));
    const people = blended.people.map((p) => {
      const m = meta.get(p.person_id) ?? later.get(p.person_id);
      return { ...(m as LivePerson), ...p } as LivePerson;
    });
    return { t, people, frame: frames[k] };
  }

  close(): void {
    this.closed = true;
    clearTimeout(this.timer);
    this.socket?.close();
  }

  private connect(): void {
    if (typeof WebSocket === "undefined") return;
    const socket = new WebSocket(this.url);
    this.socket = socket;
    socket.onopen = () => {
      this.backoff = 250;
      this.connected = true;
      for (const l of this.listeners) l();
    };
    socket.onmessage = (ev) => {
      try {
        this.receive(JSON.parse(String(ev.data)));
      } catch {
        /* ignore malformed frames */
      }
    };
    socket.onclose = () => {
      this.connected = false;
      for (const l of this.listeners) l();
      if (this.closed) return;
      this.timer = setTimeout(() => this.connect(), this.backoff);
      this.backoff = Math.min(this.backoff * 2, 5000);
    };
  }
}

/** ws(s)://<this host><path>, for pages served through the dev proxy or by the backend. */
export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${path}`;
}

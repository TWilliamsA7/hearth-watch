import { useEffect, useRef } from "react";

import { DEFAULT_THEME, EMPTY_SCENE, drawRoom, type Theme } from "./draw";
import type { FeedSource } from "./feed";
import type { Scene } from "./types";
import { INITIAL_VIEW, frameAt, reduce, type ViewFilter, type ViewModel } from "./viewModel";

export interface RoomViewProps extends ViewFilter {
  feed: FeedSource;
  /** Room geometry. Simulated runs supply it (RunInfo.scene); live rooms may omit it. */
  scene?: Scene | null;
  /** Owned by the dashboard's alert state (open, not yet resolved). Draws the red room border. */
  alertActive?: boolean;
  /** person_ids with an open alert; their skeletons turn red. */
  alertPersonIds?: readonly string[];
  /** Treat the room as unknown after this long without an observation. Matches the backend's 3 s. */
  staleAfterMs?: number;
  /** Interpolate between observations for smoother motion (adds about one frame of latency). */
  smoothing?: boolean;
  theme?: Theme;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * The camera view: room backdrop plus every observed skeleton, drawn from contract
 * Observations only. Several people are told apart by person_id and tagged P1, P2...
 * Ground truth never reaches this component.
 * Sizes itself to its container's width; height follows the scene's frame aspect.
 */
export function RoomView({
  feed,
  scene,
  roomId,
  source,
  alertActive = false,
  alertPersonIds,
  staleAfterMs = 3000,
  smoothing = true,
  theme = DEFAULT_THEME,
  className,
  style,
}: RoomViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const vm = useRef<ViewModel>(INITIAL_VIEW);
  const props = useRef({ scene, alertActive, alertPersonIds, staleAfterMs, smoothing, theme });
  props.current = { scene, alertActive, alertPersonIds, staleAfterMs, smoothing, theme };

  useEffect(() => {
    vm.current = INITIAL_VIEW;
    return feed.subscribe((msg, receivedAt) => {
      vm.current = reduce(vm.current, msg, receivedAt, { roomId, source });
    });
  }, [feed, roomId, source]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    let raf = 0;
    const render = () => {
      const p = props.current;
      const sc = p.scene ?? EMPTY_SCENE;
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
      const height = Math.round(width / sc.frame_aspect);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const frame = frameAt(vm.current, feed.now(), p);
      drawRoom(ctx, width, height, frame, sc, p.theme);
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [feed]);

  const aspect = (scene ?? EMPTY_SCENE).frame_aspect;
  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ display: "block", width: "100%", aspectRatio: String(aspect), ...style }}
      role="img"
      aria-label="Room camera view"
    />
  );
}

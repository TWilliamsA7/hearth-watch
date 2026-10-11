import { useEffect, useState } from 'react';
import { LIVE_SOURCES, calibrationText, linkStateText } from '@hearth/room-view';
import type { LiveLink } from './useLiveLink';

// Pull the people a real camera (or a replay of one) sees into the simulated room.
export default function LiveLinkPanel({ live, sources }: { live: LiveLink; sources: string[] }) {
  const { link } = live;
  const choices = [...new Set([...LIVE_SOURCES, ...sources])];
  const [source, setSource] = useState(LIVE_SOURCES[0]);
  const [height, setHeight] = useState(1.7);
  const maxPeople = link?.max_people ?? 3;
  // Follow the server: the link may already be on (another tab, or the Sim Studio).
  useEffect(() => { if (link?.enabled) setSource(link.source); }, [link?.enabled, link?.source]);
  const relink = (nextSource: string, nextHeight: number) => { if (link?.enabled) void live.setLinked(true, nextSource, nextHeight); };
  return <div className="live-link">
    <h3>Live people</h3>
    <p className="small">Show the people a real camera sees in this simulated room, beside the simulated residents. Their alerts come from the monitor as usual.</p>
    <label>Camera source<select value={source} onChange={e => { setSource(e.target.value); relink(e.target.value, height); }}>
      {choices.map(value => <option key={value} value={value}>{value}</option>)}
    </select></label>
    <label>Their height · {height.toFixed(2)} m<input type="range" min={1.4} max={2} step={0.01} value={height}
      onChange={e => { const value = Number(e.target.value); setHeight(value); relink(source, value); }} /></label>
    <label>Maximum live people<select value={maxPeople} disabled={live.busy || !link} onChange={e => {
      void live.setLinked(!!link?.enabled, source, height, Number(e.target.value));
    }}>
      {Array.from({ length: 10 }, (_, i) => i + 1).map(value => <option key={value} value={value}>{value}</option>)}
    </select></label>
    <p className="small">People appear after 1.5 seconds of consistent detection. Avatars follow walking at a controlled pace.</p>
    <div className="buttons">
      <button className={link?.enabled ? '' : 'go'} disabled={live.busy || !link} onClick={() => void live.setLinked(!link?.enabled, source, height)}>
        {link?.enabled ? 'Unlink' : 'Link into this room'}
      </button>
    </div>
    <p role="status" className="link-state">{link ? linkStateText(link) : 'Live link unavailable.'}</p>
    <details className="calibration">
      <summary>Calibrate the camera</summary>
      <p className="small">Tape a rectangle on the floor with an X at its centre, and mount the camera 2.2–2.5 m up above a front corner, aimed at the X and tilted 30–35° down. Then:</p>
      <div className="buttons">
        <button disabled={!link?.enabled || !!live.calibrating} onClick={() => void live.calibrate('walk', height)}
          title="12 s: walk slowly all over the area, upright, head and hips or knees in view">
          {live.calibrating === 'walk' ? '1 · Walk around the area…' : '1 · Walk'}
        </button>
        <button disabled={!link?.enabled || !!live.calibrating || link?.camera?.preset !== 'jetson_corner'} onClick={() => void live.calibrate('centre', height)}
          title="3 s: stand still on the X at the centre of the area">
          {live.calibrating === 'centre' ? '2 · Stand on the X…' : '2 · Stand on the X'}
        </button>
      </div>
      {link?.enabled && link.camera?.preset !== 'jetson_corner' && <p className="small">Step 2 needs a room generated with the "Jetson, area corner" camera (Room settings).</p>}
      {live.calibration && <p role="status" className={live.calibration.ok ? 'small' : 'error'}>{calibrationText(live.calibration)}</p>}
      {link?.camera && <p className="small">Simulated camera: {link.camera.preset === 'jetson_corner' ? `${link.camera.corner} corner` : link.camera.preset}, {link.camera.height.toFixed(2)} m, tilt {link.camera.tilt.toFixed(1)}°, pan {link.camera.pan.toFixed(1)}°</p>}
    </details>
    {live.error && <p role="alert" className="error">{live.error}</p>}
  </div>;
}

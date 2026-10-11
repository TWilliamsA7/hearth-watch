import { useEffect, useState } from 'react';
import { DEFAULT_ROOM_SETTINGS, roomSettings, type RoomSettings } from './liveSimulation';
import type { useLiveSimulation } from './useLiveSimulation';

export default function RoomSettingsPanel({ simulation }: { simulation: ReturnType<typeof useLiveSimulation> }) {
  const [settings, setSettings] = useState<RoomSettings>(DEFAULT_ROOM_SETTINGS);
  const world = simulation.client?.world;
  useEffect(() => {
    if (world) setSettings(roomSettings(world));
  }, [world]);
  function set<K extends keyof RoomSettings>(key: K, value: RoomSettings[K]) {
    setSettings(previous => ({ ...previous, [key]: value }));
  }
  function number(label: string, key: keyof Pick<RoomSettings, 'actors' | 'width' | 'depth' | 'furniture' | 'cameraHeight' | 'tilt' | 'fov' | 'activity' | 'social'>, min: number, max: number, step: number) {
    return <label>{label}<input type="number" required min={min} max={max} step={step} value={settings[key]} onChange={e => set(key, Number(e.target.value))} /></label>;
  }
  return <details className="room-settings">
    <summary>Room settings</summary>
    <form className="room-settings-panel" onSubmit={event => {
      event.preventDefault();
      const button = (event.nativeEvent as SubmitEvent).submitter as HTMLButtonElement | null;
      void simulation.generate(settings, button?.value !== 'paused');
    }}>
      <h3>Generate a simulated room</h3>
      <p className="small">Change these settings, then generate a new room. This replaces the current simulation.</p>
      <fieldset disabled={!simulation.client?.connected || simulation.busy}>
        <legend className="sr-only">Room generation settings</legend>
        <div className="room-settings-grid">
          <label>Seed<div className="seed-input"><input aria-label="Seed" type="number" required step={1} value={settings.seed} onChange={e => set('seed', Number(e.target.value))} /><button type="button" onClick={() => set('seed', Math.floor(Math.random() * 100000))}>Random</button></div></label>
          {number('Residents', 'actors', 0, 20, 1)}
          {number('Width (m)', 'width', 4, 10, 0.1)}
          {number('Depth (m)', 'depth', 4, 9, 0.1)}
          <label>Furniture · {Math.round(settings.furniture * 100)}%<input type="range" min={0} max={1} step={0.05} value={settings.furniture} onChange={e => set('furniture', Number(e.target.value))} /></label>
          <label>Camera location<select value={settings.camera} onChange={e => {
            const camera = e.target.value as RoomSettings['camera'];
            setSettings(previous => ({ ...previous, camera, fov: camera === 'jetson' || camera === 'jetson_corner' ? 43 : previous.fov }));
          }}>
            <option value="corner_left">Corner (left)</option><option value="corner_right">Corner (right)</option>
            <option value="front">Front wall</option><option value="ceiling">Ceiling</option>
            <option value="jetson">Jetson, front wall</option><option value="jetson_corner">Jetson, area corner</option>
          </select></label>
          {number('Camera height (m)', 'cameraHeight', 1, 3.4, 0.05)}
          {number('Field of view (°)', 'fov', 30, 110, 1)}
          {(settings.camera === 'jetson' || settings.camera === 'jetson_corner') && number('Tilt down (°)', 'tilt', -15, 60, 1)}
          <label>Lighting<select value={settings.lighting} onChange={e => set('lighting', e.target.value as RoomSettings['lighting'])}>
            <option value="day">Day</option><option value="evening">Evening</option><option value="dim">Dim</option>
          </select></label>
          {number('Activity (×)', 'activity', 0.25, 4, 0.25)}
          {number('Socialising (×)', 'social', 0, 4, 0.25)}
        </div>
        <div className="buttons"><button type="submit" value="playing">Generate room</button><button type="submit" value="paused">Generate paused</button></div>
      </fieldset>
      {!simulation.client?.connected && <p className="small">Connect to the simulation server to generate a room.</p>}
      {simulation.error && <p role="alert" className="error">{simulation.error}</p>}
    </form>
  </details>;
}

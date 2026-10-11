import { Component, type ReactNode, useState } from 'react';
import { LiveView3D } from '@hearth/room-view';
import type { useLiveSimulation } from './useLiveSimulation';

class RenderBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    return this.state.failed ? <div className="simulation-message" role="alert">The 3D room could not be displayed. Enable WebGL and reload.</div> : this.props.children;
  }
}

function supportsWebGL() {
  try { return !!document.createElement('canvas').getContext('webgl2'); }
  catch { return false; }
}

export default function LiveSimulationView({ simulation, alertPersonIds }: {
  simulation: ReturnType<typeof useLiveSimulation>; alertPersonIds: string[];
}) {
  const [supported] = useState(supportsWebGL);
  const { client, ready, error, selected, setSelected } = simulation;
  return <div className="live-simulation">
    {supported && client && <RenderBoundary key={client.world?.run_id ?? 'loading'}>
      <LiveView3D client={client} selected={selected} onSelect={setSelected} alertPersonIds={alertPersonIds} cameraMode="orbit" />
    </RenderBoundary>}
    <span className="simulation-badge">SIMULATED · 3D ROOM</span>
    {!supported ? <div className="simulation-message" role="alert">The 3D room requires WebGL. Enable hardware acceleration in your browser and reload.</div>
      : (!ready || error) && <div className="simulation-message" role="status">
        <p>{error || (client?.connected ? 'Loading the simulated room…' : 'Simulation disconnected. Start the monitoring server to see the room.')}</p>
        <button onClick={simulation.retry}>Retry connection</button>
      </div>}
    {ready && !error && <span className="simulation-hint">{selected ? `Selected: ${selected} · ` : ''}Click a person to select · drag to orbit · scroll to zoom</span>}
  </div>;
}

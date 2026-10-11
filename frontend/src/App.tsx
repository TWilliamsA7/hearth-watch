import { useEffect, useState } from 'react';
import type { SourceMode } from './feedState';
import { useLiveSimulation } from './useLiveSimulation';
import { useRoomFeed } from './useRoomFeed';
import { useRooms } from './useRooms';
import MonitorView from './MonitorView';
import SimulatorPanel from './SimulatorPanel';
import RecognitionView from './RecognitionView';
import Popover from './Popover';
import Icon from './Icon';

// The Hearth server serves this page, so the API, WebSockets and Sim Studio share its origin.
const base = window.location.origin;
const TABS = [
  { id: 'monitor', path: '/', label: 'Monitor' },
  { id: 'scenarios', path: '/scenarios', label: 'Scenarios' },
  { id: 'recognition', path: '/recognition', label: 'Recognition' },
] as const;
type Tab = typeof TABS[number]['id'];
const tabFor = (path: string): Tab => TABS.find(tab => tab.path === path)?.id ?? 'monitor';

export default function App() {
  const [tab, setTab] = useState<Tab>(() => tabFor(window.location.pathname));
  const [mode, setMode] = useState<SourceMode>('simulation');
  const room = useRooms(base).room_ids[0];
  const feed = useRoomFeed(base, mode, room);
  const simulation = useLiveSimulation(tab === 'monitor' && mode === 'simulation' ? base : null, room);
  useEffect(() => {
    const back = () => setTab(tabFor(window.location.pathname));
    window.addEventListener('popstate', back);
    return () => window.removeEventListener('popstate', back);
  }, []);
  function go(next: Tab) {
    const path = TABS.find(item => item.id === next)!.path;
    if (window.location.pathname !== path) window.history.pushState(null, '', path);
    setTab(next);
  }
  const active = Object.values(feed.state.alerts).filter(alert => alert.status !== 'resolved').length;
  return <main className="dashboard">
    <a className="skip-link" href="#main-content">Skip to content</a>
    <header className="page-header">
      <a className="brand" href="/" onClick={event => { event.preventDefault(); go('monitor'); }}><img className="brand-mark" src="/hearthwatch-mark.png" alt="" width="60" height="52" /><h1 aria-label="HearthWatch">HearthWatc<span className="wordmark-tail" aria-hidden="true">h<svg className="heart-flourish" viewBox="0 0 36 36" fill="none"><path d="M 1 27 C 6 31 10 29 14 25 C 19 21 29 16 29 10 C 29 4 22 3 18 9 C 14 3 7 4 7 10 C 7 16 14 20 18 23" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" /></svg></span></h1></a>
      <nav className="tabs" aria-label="Views">{TABS.map(item =>
        <a key={item.id} href={item.path} aria-current={tab === item.id ? 'page' : undefined} className={tab === item.id ? 'selected' : ''}
          onClick={event => { event.preventDefault(); go(item.id); }}>{item.label}</a>)}</nav>
      <div className="header-actions">
        <button className={`status ${!feed.connected ? 'unknown' : active ? 'warning' : ''}`} onClick={() => go('monitor')} title="Show alerts">
          <span className="tiny-dot" />{!feed.connected ? 'Disconnected' : active ? `${active} active alert${active === 1 ? '' : 's'}` : 'No active alerts'}
        </button>
        <Popover label={<><Icon name="settings" /> Developer</>} title="Developer tools" align="right">{() => <ul className="developer-links">
          <li><a href="/studio/" target="_blank" rel="noopener">Sim Studio</a><span className="small">Resident inspector, overlays, actor commands, camera calibration</span></li>
          <li><a href="/studio/runs" target="_blank" rel="noopener">Recorded run player</a><span className="small">Saved sim runs against ground truth</span></li>
          <li><a href="/docs" target="_blank" rel="noopener">API reference</a><span className="small">Every monitoring, simulation and evaluation route</span></li>
        </ul>}</Popover>
      </div>
    </header>
    {feed.error && <p role="alert" className="error">{feed.error}</p>}
    <div id="main-content" className="content">
      {tab === 'monitor' && <MonitorView base={base} room={room} mode={mode} setMode={setMode} feed={feed} simulation={simulation} />}
      {tab === 'scenarios' && <div className="page scenarios-page"><SimulatorPanel base={base} /></div>}
      {tab === 'recognition' && <RecognitionView room={room} />}
    </div>
  </main>;
}

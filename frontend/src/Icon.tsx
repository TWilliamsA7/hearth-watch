type IconName = 'home' | 'room' | 'alert' | 'play' | 'chart' | 'signal' | 'clock' | 'check' | 'settings';
const paths: Record<IconName, string> = {
  home: 'M3 10 12 3l9 7M5 9v12h14V9M9 21v-7h6v7',
  room: 'M4 3h16v18H4zM8 3v18M15 12h1',
  alert: 'm12 3 10 18H2L12 3ZM12 9v5M12 17h.01',
  play: 'm9 5 11 7-11 7V5Z',
  chart: 'M4 3v18h17M8 16v-5M13 16V7M18 16v-9',
  signal: 'M4 19v-3M9 19v-7M14 19V8M19 19V4',
  clock: 'M12 8v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
  check: 'm6 12 4 4 8-8',
  settings: 'M4 7h16M4 17h16M8 4v6M16 14v6',
};
export default function Icon({ name, className = '' }: { name: IconName; className?: string }) {
  return <svg className={`icon ${className}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}

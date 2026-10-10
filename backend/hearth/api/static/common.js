export const $ = (id) => document.getElementById(id);
export const labels = {possible_fall: 'Possible fall', help_requested: 'Help requested', source_offline: 'Source offline', source_online: 'Source reconnected', recovered: 'Upright posture observed', alert_acknowledged: 'Accepted / acknowledged', alert_resolved: 'Resolved'};
export function element(tag, text = '', className = '') { const node = document.createElement(tag); node.textContent = text; node.className = className; return node; }
export function time(value) { return new Date(value).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}); }
export function notice(text = '') { $('notice').textContent = text; }
export async function api(path, method = 'GET', body) {
  const response = await fetch(path, {method, headers: body ? {'Content-Type':'application/json'} : {}, body:body ? JSON.stringify(body) : undefined});
  const data = await response.json();
  if (!response.ok) { const error = new Error(typeof data.detail === 'string' ? data.detail : 'Check the submitted fields and try again.'); error.status = response.status; throw error; }
  return data;
}
export function connect(path, onMessage, onState) {
  let socket, timer, stopped = false;
  function open() {
    if (stopped) return;
    onState('Connecting…');
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${path}`);
    socket.onopen = () => onState('Live');
    socket.onmessage = (event) => { try { onMessage(JSON.parse(event.data)); } catch (error) { notice(error.message); } };
    socket.onerror = () => onState('Connection interrupted');
    socket.onclose = (event) => {
      if (stopped) return;
      onState('Reconnecting…'); timer = setTimeout(open, 2000);
    };
  }
  open();
  return () => { stopped = true; clearTimeout(timer); socket?.close(); };
}
let audio;
export async function enableSound() { audio ||= new (window.AudioContext || window.webkitAudioContext)(); await audio.resume(); beep(); }
export function beep() {
  if (!audio || audio.state !== 'running') return;
  const oscillator = audio.createOscillator(), gain = audio.createGain();
  oscillator.connect(gain); gain.connect(audio.destination); oscillator.frequency.value = 660;
  gain.gain.setValueAtTime(.08, audio.currentTime); gain.gain.exponentialRampToValueAtTime(.001, audio.currentTime + .35);
  oscillator.start(); oscillator.stop(audio.currentTime + .35);
}
export function sourceLabel(type) { return {edge:'LIVE',simulation:'SIMULATED',replay:'REPLAY',manual:'BUTTON REQUEST'}[type] || type; }
export function requestId() { return typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : Array.from(crypto.getRandomValues(new Uint32Array(4)),n=>n.toString(16).padStart(8,'0')).join(''); }
export function postureLabel(source) {
  if (!source.posture_model) return '';
  if (!source.online || source.observation_stale || source.posture === 'unknown') return ' · Posture unknown';
  const label = {standing:'Standing',sitting:'Sitting',fallen:'Fallen'}[source.posture] || 'Unknown';
  return ` · ${label} (model${source.posture_confidence == null ? '' : ', '+Math.round(source.posture_confidence*100)+'%'})`;
}

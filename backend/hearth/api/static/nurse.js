import {$, api, element, time, notice, labels, connect, enableSound, beep, sourceLabel, postureLabel} from './common.js';
let nurse = null, disconnect = null, sound = false;
const alerts = new Map(), events = new Map(), sources = new Map();
const sourceKey = (s) => JSON.stringify([s.room_id,s.source_type,s.source,s.scenario_id,s.run_id]);
function setConnection(state) { $('connection').textContent = state; $('connection').className = 'pill' + (state === 'Live' ? ' green' : ''); }
function preferences() {
  return {display_name:$('name').value.trim(), room_ids:[...document.querySelectorAll('input[name="room"]:checked')].map(n=>n.value),
    alert_kinds:[...document.querySelectorAll('input[name="kind"]:checked')].map(n=>n.value), on_duty:$('duty').checked};
}
function fillPreferences() {
  $('name').value = nurse.display_name; $('duty').checked = nurse.on_duty;
  document.querySelectorAll('input[name="room"]').forEach(n=>n.checked=nurse.room_ids.includes(n.value));
  document.querySelectorAll('input[name="kind"]').forEach(n=>n.checked=nurse.alert_kinds.includes(n.value));
  $('save').textContent = 'Save coverage';
  $('session-note').textContent = nurse.on_duty ? `Receiving for ${nurse.display_name}. Keep this page open. Updates require saving.` : 'Off duty. No new matching alerts; requests you accepted remain available.';
}
function loadSnapshot(snapshot) {
  nurse = snapshot.nurse; alerts.clear(); events.clear(); sources.clear();
  snapshot.alerts.forEach(a=>alerts.set(a.alert_id,a)); snapshot.events.forEach(e=>events.set(e.event_id,e)); snapshot.sources.forEach(s=>sources.set(sourceKey(s),s));
  fillPreferences(); render();
}
function startStream() {
  disconnect?.();
  disconnect = connect(`/ws/nurse/${encodeURIComponent(nurse.nurse_id)}`, message => {
    if (message.type === 'nurse_snapshot') return loadSnapshot(message.data);
    if (message.type === 'alert') { const fresh = !alerts.has(message.data.alert_id); alerts.set(message.data.alert_id,message.data); if (fresh && sound && message.data.status === 'open') beep(); }
    if (message.type === 'event') events.set(message.data.event_id,message.data);
    if (message.type === 'source_status') sources.set(sourceKey(message.data),message.data);
    render();
  }, async state => {
    setConnection(state);
    if (state === 'Reconnecting…' && nurse) {
      try { await api(`/nurse-sessions/${nurse.nurse_id}/snapshot`); }
      catch (error) { if (error.status === 404) { disconnect?.(); nurse=null; sessionStorage.removeItem('hearth-nurse'); alerts.clear(); events.clear(); sources.clear(); render(); $('save').textContent='Start receiving alerts'; setConnection('Session expired'); notice('The backend restarted. Start a new nurse session to reconnect.'); } }
    }
  });
}
async function action(alert, verb) {
  try { notice(); const updated = await api(`/alerts/${alert.alert_id}/${verb}`, 'POST', {nurse_id:nurse.nurse_id}); alerts.set(updated.alert_id,updated); render(); }
  catch (error) { notice(error.message); if(nurse) { try { loadSnapshot(await api(`/nurse-sessions/${nurse.nurse_id}/snapshot`)); } catch {} } }
}
function render() {
  const all = [...alerts.values()].sort((a,b)=>new Date(b.opened_at)-new Date(a.opened_at));
  $('open-count').textContent=all.filter(a=>a.status==='open').length;
  $('accepted-count').textContent=all.filter(a=>a.status==='acknowledged').length;
  $('resolved-count').textContent=all.filter(a=>a.status==='resolved').length;
  const visible=all.filter(a=>$('show-resolved').checked || a.status!=='resolved');
  $('alerts').replaceChildren();
  if (!visible.length) {
    const empty=element('div','','empty'); empty.append(element('h3',!nurse?'No session started':nurse.on_duty?'No requests need attention.':'You’re off duty.'),element('p',!nurse?'Enter your name and select coverage on the left.':nurse.on_duty?'Matching requests will appear here. No alerts does not confirm room safety.':'Go on duty and save coverage to receive new requests.')); $('alerts').append(empty);
  }
  for (const alert of visible) {
    const card=element('article','',`alert-card ${alert.status}`), top=element('div','','card-top'), tags=element('div','','tags');
    tags.append(element('span',alert.room_id,'pill'),element('span',sourceLabel(alert.source_type),`pill ${alert.source_type}`),element('span',alert.status==='acknowledged'?'ACCEPTED':alert.status.toUpperCase(),'pill '+(alert.status==='open'?'warning':'green')));
    top.append(tags,element('time',time(alert.opened_at),'small')); card.append(top,element('h3',labels[alert.kind]));
    const description=alert.kind==='help_requested' ? (alert.metadata.method==='button'?'A person pressed the room help button.':alert.metadata.method==='hand_gesture'?'A hand help signal was held toward the camera.':'A raised arm was held long enough to request assistance.') : alert.kind==='possible_fall'?(alert.metadata.reason==='sustained_model_fallen'?'Sustained lying posture indicates a possible fall. Please check in.':'Sustained floor-level posture was observed. Please check in.'):'This source stopped sending. The room cannot be verified through this source.';
    card.append(element('p',alert.kind==='appointment_overdue' ? `${alert.metadata.resident_label}: arrival not confirmed for the appointment at ${time(alert.metadata.scheduled_at)}. Please follow up.` : description));
    if(alert.recovered_at) card.append(element('p',alert.kind==='appointment_overdue'?'Arrival subsequently checked in. Review before resolving.':alert.kind==='source_offline'?'Source reconnected. Review before resolving.':'Upright posture observed afterward. Review before resolving.','small'));
    const bottom=element('div','','card-bottom');
    bottom.append(element('span',alert.accepted_by_name?`Accepted by ${alert.accepted_by_name}`:alert.status==='acknowledged'?'Acknowledged without an assigned nurse':'No nurse has accepted this request.','small'));
    if(alert.status!=='resolved') {
      const mine=alert.accepted_by===nurse?.nurse_id;
      const button=element('button',mine?'Resolve':alert.accepted_by?'Assigned':'Accept request');
      button.disabled=Boolean(alert.accepted_by&&!mine)||!nurse||(!mine&&!nurse.on_duty);
      button.onclick=()=>action(alert,mine?'resolve':'accept'); bottom.append(button);
    }
    card.append(bottom); $('alerts').append(card);
  }
  $('timeline').replaceChildren();
  const history=[...events.values()].sort((a,b)=>new Date(b.timestamp)-new Date(a.timestamp)).slice(0,40);
  for(const event of history){const li=element('li');li.append(element('span',`${event.room_id} · ${labels[event.event_type]||event.event_type}${event.metadata.nurse_name?' · '+event.metadata.nurse_name:''}`),element('time',time(event.timestamp)));$('timeline').append(li);}
  if(!history.length)$('timeline').append(element('li','No matching activity yet.'));
  $('sources').replaceChildren();
  for(const source of sources.values()) $('sources').append(element('div',`${source.room_id} · ${source.source} · ${sourceLabel(source.source_type)} — ${!source.online?'Offline':source.observation_stale?'Connected; observations stale':'Observations arriving'}${postureLabel(source)}`,'source-row'));
  if(!sources.size)$('sources').textContent='No source reports in your current coverage.';
}
$('preferences').onsubmit=async event=>{
  event.preventDefault(); const prefs=preferences();
  if(!prefs.display_name || !prefs.room_ids.length || !prefs.alert_kinds.length)return notice('Enter a name and choose at least one place and situation.');
  $('save').disabled=true;
  try{notice();nurse=await api(nurse?`/nurse-sessions/${nurse.nurse_id}`:'/nurse-sessions',nurse?'PUT':'POST',prefs);sessionStorage.setItem('hearth-nurse',nurse.nurse_id);loadSnapshot(await api(`/nurse-sessions/${nurse.nurse_id}/snapshot`));startStream();}
  catch(error){notice(error.message);}finally{$('save').disabled=false;}
};
$('show-resolved').onchange=render;
$('sound').onclick=async()=>{try{await enableSound();sound=true;$('sound').textContent='Alert sound enabled';}catch{notice('Your browser could not enable audio. Visual alerts remain available.');}};
async function init(){
  try{
    const config=await api('/rooms');
    config.room_ids.forEach((room,index)=>{const label=element('label'),input=document.createElement('input');input.type='checkbox';input.name='room';input.value=room;input.checked=index===0;label.append(input,document.createTextNode(room));$('rooms').append(label);});
    const saved=sessionStorage.getItem('hearth-nurse');
    if(saved){try{loadSnapshot(await api(`/nurse-sessions/${saved}/snapshot`));startStream();}catch(error){sessionStorage.removeItem('hearth-nurse');notice('Start a new session; the previous session is no longer available.');}}
  }catch(error){notice('Cannot reach the backend. '+error.message);}
}
init();

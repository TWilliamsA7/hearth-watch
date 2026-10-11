const $ = id => document.getElementById(id);
let stream = null, session = null, generation = 0, ready = false, busy = false;
const say = text => { $('result').textContent = text; };
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function api(path, method = 'GET', data) {
  const response = await fetch(path, {method, headers: {'Content-Type':'application/json','X-Hearth-Face-Demo':'1'},
    body: data === undefined ? undefined : JSON.stringify(data)});
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : 'Check the supplied fields.');
  return value;
}
function controls() {
  $('open').disabled = !ready || !!stream || busy;
  $('enroll').disabled = !ready || !stream || busy || !$('consent').checked;
  $('recognize').disabled = !ready || !stream || busy;
  $('stop').disabled = !stream;
  $('room').disabled = busy;
  $('create-appointment').disabled = !ready || busy;
}
async function stop() {
  generation++;
  if (session) { const id = session; session = null; api(`/face-demo/sessions/${id}`, 'DELETE').catch(() => {}); }
  stream?.getTracks().forEach(track => track.stop());
  stream = null; $('camera').srcObject = null; busy = false; controls();
  say('Camera is off. Nothing is being captured.');
}
function frame() {
  const video = $('camera');
  if (!stream || !video.videoWidth) throw new Error('Start the camera and wait for a preview.');
  const canvas = document.createElement('canvas');
  canvas.width = 640; canvas.height = Math.round(video.videoHeight * 640 / video.videoWidth);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', .85).split(',')[1];
}
async function refresh() {
  const profiles = await api('/face-demo/profiles');
  $('profiles').replaceChildren();
  if (!profiles.length) $('profiles').textContent = 'No patients enrolled.';
  for (const profile of profiles) {
    const row = document.createElement('div'); row.className = 'enrollment';
    const label = document.createElement('span'); label.textContent = `${profile.alias} · expires ${new Date(profile.expires_at * 1000).toLocaleString()}`;
    const remove = document.createElement('button'); remove.textContent = 'Withdraw & delete'; remove.className = 'secondary';
    remove.onclick = async () => { try { await stop(); await api(`/face-demo/profiles/${profile.profile_id}`, 'DELETE'); await refresh(); say('Enrollment deleted. Recognition has stopped.'); } catch (error) { say(error.message); } };
    const link = document.createElement('button'); link.textContent = 'Link selected appointment'; link.className = 'secondary';
    link.onclick = async () => { try { const id = $('appointment').value; if (!id) throw new Error('Select an open demo appointment first.'); await api(`/face-demo/profiles/${profile.profile_id}/appointments/${id}`, 'POST'); say(`Appointment linked to ${profile.alias}. No new face enrollment needed.`); } catch (error) { say(error.message); } };
    row.append(label, link, remove); $('profiles').append(row);
  }
  const appointments = await api('/appointments');
  const current = $('appointment').value;
  $('appointment').replaceChildren(new Option('Recognition only', ''));
  for (const item of appointments.filter(a => a.demo && a.room_id === $('room').value && ['scheduled','overdue'].includes(a.status))) {
    $('appointment').add(new Option(`${item.resident_label} · ${new Date(item.scheduled_at).toLocaleTimeString()}`, item.appointment_id));
  }
  if ([...$('appointment').options].some(option => option.value === current)) $('appointment').value = current;
}
$('consent').onchange = controls;
$('room').onchange = () => refresh().catch(error => say(error.message));
$('stop').onclick = stop;
$('open').onclick = async () => {
  try {
    const request = ++generation;
    const opened = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
    if (request !== generation) { opened.getTracks().forEach(track => track.stop()); return; }
    stream = opened; $('camera').srcObject = stream; await $('camera').play();
    say('Preview is local. Enroll yourself or start recognition when ready.'); controls();
  } catch (error) { say(`Camera unavailable: ${error.message}`); }
};
$('enroll').onclick = async () => {
  const alias = $('alias').value.trim();
  if (!alias || !$('consent').checked) { say('Enter a demo alias and confirm your consent.'); return; }
  busy = true; controls(); const active = ++generation;
  const frames = [];
  try {
    for (const direction of ['Look straight ahead.', 'Turn slightly left.', 'Turn slightly right.']) {
      say(direction); await delay(1400);
      if (active !== generation) return;
      frames.push(frame());
    }
    await api('/face-demo/enroll', 'POST', {alias, consent:$('consent').checked, identify_alerts:true, frames,
      appointment_id:$('appointment').value || null});
    $('consent').checked = false;
    await refresh(); say(`${alias} enrolled. Start recognition to test the check-in.`);
  } catch (error) { say(error.message); }
  finally { frames.length = 0; if (active === generation) { busy = false; controls(); } }
};
$('recognize').onclick = async () => {
  busy = true; controls(); const active = ++generation;
  let ownSession = null;
  try {
    ownSession = (await api('/face-demo/sessions', 'POST', {room_id:$('room').value})).session_id;
    if (active !== generation) return;
    session = ownSession;
    while (active === generation && stream) {
      try {
        const result = await api('/face-demo/recognize', 'POST', {session_id:ownSession, frame:frame()});
        if (active !== generation) break;
        if (result.status === 'recognized') {
          say(`${result.profile.alias} recognized.${result.checked_in ? ' Demo appointment checked in.' : ' Recognition confirmed.'}`);
          await refresh(); break;
        }
        say(result.status === 'confirming' ? `Confirming match: ${result.matches}/3` : 'No confirmed match. Face the camera in good light.');
      } catch (error) { say(error.message); }
      await delay(650);
    }
  } catch (error) { say(error.message); }
  finally {
    if (ownSession) await api(`/face-demo/sessions/${ownSession}`, 'DELETE').catch(() => {});
    if (session === ownSession) session = null;
    if (active === generation) { busy = false; controls(); }
  }
};
$('create-appointment').onclick = async () => {
  try {
    const alias = $('alias').value.trim(); if (!alias) throw new Error('Enter a fictional demo alias first.');
    const item = await api('/appointments', 'POST', {resident_label:alias, room_id:$('room').value,
      scheduled_at:new Date(Date.now()+30000).toISOString(), grace_seconds:15, demo:true});
    await refresh(); $('appointment').value = item.appointment_id;
    say('Demo appointment created. Enroll with it selected, then recognize to check in.');
  } catch (error) { say(error.message); }
};
window.addEventListener('pagehide', stop);
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
async function init() {
  try {
    const state = await api('/face-demo/status'); ready = state.enabled; $('notice').textContent = state.message;
    const rooms = await api('/rooms'); rooms.room_ids.forEach(room => $('room').add(new Option(room, room)));
    if (ready) await refresh(); controls();
  } catch (error) { $('notice').textContent = error.message; }
}
init();

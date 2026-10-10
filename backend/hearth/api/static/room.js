import {$,api,element,notice,connect,time,sourceLabel,requestId,postureLabel} from './common.js';
let disconnect=null, connected=false, current=null, pendingId=null, roomVersion=0;
const alerts=new Map(), sources=new Map();
function render(){
  const requests=[...alerts.values()].filter(a=>a.kind==='help_requested').sort((a,b)=>new Date(b.opened_at)-new Date(a.opened_at));
  current=requests.find(a=>a.status!=='resolved')||requests[0]||null;
  const active=current&&current.status!=='resolved';
  $('help').disabled=!connected||Boolean(active);$('help').textContent=active?'Request sent':'Request help';
  let title='No request pending',detail='Acknowledgement will appear here when a nurse accepts.';
  if(current){
    if(current.status==='open'){title='Help requested';detail='Waiting for a nurse to accept your request.';}
    else if(current.status==='acknowledged'){title='Your request was acknowledged';detail=current.accepted_by_name?`${current.accepted_by_name} accepted your request.`:'Your request was acknowledged by care staff.';}
    else{title='Request resolved';detail='You can request help again if you need it.';}
    $('request-detail').textContent=`${sourceLabel(current.source_type)} · ${current.metadata.method==='button'?'Help button':'Raised arm'} · Requested at ${time(current.opened_at)}${current.accepted_by_name?' · Accepted by '+current.accepted_by_name:''}`;
  }else $('request-detail').textContent='No request in this room yet.';
  if(!connected){title='Connection interrupted';detail='Request status is unavailable. Reconnecting to the care desk…';}
  $('request-status').replaceChildren(element('h2',title),element('p',detail));
  const data=[...sources.values()];
  $('room-connection').textContent=!data.length?'No camera observations yet. The help button works independently of the camera.':data.map(s=>`${s.source} (${sourceLabel(s.source_type)}): ${!s.online?'offline':s.observation_stale?'observations stale':'observations arriving'}${postureLabel(s)}`).join(' · ');
}
async function selectRoom(){
  const version=++roomVersion, room=$('room').value;
  disconnect?.();connected=false;alerts.clear();sources.clear();pendingId=null;render();
  // Subscribe first; the server replays the current room state without a fetch/subscribe gap.
  disconnect=connect(`/ws/room/${encodeURIComponent(room)}`,message=>{
    if(version!==roomVersion)return;
    if(message.type==='alert')alerts.set(message.data.alert_id,message.data);
    if(message.type==='source_status'){const s=message.data;sources.set(JSON.stringify([s.source_type,s.source,s.scenario_id,s.run_id]),s);}
    render();
  },state=>{if(version!==roomVersion)return;if(state==='Connecting…'){alerts.clear();sources.clear();}connected=state==='Live';$('connection').textContent=state;$('connection').className='pill'+(connected?' green':'');render();});
}
$('room').onchange=selectRoom;
$('help').onclick=async()=>{
  const room=$('room').value,version=roomVersion; $('help').disabled=true;
  pendingId ||= requestId();
  try{notice();const alert=await api('/help-requests','POST',{request_id:pendingId,room_id:room});if(version!==roomVersion)return;alerts.set(alert.alert_id,alert);pendingId=null;}
  catch(error){notice('Request was not confirmed. Please retry. '+error.message);}
  finally{if(version===roomVersion)render();}
};
async function init(){try{const config=await api('/rooms');config.room_ids.forEach(room=>{const option=element('option',room);option.value=room;$('room').append(option);});$('gesture-instruction').textContent=`Hold one hand above your head for ${config.gesture_hold_seconds} seconds in view of the camera.`;await selectRoom();}catch(error){notice(error.message);}}
init();

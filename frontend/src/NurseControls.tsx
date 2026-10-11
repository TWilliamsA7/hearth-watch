import { useEffect, useState } from 'react';
import type { NursePreferences } from './generated/contract';
import type { NurseSession } from './useNurseSession';

export default function NurseControls({session, rooms}: {session: NurseSession; rooms: string[]}) {
  const [preferences,setPreferences] = useState<NursePreferences>({display_name:'',room_ids:rooms as NursePreferences['room_ids'],alert_kinds:['help_requested','possible_fall','appointment_overdue'],on_duty:true});
  useEffect(() => { if (session.nurse) setPreferences(session.nurse); },[session.nurse]);
  // Cover every configured room by default once the list arrives.
  useEffect(() => { if (!session.nurse) setPreferences(previous => ({...previous, room_ids:rooms as NursePreferences['room_ids']})); },[rooms.join(','), !!session.nurse]);
  const kinds = [['help_requested','Help requests'],['possible_fall','Possible falls'],['source_offline','Camera / source offline'],['appointment_overdue','Appointment reminders']] as const;
  return <details className="observation-help" open={!session.nurse}>
    <summary>Nurse session{session.nurse ? ` · ${session.nurse.display_name} · ${session.connected ? session.nurse.on_duty ? 'On duty':'Off duty':'Disconnected'}` : ' · Start receiving alerts'}</summary>
    <form onSubmit={event => {event.preventDefault(); const {display_name,room_ids,alert_kinds,on_duty} = preferences; void session.save({display_name,room_ids,alert_kinds,on_duty});}}>
      <label>Display name <input required maxLength={80} value={preferences.display_name} onChange={e => setPreferences({...preferences,display_name:e.target.value})} /></label>
      <fieldset><legend>Rooms</legend>{rooms.map(room => <label key={room}><input type="checkbox" checked={preferences.room_ids.includes(room)} onChange={e => setPreferences({...preferences,room_ids:(e.target.checked ? [...preferences.room_ids,room] : preferences.room_ids.filter(value => value!==room)) as NursePreferences['room_ids']})}/>{room} </label>)}</fieldset>
      <fieldset><legend>Situations</legend>{kinds.map(([kind,label]) => <label key={kind}><input type="checkbox" checked={preferences.alert_kinds.includes(kind)} onChange={e => setPreferences({...preferences,alert_kinds:(e.target.checked ? [...preferences.alert_kinds,kind] : preferences.alert_kinds.filter(value => value!==kind)) as NursePreferences['alert_kinds']})}/>{label} </label>)}</fieldset>
      <label><input type="checkbox" checked={preferences.on_duty} onChange={e => setPreferences({...preferences,on_duty:e.target.checked})}/>On duty</label>
      <button disabled={session.pending || !preferences.display_name.trim() || !preferences.room_ids.length || !preferences.alert_kinds.length}>{session.nurse ? 'Save coverage':'Start session'}</button>
      <button type="button" onClick={() => void session.enableSound()} disabled={session.sound}>{session.sound ? 'Alert sound enabled':'Enable alert sound'}</button>
    </form>
    <p className="small">Keep this page open to receive alerts. Off-duty sessions retain their accepted work. Each tab starts its own session.</p>
    {session.error && <p role="alert" className="error">{session.error}</p>}
  </details>;
}

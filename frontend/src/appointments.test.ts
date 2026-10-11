import test from 'node:test';
import assert from 'node:assert/strict';
import { parseMessage } from './protocol.ts';
import { emptyFeed, reduceFeed } from './feedState.ts';

test('appointment reminders pass the contract and reach every source view, but not other rooms', () => {
  const message = parseMessage({type:'event',data:{schema_version:'1',event_id:'schedule-test',timestamp:new Date().toISOString(),source:'appointments',source_type:'simulation',room_id:'room-01',scenario_id:null,run_id:null,event_type:'appointment_overdue',confidence:null,metadata:{demo:true}}});
  assert.ok(message);
  for (const mode of ['edge','simulation','replay'] as const) {
    assert.equal(reduceFeed(emptyFeed(),message,Date.now(),mode).events.length,1);
    const other = {...message,data:{...message.data,room_id:'room-02'}};
    assert.equal(reduceFeed(emptyFeed(),other,Date.now(),mode).events.length,0);
  }
});

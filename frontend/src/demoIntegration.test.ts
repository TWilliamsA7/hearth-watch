import test from 'node:test';
import assert from 'node:assert/strict';
import messages from '../contracts/demo-integration.fixture.json' with { type: 'json' };
import { parseMessage, validateApi } from './protocol.ts';

test('current composed backend messages and resident acknowledgements pass validation', () => {
  assert.ok(messages.length > 0);
  for (const message of messages) assert.ok(parseMessage(message), `Rejected ${message.type}`);
  const alert = messages.find(message => message.type === 'alert' && message.data.kind === 'help_requested');
  assert.ok(alert);
  assert.ok(validateApi('monitoring#/$defs/Alert', alert.data));
});

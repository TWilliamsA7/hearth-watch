import Ajv from 'ajv';
import monitoring from '../contracts/monitoring.schema.json' with { type: 'json' };
import simulation from '../contracts/simulation-contract.schema.json' with { type: 'json' };
import simApi from '../contracts/simulation-api.schema.json' with { type: 'json' };
import unified from '../contracts/unified.schema.json' with { type: 'json' };
import currentSimApi from '../contracts/current-simulation-api.schema.json' with { type: 'json' };
import type { ServerMessage, Observation, SourceStatus } from './generated/contract';
import type { ServerMessage as SimMessage, Observation as SimObservation } from './generated/simulationContract';

export type FeedMessage = ServerMessage | SimMessage;
export type RoomObservation = Observation | SimObservation;
export type RoomSource = SourceStatus | Extract<SimMessage, { type: 'source_status' }>['data'];
const ajv = new Ajv({ strict: false, validateFormats: false });
ajv.addSchema(monitoring, 'monitoring');
ajv.addSchema(simulation, 'simulation');
ajv.addSchema(simApi, 'simApi');
ajv.addSchema(unified, 'unified');
ajv.addSchema(currentSimApi, 'currentSimApi');
const unifiedMessage = ajv.compile({ $ref: 'unified#/$defs/ServerMessage' });
const monitorMessage = ajv.compile({ $ref: 'monitoring#/$defs/ServerMessage' });
const simulatorMessage = ajv.compile({ $ref: 'simulation#/$defs/ServerMessage' });

export function parseMessage(value: unknown): FeedMessage | null {
  return monitorMessage(value) || simulatorMessage(value) || unifiedMessage(value) ? value as FeedMessage : null;
}
export function validateApi<T>(name: string, value: unknown): T {
  const validate = ajv.getSchema(name) ?? ajv.compile({ $ref: name });
  const unifiedName = name.startsWith('monitoring#') ? name.replace('monitoring#', 'unified#')
    : name.startsWith('simApi#') ? name.replace('simApi#', 'currentSimApi#') : null;
  const current = unifiedName ? ajv.getSchema(unifiedName) ?? ajv.compile({ $ref: unifiedName }) : null;
  if (!validate(value) && !current?.(value)) throw new Error('Backend response does not match the agreed contract.');
  return value as T;
}

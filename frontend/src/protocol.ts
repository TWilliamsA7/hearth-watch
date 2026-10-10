import Ajv from 'ajv';
import monitoring from '../contracts/monitoring.schema.json' with { type: 'json' };
import simulation from '../contracts/simulation-contract.schema.json' with { type: 'json' };
import simApi from '../contracts/simulation-api.schema.json' with { type: 'json' };
import type { ServerMessage, Observation, SourceStatus } from './generated/contract';
import type { ServerMessage as SimMessage, Observation as SimObservation } from './generated/simulationContract';

export type FeedMessage = ServerMessage | SimMessage;
export type RoomObservation = Observation | SimObservation;
export type RoomSource = SourceStatus | Extract<SimMessage, { type: 'source_status' }>['data'];
const ajv = new Ajv({ strict: false, validateFormats: false });
ajv.addSchema(monitoring, 'monitoring');
ajv.addSchema(simulation, 'simulation');
ajv.addSchema(simApi, 'simApi');
const monitorMessage = ajv.compile({ $ref: 'monitoring#/$defs/ServerMessage' });
const simulatorMessage = ajv.compile({ $ref: 'simulation#/$defs/ServerMessage' });

export function parseMessage(value: unknown): FeedMessage | null {
  return monitorMessage(value) || simulatorMessage(value) ? value as FeedMessage : null;
}
export function validateApi<T>(name: string, value: unknown): T {
  const validate = ajv.getSchema(name) ?? ajv.compile({ $ref: name });
  if (!validate(value)) throw new Error('Backend response does not match the agreed contract.');
  return value as T;
}

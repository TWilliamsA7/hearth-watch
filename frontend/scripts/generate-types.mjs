import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { compile } from 'json-schema-to-typescript';

// Generate from the team's contract; do not maintain a second contract by hand.
const directory = new URL('../src/generated/', import.meta.url);
await mkdir(directory, { recursive: true });
for (const [schemaName, outputName] of [
  ['monitoring.schema.json', 'contract.ts'],
  ['simulation-contract.schema.json', 'simulationContract.ts'],
  ['simulation-api.schema.json', 'simulation.ts'],
]) {
  const schema = JSON.parse(await readFile(new URL(`../contracts/${schemaName}`, import.meta.url), 'utf8'));
  const output = await compile(schema, 'HearthSchema', { unreachableDefinitions: true });
  await writeFile(new URL(outputName, directory), output);
}

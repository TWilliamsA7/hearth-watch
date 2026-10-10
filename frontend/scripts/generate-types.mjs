import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { compile } from 'json-schema-to-typescript';

// Generate from the team's contract; do not maintain a second contract by hand.
const schema = JSON.parse(await readFile(new URL('../../shared/schema/hearth.schema.json', import.meta.url), 'utf8'));
const output = await compile(schema, 'HearthSchema', { unreachableDefinitions: true });
const directory = new URL('../src/generated/', import.meta.url);
await mkdir(directory, { recursive: true });
await writeFile(new URL('contract.ts', directory), output);

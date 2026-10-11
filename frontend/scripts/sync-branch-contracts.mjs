import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';

// Read teammate branches without checking them out or changing their files.
const files = [
  ['origin/monitoring', 'shared/schema/hearth.schema.json', 'monitoring.schema.json'],
  ['origin/feat/simulation', 'shared/schema/hearth.schema.json', 'simulation-contract.schema.json'],
  ['origin/feat/simulation', 'backend/schema/sim.schema.json', 'simulation-api.schema.json'],
  ['origin/monitoring', 'backend/tests/fixtures/gaston_edge.json', 'edge.fixture.json'],
];
const directory = new URL('../contracts/', import.meta.url);
await mkdir(directory, { recursive: true });
for (const [branch, path, name] of files) {
  const local = process.argv.includes('--local') && name !== 'edge.fixture.json';
  await writeFile(new URL(name, directory), local
    ? await readFile(new URL(`../../${path}`, import.meta.url))
    : execFileSync('git', ['show', `${branch}:${path}`]));
}
const branches = ['origin/monitoring', 'origin/gaston/edge', 'origin/feat/simulation'];
await writeFile(new URL('revisions.json', directory), JSON.stringify(Object.fromEntries(branches.map(branch =>
  [branch, execFileSync('git', ['rev-parse', branch], { encoding: 'utf8' }).trim()]
)), null, 2) + '\n');

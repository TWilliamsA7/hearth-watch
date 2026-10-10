// Generate TypeScript types from the JSON Schemas; never edit the output by hand.
//   shared/schema/hearth.schema.json  -> src/generated/contract.ts  (shared contract)
//   backend/schema/sim.schema.json    -> src/generated/sim.ts       (sim + evaluation API)
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { compile } from "json-schema-to-typescript";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const outDir = resolve(here, "../src/generated");

const targets = [
  { schema: "shared/schema/hearth.schema.json", out: "contract.ts", name: "HearthContract" },
  { schema: "backend/schema/sim.schema.json", out: "sim.ts", name: "HearthSimApi" },
];

// Pydantic titles every field ("Timestamp", "Room Id"), which makes the generator emit a
// named alias per field. Keep titles only on the models themselves.
// Keys under "properties" are field names (a field may be called "title"), not keywords.
function stripFieldTitles(node, isModel = false) {
  if (Array.isArray(node)) return node.map((n) => stripFieldTitles(n));
  if (node === null || typeof node !== "object") return node;
  const out = {};
  for (const [key, value] of Object.entries(node)) {
    if (key === "title" && !isModel) continue;
    out[key] =
      key === "properties"
        ? Object.fromEntries(Object.entries(value).map(([field, schema]) => [field, stripFieldTitles(schema)]))
        : stripFieldTitles(value);
  }
  return out;
}

await mkdir(outDir, { recursive: true });
for (const { schema, out, name } of targets) {
  const bundle = JSON.parse(await readFile(resolve(repo, schema), "utf8"));
  // The bundles keep every model under $defs; reference each one so all get emitted.
  const root = {
    title: name,
    type: "object",
    additionalProperties: false,
    properties: Object.fromEntries(Object.keys(bundle.$defs).map((k) => [k, { $ref: `#/$defs/${k}` }])),
    $defs: Object.fromEntries(Object.entries(bundle.$defs).map(([k, v]) => [k, stripFieldTitles(v, true)])),
  };
  const ts = await compile(root, name, {
    bannerComment: `/* Generated from ${schema} by scripts/gen-types.mjs. Do not edit. */`,
    additionalProperties: false,
    unreachableDefinitions: true,
    format: true,
  });
  await writeFile(resolve(outDir, out), ts);
  console.log(`wrote src/generated/${out}`);
}

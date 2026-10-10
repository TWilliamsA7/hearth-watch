import { readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

// Runs written by `python -m hearth.sim run` (git-ignored).
const RUNS_DIR = resolve(import.meta.dirname, "../../recordings/sim");

/** Dev-only endpoints that serve recorded sim runs to the dev page. */
function simRuns(): Plugin {
  return {
    name: "hearth-sim-runs",
    configureServer(server) {
      server.middlewares.use("/dev-api/runs", async (req, res) => {
        try {
          const path = (req.url ?? "/").split("?")[0];
          if (path === "/" || path === "") {
            const files = (await readdir(RUNS_DIR).catch(() => [])).filter((f) => f.endsWith(".run.json")).sort();
            const runs = await Promise.all(files.map(async (f) => JSON.parse(await readFile(resolve(RUNS_DIR, f), "utf8"))));
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(runs));
            return;
          }
          const match = /^\/([a-z0-9-]+-s\d+)\/(stream|truth)$/.exec(path);
          if (!match) {
            res.statusCode = 404;
            res.end();
            return;
          }
          const [, runId, kind] = match;
          res.setHeader("Content-Type", "application/x-ndjson");
          res.end(await readFile(resolve(RUNS_DIR, `${runId}.${kind}.jsonl`), "utf8"));
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });
    },
  };
}

export default defineConfig({
  root: resolve(import.meta.dirname, "dev"),
  plugins: [react(), simRuns()],
  server: { port: 5174 },
  test: { root: import.meta.dirname, include: ["src/**/*.test.ts"] },
});

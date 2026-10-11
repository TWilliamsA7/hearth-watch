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
          const match = /^\/([a-z0-9-]+-s\d+)\/(stream|truth|world)$/.exec(path);
          if (!match) {
            res.statusCode = 404;
            res.end();
            return;
          }
          const [, runId, kind] = match;
          const world = kind === "world"; // one JSON document; the others are JSON lines
          res.setHeader("Content-Type", world ? "application/json" : "application/x-ndjson");
          res.end(await readFile(resolve(RUNS_DIR, `${runId}.${kind}.${world ? "json" : "jsonl"}`), "utf8"));
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });
    },
  };
}

// The backend the dev pages talk to (Sim Studio's /sim/live routes, the room feed).
const BACKEND = process.env.HEARTH_BACKEND ?? "http://127.0.0.1:8000";
const proxy = Object.fromEntries(
  ["/sim", "/ws", "/eval", "/events", "/alerts", "/sources"].map((path) => [path, { target: BACKEND, ws: true, changeOrigin: true }]),
);

// `npm run build` makes the copy the Hearth server serves at /studio/ (Studio) and
// /studio/runs (run player), with /dev-api/runs answered by the server itself.
export default defineConfig(({ command }) => ({
  root: resolve(import.meta.dirname, "dev"),
  base: command === "build" ? "/studio/" : "/",
  plugins: [react(), simRuns()],
  server: { port: Number(process.env.PORT ?? 5174), proxy },
  build: {
    outDir: resolve(import.meta.dirname, "dist"),
    emptyOutDir: true,
    rollupOptions: {
      input: { index: resolve(import.meta.dirname, "dev/index.html"), studio: resolve(import.meta.dirname, "dev/studio.html") },
    },
  },
  test: { root: import.meta.dirname, include: ["src/**/*.test.ts"] },
}));

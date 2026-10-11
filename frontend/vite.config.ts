import { defineConfig } from 'vite';

// Normally the Hearth server serves the built dashboard itself (scripts/start.ps1).
// `start.ps1 -Dev` also runs this dev server for hot reload and forwards the API,
// the Sim Studio and the docs to that server.
const backend = process.env.HEARTH_BACKEND ?? 'http://127.0.0.1:8000';

// The local renderer has its own development React dependency; share the app's
// React instance when importing the linked package.
export default defineConfig({
  resolve: { dedupe: ['react', 'react-dom'] },
  server: {
    host: '127.0.0.1', port: 5173, strictPort: true,
    // Preserve Host and Origin: local recognition checks still apply through Vite.
    proxy: Object.fromEntries([
      '/appointments', '/alerts', '/events', '/sources', '/rooms', '/health',
      '/ingest', '/help-requests', '/nurse-sessions', '/ws', '/sim', '/eval',
      '/camera-preview', '/face-demo', '/studio', '/dev-api', '/docs', '/openapi.json',
    ].map(path => [path, { target: backend, ws: true }])),
  },
});

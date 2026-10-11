import { defineConfig } from 'vite';

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
      '/face-demo', '/face-check-in', '/assets/face-check-in.js', '/assets/style.css',
    ].map(path => [path, { target: 'http://127.0.0.1:8002', ws: true }])),
  },
});

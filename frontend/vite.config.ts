import { defineConfig } from 'vite';

// The local renderer has its own development React dependency; share the app's
// React instance when importing the linked package.
export default defineConfig({ resolve: { dedupe: ['react', 'react-dom'] } });

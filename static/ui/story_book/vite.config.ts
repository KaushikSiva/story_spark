import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Optional: proxy static assets if you decide to request them relatively
      '/static': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
});

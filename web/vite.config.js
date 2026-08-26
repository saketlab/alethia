import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Bind to every interface so the app is reachable from another machine by IP, not just
// from a browser on this host.
//
// Note on speed: ONNX Runtime Web uses multi-threaded WASM only in a cross-origin
// isolated, secure context. Reaching this over plain http://<ip> is neither, so it
// falls back to a single thread, correct, just slower. localhost is exempt from the
// secure-context rule, and a real deployment behind HTTPS with the COOP/COEP headers
// below gets threading back.
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
  worker: { format: 'es' },
  build: { target: 'es2022', sourcemap: true },
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // Vite refuses requests whose Host header it does not recognise; without this an
    // access by bare IP is rejected with "Blocked request".
    allowedHosts: true,
    headers: crossOriginIsolation,
  },
  preview: {
    host: '0.0.0.0',
    port: 4173,
    strictPort: true,
    allowedHosts: true,
    headers: crossOriginIsolation,
  },
});

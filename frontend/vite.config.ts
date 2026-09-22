import { defineConfig } from "vite";

// `npm run dev` proxies /api to the Flask server (`python scripts/run_server.py`), so the frontend can be
// developed with hot reload without rebuilding; `npm run build` writes the static bundle Flask serves in
// production (see ai_alarm/system.py, FRONTEND_DIR).
export default defineConfig({
  server: {
    proxy: { "/api": "http://localhost:5000" },
  },
  build: {
    outDir: "dist",
  },
});

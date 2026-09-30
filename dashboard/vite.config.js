import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 127.0.0.1, not "localhost": on Windows, Node resolves localhost to IPv6 (::1)
// first, which can reach a different program (e.g. an old Docker container)
// listening on the same port instead of your local API.
const api = process.env.VITE_API_PROXY || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": api, "/health": api, "/predict": api },
  },
});

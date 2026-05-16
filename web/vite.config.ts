import path from "node:path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

const API_TARGET = process.env.LORAHUB_API_TARGET ?? "http://127.0.0.1:18765"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 6006,
    // Accept any Host header so the Vite dev server is reachable through
    // SSH/k8s/cloud port-forwards (e.g. damodel maps 6006 -> external 48357,
    // and vite would otherwise 403 the external hostname).
    allowedHosts: true,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
})

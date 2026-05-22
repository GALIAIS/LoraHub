import { execSync } from "node:child_process"
import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const API_TARGET = process.env.LORAHUB_API_TARGET ?? "http://127.0.0.1:18765"

/**
 * Resolve the user-facing version string at build time.
 *
 * Priority:
 *  1. `git describe --tags --dirty --always`, e.g. `v0.3.0`,
 *     `v0.3.0-12-g1a2b3c4`, or `1a2b3c4-dirty` if the user is between
 *     tags / on a fresh clone with no tags yet.
 *  2. `web/package.json`'s `version` field — kept around as a fallback
 *     for environments where git history isn't available (CI tarball
 *     downloads, vendored bundles).
 *  3. The literal `dev` so the UI never crashes on a missing string.
 *
 * The leading `v` from a git tag is stripped so the runtime can
 * append it consistently (`v{__APP_VERSION__}`).
 */
function resolveAppVersion(): string {
  try {
    const raw = execSync("git describe --tags --dirty --always", {
      cwd: path.resolve(__dirname, ".."),
      stdio: ["ignore", "pipe", "ignore"],
    })
      .toString()
      .trim()
    if (raw) return raw.replace(/^v/, "")
  } catch {
    // git not installed, not a checkout, or no tags reachable.
  }
  try {
    const pkg = JSON.parse(
      readFileSync(path.resolve(__dirname, "package.json"), "utf-8"),
    )
    if (typeof pkg.version === "string" && pkg.version) return pkg.version
  } catch {
    // package.json missing — fall through to literal.
  }
  return "dev"
}

const APP_VERSION = resolveAppVersion()

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    __APP_VERSION__: JSON.stringify(APP_VERSION),
  },
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
    // ``hidden`` keeps source maps next to the bundle (so a debugger
    // can resolve frames if it picks them up) without referencing
    // them from the ship-able .js — production stack traces stay
    // unreadable to anyone poking at the network panel, but a
    // developer with the dist tree on disk still gets pretty traces.
    sourcemap: "hidden",
    // Split the vendor stack into chunks roughly grouped by "owns its
    // own update cadence". The default rollup chunking lumped every
    // dependency into the entry chunk, which weighed in around 500 KB
    // gzipped — every minor route change (the SPA does most rendering
    // through lazy-loaded route chunks) re-shipped the whole vendor
    // bundle to the browser.
    //
    // Grouping rules:
    //   * react / router stay in their own chunk because they're on
    //     every route.
    //   * lucide-react ships ~1k icon paths; isolating it keeps an
    //     icon-only edit (e.g. swapping ``Plus`` for ``X``) from
    //     invalidating the rest of the vendor.
    //   * markdown is only used by analysis / about pages — its own
    //     chunk so the rest of the app can skip it on cold start.
    //   * @base-ui, @radix, react-virtuoso are heavy headless UI
    //     primitives; isolating them keeps a shadcn-component tweak
    //     from re-shipping unrelated icon / markdown bytes.
    //   * tanstack query is its own line so devtools (dev only) and
    //     the runtime stay grouped.
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          icons: ["lucide-react"],
          markdown: ["react-markdown", "remark-gfm"],
          ui: ["@base-ui/react", "@radix-ui/react-context-menu", "react-virtuoso"],
          query: ["@tanstack/react-query"],
        },
      },
    },
  },
})

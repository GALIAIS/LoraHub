import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ReactQueryDevtools } from "@tanstack/react-query-devtools"

import App from "./App"
import "./index.css"
import { ErrorBoundary } from "./components/error-boundary"
import { installDynamicImportRecovery } from "./lib/dynamic-import-recovery"
import { installGlobalErrorHandlers } from "./lib/install-global-error-handlers"
import { clearChunkReloadGuard, lazyWithRetry } from "./lib/lazy-with-retry"

// Code-split every page route. Suspense lives inside <App> so the
// shell stays mounted while a chunk downloads (see App.tsx). The big
// wins are image-studio (virtualised grid + smart-caption modals),
// configs (the schema-driven form), and analysis (the chart stack)
// — without splitting they all rode the initial 1.2 MB bundle.
const DashboardPage = lazyWithRetry(
  () => import("./pages/dashboard").then((m) => ({ default: m.DashboardPage })),
  "route:dashboard",
)
const JobsPage = lazyWithRetry(
  () => import("./pages/jobs").then((m) => ({ default: m.JobsPage })),
  "route:jobs",
)
const AnalysisPage = lazyWithRetry(
  () => import("./pages/analysis").then((m) => ({ default: m.AnalysisPage })),
  "route:analysis",
)
const SweepsPage = lazyWithRetry(
  () => import("./pages/sweeps").then((m) => ({ default: m.SweepsPage })),
  "route:sweeps",
)
const ConfigsPage = lazyWithRetry(
  () => import("./pages/configs").then((m) => ({ default: m.ConfigsPage })),
  "route:configs",
)
const DatasetsPage = lazyWithRetry(
  () => import("./pages/datasets").then((m) => ({ default: m.DatasetsPage })),
  "route:datasets",
)
const ImageStudioPage = lazyWithRetry(
  () => import("./pages/image-studio").then((m) => ({ default: m.ImageStudioPage })),
  "route:image-studio",
)
const ImageStudioToolPage = lazyWithRetry(
  () =>
    import("./pages/image-studio/tool-page").then((m) => ({ default: m.ToolPage })),
  "route:image-studio-tool",
)
const GalleryPage = lazyWithRetry(
  () => import("./pages/gallery").then((m) => ({ default: m.GalleryPage })),
  "route:gallery",
)
const SettingsPage = lazyWithRetry(
  () => import("./pages/settings").then((m) => ({ default: m.SettingsPage })),
  "route:settings",
)
const AboutPage = lazyWithRetry(
  () => import("./pages/about").then((m) => ({ default: m.AboutPage })),
  "route:about",
)
const TerminalPage = lazyWithRetry(
  () => import("./pages/terminal").then((m) => ({ default: m.TerminalPage })),
  "route:terminal",
)
const ArtifactsPage = lazyWithRetry(
  () => import("./pages/artifacts").then((m) => ({ default: m.ArtifactsPage })),
  "route:artifacts",
)

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

installDynamicImportRecovery()
installGlobalErrorHandlers()

createRoot(document.getElementById("root")!, {
  onUncaughtError(error, errorInfo) {
    console.error("React root uncaught error:", error, errorInfo)
  },
  onCaughtError(error, errorInfo) {
    console.error("React root caught error:", error, errorInfo)
  },
  onRecoverableError(error, errorInfo) {
    console.warn("React root recoverable error:", error, errorInfo)
  },
}).render(
  <StrictMode>
    <ErrorBoundary reporterSource="frontend.render">
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route element={<App />}>
              <Route index element={<DashboardPage />} />
              <Route path="jobs" element={<JobsPage />} />
              <Route path="analysis" element={<AnalysisPage />} />
              <Route path="analysis/compare" element={<AnalysisPage />} />
              <Route path="analysis/:jobId" element={<AnalysisPage />} />
              <Route path="sweeps" element={<SweepsPage />} />
              <Route path="configs" element={<ConfigsPage />} />
              <Route path="datasets" element={<DatasetsPage />} />
              <Route path="image-studio" element={<ImageStudioPage />} />
              <Route
                path="image-studio/tools/:toolId"
                element={<ImageStudioToolPage />}
              />
              <Route path="gallery" element={<GalleryPage />} />
              <Route path="terminal" element={<TerminalPage />} />
              <Route path="artifacts" element={<ArtifactsPage />} />
              <Route path="settings" element={<SettingsPage />} />
              <Route path="about" element={<AboutPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
        {/* Tree-shaken in production: import.meta.env.DEV is statically
            replaced with `false` at build time, so the devtools chunk
            and its bundled deps drop out of the prod bundle entirely. */}
        {import.meta.env.DEV && (
          <ReactQueryDevtools
            initialIsOpen={false}
            buttonPosition="bottom-right"
          />
        )}
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
)

// Boot succeeded — clear the reload guard so a future stale-chunk
// event can reload again.
clearChunkReloadGuard()

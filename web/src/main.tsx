import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

import App from "./App"
import "./index.css"
import { clearChunkReloadGuard, lazyWithRetry } from "./lib/lazy-with-retry"

// Code-split every page route. Suspense lives inside <App> so the
// shell stays mounted while a chunk downloads (see App.tsx). The big
// wins are image-studio (virtualised grid + smart-caption modals),
// configs (the schema-driven form), and analysis (the chart stack)
// — without splitting they all rode the initial 1.2 MB bundle.
//
// `lazyWithRetry` recovers from stale-chunk errors after a deploy by
// hard-reloading once when a chunk import fails — see
// `lib/lazy-with-retry.ts`.
const DashboardPage = lazyWithRetry(() =>
  import("./pages/dashboard").then((m) => ({ default: m.DashboardPage })),
)
const JobsPage = lazyWithRetry(() =>
  import("./pages/jobs").then((m) => ({ default: m.JobsPage })),
)
const AnalysisPage = lazyWithRetry(() =>
  import("./pages/analysis").then((m) => ({ default: m.AnalysisPage })),
)
const SweepsPage = lazyWithRetry(() =>
  import("./pages/sweeps").then((m) => ({ default: m.SweepsPage })),
)
const ConfigsPage = lazyWithRetry(() =>
  import("./pages/configs").then((m) => ({ default: m.ConfigsPage })),
)
const DatasetsPage = lazyWithRetry(() =>
  import("./pages/datasets").then((m) => ({ default: m.DatasetsPage })),
)
const ImageStudioPage = lazyWithRetry(() =>
  import("./pages/image-studio").then((m) => ({ default: m.ImageStudioPage })),
)
const GalleryPage = lazyWithRetry(() =>
  import("./pages/gallery").then((m) => ({ default: m.GalleryPage })),
)
const SettingsPage = lazyWithRetry(() =>
  import("./pages/settings").then((m) => ({ default: m.SettingsPage })),
)
const AboutPage = lazyWithRetry(() =>
  import("./pages/about").then((m) => ({ default: m.AboutPage })),
)

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

// Catch chunk-load rejections that escape lazyWithRetry — anything that
// uses raw `import()` (eg. on-demand third-party widgets) falls through
// to the global handler.
window.addEventListener("unhandledrejection", (ev) => {
  const reason = ev.reason
  const msg = reason instanceof Error ? reason.message : String(reason ?? "")
  const looksStale =
    msg.includes("Failed to fetch dynamically imported module") ||
    msg.includes("Importing a module script failed") ||
    /Loading chunk \S+ failed/.test(msg)
  if (!looksStale) return
  const flag = "lorahub:chunk-reload"
  if (sessionStorage.getItem(flag) === "1") return
  sessionStorage.setItem(flag, "1")
  ev.preventDefault()
  window.location.reload()
})

createRoot(document.getElementById("root")!).render(
  <StrictMode>
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
            <Route path="gallery" element={<GalleryPage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="about" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)

// Boot succeeded — clear the reload guard so a future stale-chunk
// event can reload again.
clearChunkReloadGuard()

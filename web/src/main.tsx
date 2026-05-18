import { lazy, StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

import App from "./App"
import "./index.css"

// Code-split every page route. Suspense lives inside <App> so the
// shell stays mounted while a chunk downloads (see App.tsx). The big
// wins are image-studio (virtualised grid + smart-caption modals),
// configs (the schema-driven form), and analysis (the chart stack)
// — without splitting they all rode the initial 1.2 MB bundle.
//
// Each `lazy()` call rewrites the named export into the default export
// `React.lazy` requires. If we ever flip the page modules to default
// exports this collapses to a one-liner per page.
const DashboardPage = lazy(() =>
  import("./pages/dashboard").then((m) => ({ default: m.DashboardPage })),
)
const JobsPage = lazy(() =>
  import("./pages/jobs").then((m) => ({ default: m.JobsPage })),
)
const AnalysisPage = lazy(() =>
  import("./pages/analysis").then((m) => ({ default: m.AnalysisPage })),
)
const SweepsPage = lazy(() =>
  import("./pages/sweeps").then((m) => ({ default: m.SweepsPage })),
)
const ConfigsPage = lazy(() =>
  import("./pages/configs").then((m) => ({ default: m.ConfigsPage })),
)
const DatasetsPage = lazy(() =>
  import("./pages/datasets").then((m) => ({ default: m.DatasetsPage })),
)
const ImageStudioPage = lazy(() =>
  import("./pages/image-studio").then((m) => ({ default: m.ImageStudioPage })),
)
const GalleryPage = lazy(() =>
  import("./pages/gallery").then((m) => ({ default: m.GalleryPage })),
)
const SettingsPage = lazy(() =>
  import("./pages/settings").then((m) => ({ default: m.SettingsPage })),
)
const AboutPage = lazy(() =>
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

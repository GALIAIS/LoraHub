import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

import App from "./App"
import { JobsPage } from "./pages/jobs"
import { DashboardPage } from "./pages/dashboard"
import { ConfigsPage } from "./pages/configs"
import { DatasetsPage } from "./pages/datasets"
import { GalleryPage } from "./pages/gallery"
import { SettingsPage } from "./pages/settings"
import { SweepsPage } from "./pages/sweeps"
import { AboutPage } from "./pages/about"
import "./index.css"

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
            <Route path="sweeps" element={<SweepsPage />} />
            <Route path="configs" element={<ConfigsPage />} />
            <Route path="datasets" element={<DatasetsPage />} />
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

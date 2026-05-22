import { importWithDynamicImportRecovery } from "@/lib/dynamic-import-recovery"

export type AppRouteModuleKey =
  | "dashboard"
  | "jobs"
  | "analysis"
  | "sweeps"
  | "configs"
  | "datasets"
  | "image-studio"
  | "gallery"
  | "settings"
  | "about"
  | "terminal"
  | "artifacts"

type AppRouteImporter = () => Promise<unknown>

export const appRouteImporters: Record<AppRouteModuleKey, AppRouteImporter> = {
  dashboard: () => import("@/pages/dashboard"),
  jobs: () => import("@/pages/jobs"),
  analysis: () => import("@/pages/analysis"),
  sweeps: () => import("@/pages/sweeps"),
  configs: () => import("@/pages/configs"),
  datasets: () => import("@/pages/datasets"),
  "image-studio": () => import("@/pages/image-studio"),
  gallery: () => import("@/pages/gallery"),
  settings: () => import("@/pages/settings"),
  about: () => import("@/pages/about"),
  terminal: () => import("@/pages/terminal"),
  artifacts: () => import("@/pages/artifacts"),
}

const inflight = new Map<AppRouteModuleKey, Promise<unknown>>()

export function preloadAppRoute(routeKey: AppRouteModuleKey) {
  const cached = inflight.get(routeKey)
  if (cached) return cached
  const promise = importWithDynamicImportRecovery(
    appRouteImporters[routeKey],
    `route:${routeKey}`,
  )
  inflight.set(routeKey, promise)
  return promise
}

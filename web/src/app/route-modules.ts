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
}

export function preloadAppRoute(routeKey: AppRouteModuleKey) {
  if (typeof window !== "undefined") {
    console.info("[router] preload route chunk", routeKey)
  }
  return importWithDynamicImportRecovery(
    appRouteImporters[routeKey],
    `route:${routeKey}`,
  )
}

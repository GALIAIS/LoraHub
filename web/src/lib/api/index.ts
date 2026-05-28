// Barrel re-export for the API client module. Callers continue to
// import from "@/lib/api" (or "../lib/api"); the relative directory
// resolves to this file thanks to ESM extensionless / index.ts
// resolution in Vite + tsc.

export * from "./core"
export * from "./backends"
export * from "./jobs"
export * from "./configs"
export * from "./sweeps"
export * from "./settings"
export * from "./datasets"
export * from "./models"
export * from "./tagging"
export * from "./network"
export * from "./ai"
export * from "./system"
export * from "./image-studio"
export * from "./terminal"
export * from "./error-reports"
export * from "./wandb"
export * from "./client"

/**
 * `React.lazy` wrapper that recovers from stale chunk references after a
 * deploy. When the server replaces `dist/` the hashed JS chunk filenames
 * change; any tab still holding the old `index.html` will throw on its
 * next route navigation:
 *
 *     Failed to fetch dynamically imported module:
 *     https://.../assets/index-<oldHash>.js
 *
 * Strategy:
 *   1. Try the import once.
 *   2. If it fails, retry once after a short delay (covers transient
 *      network blips).
 *   3. If the retry also fails, hand recovery to the shared dynamic
 *      import recovery helper.
 */
import { lazy, type ComponentType } from "react"
import {
  clearDynamicImportRecovery,
  importWithDynamicImportRecovery,
  isDynamicImportRecoveryError,
} from "./dynamic-import-recovery"

export function lazyWithRetry<T extends ComponentType<any>>(
  factory: () => Promise<{ default: T }>,
  scope = "route:unknown",
): React.LazyExoticComponent<T> {
  return lazy(async () => {
    try {
      return await factory()
    } catch {
      // Single retry — covers a transient blip without forcing a reload.
      try {
        await new Promise((r) => setTimeout(r, 200))
        return await importWithDynamicImportRecovery(factory, scope)
      } catch (second) {
        if (isDynamicImportRecoveryError(second)) {
          // Recovery orchestration moved into
          // `dynamic-import-recovery.ts`; if we are still here, that
          // module already decided not to reload again in this window.
          clearDynamicImportRecovery(scope)
        }
        throw second
      }
    }
  })
}

// Clear the global guards once the app has successfully booted, so the
// next stale-chunk event can trigger another reload.
export function clearChunkReloadGuard(): void {
  clearDynamicImportRecovery("unhandledrejection")
  clearDynamicImportRecovery("vite-preload")
}

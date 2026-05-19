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
 *   3. If the retry also fails AND the error matches a chunk-load
 *      pattern, hard-reload the page once. A sessionStorage flag
 *      prevents reload loops if the failure is genuine (offline,
 *      server down, …).
 */
import { lazy, type ComponentType } from "react"

const RELOAD_FLAG = "lorahub:chunk-reload"

function isChunkLoadError(err: unknown): boolean {
  if (!(err instanceof Error)) return false
  const msg = err.message || ""
  // Vite / native dynamic-import / webpack chunk-load shapes.
  return (
    msg.includes("Failed to fetch dynamically imported module") ||
    msg.includes("Importing a module script failed") ||
    msg.includes("error loading dynamically imported module") ||
    /Loading chunk \S+ failed/.test(msg) ||
    msg.includes("ChunkLoadError")
  )
}

export function lazyWithRetry<T extends ComponentType<any>>(
  factory: () => Promise<{ default: T }>,
): React.LazyExoticComponent<T> {
  return lazy(async () => {
    try {
      return await factory()
    } catch (first) {
      // Single retry — covers a transient blip without forcing a reload.
      try {
        await new Promise((r) => setTimeout(r, 200))
        return await factory()
      } catch (second) {
        if (isChunkLoadError(second)) {
          const alreadyReloaded =
            sessionStorage.getItem(RELOAD_FLAG) === "1"
          if (!alreadyReloaded) {
            sessionStorage.setItem(RELOAD_FLAG, "1")
            window.location.reload()
            // Return a never-resolving promise so React doesn't render
            // an error boundary in the split-second before reload.
            return new Promise<{ default: T }>(() => {})
          }
          // Reload already attempted — clear the flag so a future
          // genuine deploy can retry, then re-throw to surface the
          // error boundary.
          sessionStorage.removeItem(RELOAD_FLAG)
        }
        throw second
      }
    }
  })
}

// Clear the reload guard once the app has successfully booted, so the
// next stale-chunk event can trigger another reload. Called from
// `main.tsx` after `createRoot().render()`.
export function clearChunkReloadGuard(): void {
  sessionStorage.removeItem(RELOAD_FLAG)
}

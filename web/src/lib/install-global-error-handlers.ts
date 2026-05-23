/**
 * Hook into ``window`` so every uncaught JS error / unhandled promise
 * rejection ends up in the backend's error registry. Boot once from
 * ``main.tsx``.
 */

import { reportError } from "./error-reporter"

export function installGlobalErrorHandlers(): void {
  // Avoid double-install on HMR re-execution of the boot module.
  if ((window as unknown as { __lorahubErrorHandlersInstalled?: boolean })
      .__lorahubErrorHandlersInstalled) {
    return
  }
  ;(window as unknown as { __lorahubErrorHandlersInstalled?: boolean })
    .__lorahubErrorHandlersInstalled = true

  window.addEventListener("error", (event) => {
    const err = event.error
    const isChunkLoad =
      err instanceof Error &&
      (err.name === "ChunkLoadError" ||
        /Loading chunk \d+ failed/i.test(err.message))
    void reportError({
      severity: isChunkLoad ? "warn" : "error",
      source: "frontend.runtime",
      category: isChunkLoad ? "chunk_load" : "uncaught",
      title: event.message || "uncaught error",
      message:
        err instanceof Error
          ? `${err.name}: ${err.message}`
          : String(err ?? event.message),
      stack: err instanceof Error ? err.stack ?? null : null,
      context: {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        href: window.location.href,
      },
      requestPath: window.location.pathname,
    })
  })

  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason
    let message: string
    let stack: string | null = null
    let name = "UnhandledRejection"
    if (reason instanceof Error) {
      message = `${reason.name}: ${reason.message}`
      stack = reason.stack ?? null
      name = reason.name
    } else if (typeof reason === "string") {
      message = reason
    } else {
      try {
        message = JSON.stringify(reason)
      } catch {
        message = String(reason)
      }
    }
    void reportError({
      severity: "error",
      source: "frontend.runtime",
      category: "unhandled_rejection",
      title: name,
      message,
      stack,
      context: { href: window.location.href },
      requestPath: window.location.pathname,
    })
  })
}

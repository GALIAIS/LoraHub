const DYNAMIC_IMPORT_RECOVERY_PREFIX = "lorahub:dynamic-import-recovery"
const DYNAMIC_IMPORT_RECOVERY_WINDOW_MS = 15_000

function getRecoveryStorageKey(scope: string) {
  return `${DYNAMIC_IMPORT_RECOVERY_PREFIX}:${scope}`
}

function readLastRecoveryAt(scope: string) {
  if (typeof window === "undefined") return 0
  const rawValue = window.sessionStorage.getItem(getRecoveryStorageKey(scope))
  const parsedValue = Number(rawValue)
  return Number.isFinite(parsedValue) ? parsedValue : 0
}

function writeRecoveryAt(scope: string, timestamp: number) {
  if (typeof window === "undefined") return
  window.sessionStorage.setItem(getRecoveryStorageKey(scope), String(timestamp))
}

export function clearDynamicImportRecovery(scope: string) {
  if (typeof window === "undefined") return
  window.sessionStorage.removeItem(getRecoveryStorageKey(scope))
}

export function isDynamicImportRecoveryError(error: unknown) {
  const normalizedMessage =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : ""

  if (!normalizedMessage) return false

  return (
    /failed to fetch dynamically imported module/iu.test(normalizedMessage) ||
    /importing a module script failed/iu.test(normalizedMessage) ||
    /error loading dynamically imported module/iu.test(normalizedMessage) ||
    /loading chunk \S+ failed/iu.test(normalizedMessage) ||
    /chunkloaderror/iu.test(normalizedMessage) ||
    /outdated optimize dep/iu.test(normalizedMessage)
  )
}

export function tryRecoverDynamicImportError(error: unknown, scope = "global") {
  if (typeof window === "undefined" || !isDynamicImportRecoveryError(error)) {
    return false
  }

  const now = Date.now()
  const lastRecoveryAt = readLastRecoveryAt(scope)
  if (now - lastRecoveryAt < DYNAMIC_IMPORT_RECOVERY_WINDOW_MS) {
    return false
  }

  writeRecoveryAt(scope, now)
  window.location.reload()
  return true
}

export async function importWithDynamicImportRecovery<T>(
  loader: () => Promise<T>,
  scope: string,
) {
  try {
    const module = await loader()
    clearDynamicImportRecovery(scope)
    return module
  } catch (error) {
    if (tryRecoverDynamicImportError(error, scope)) {
      return new Promise<T>(() => {})
    }
    throw error
  }
}

let dynamicImportRecoveryInstalled = false

export function installDynamicImportRecovery() {
  if (typeof window === "undefined" || dynamicImportRecoveryInstalled) return
  dynamicImportRecoveryInstalled = true

  window.addEventListener("vite:preloadError", (event) => {
    if (tryRecoverDynamicImportError(event.payload, "vite-preload")) {
      event.preventDefault()
    }
  })

  window.addEventListener("unhandledrejection", (event) => {
    if (tryRecoverDynamicImportError(event.reason, "unhandledrejection")) {
      event.preventDefault()
    }
  })
}

// Tiny wrapper around react-router's useSearchParams that lifts common
// "set this key, drop empties" patterns out of the page components so
// each page reads cleanly. ``null`` or ``""`` deletes the key (lets
// callers round-trip "default value → no URL noise"); ``undefined``
// leaves the key untouched, which is handy when one writer wants to
// patch only some keys.
import { useCallback } from "react"
import { useSearchParams } from "react-router-dom"

export type UrlPatch = Record<string, string | null | undefined>

export function useUrlState() {
  const [params, setParams] = useSearchParams()

  const update = useCallback(
    (patch: UrlPatch, opts?: { replace?: boolean }) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          for (const [k, v] of Object.entries(patch)) {
            if (v === undefined) continue
            if (v === null || v === "") {
              next.delete(k)
            } else {
              next.set(k, v)
            }
          }
          return next
        },
        opts?.replace ? { replace: true } : undefined,
      )
    },
    [setParams],
  )

  return { params, update }
}

// Boolean params live in the URL as ``?key=1``. Anything else (missing,
// empty, "0", "false") reads as false.
export function readBool(params: URLSearchParams, key: string): boolean {
  const v = params.get(key)
  return v === "1" || v === "true"
}

// CSV list helper for compact id arrays in the URL. Empty strings are
// dropped, so ``?compare_ids=`` parses as ``[]``.
export function readList(params: URLSearchParams, key: string): string[] {
  const raw = params.get(key)
  if (!raw) return []
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
}

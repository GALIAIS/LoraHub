/**
 * Tween a number over a short duration via requestAnimationFrame.
 *
 * Designed for analysis KPIs: when a verdict's headline value
 * changes (e.g. SNR moved from -1.2 to -1.8 after a fresh metric
 * batch), we want the number to roll up to its new value over
 * ~200 ms instead of jumping. The hook is conservative:
 *
 *   - First render returns the target value as-is. We only animate
 *     subsequent updates so the page doesn't slow-mo into existence.
 *   - When the new target equals the previous one (a refetch with
 *     identical data), no animation runs.
 *   - When prefers-reduced-motion is on, we just snap to the target.
 *   - When the caller passes a non-finite number (NaN / Infinity)
 *     the hook returns it as-is — interpolating doesn't make sense.
 *
 * The eased curve is the same cubic-bezier(0.22, 1, 0.36, 1) used by
 * the page-level fade-ins so number motion reads as part of the
 * same family.
 */
import { useEffect, useRef, useState } from "react"

const EASE = (t: number): number => {
  // cubic-bezier(0.22, 1, 0.36, 1) — out-quint feel.
  const x = 1 - t
  return 1 - x * x * x * x * x
}

export function useAnimatedNumber(
  target: number,
  durationMs: number = 220,
): number {
  const [value, setValue] = useState(target)
  const fromRef = useRef(target)
  const isFirstRef = useRef(true)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    if (isFirstRef.current) {
      isFirstRef.current = false
      fromRef.current = target
      setValue(target)
      return
    }
    if (!Number.isFinite(target)) {
      setValue(target)
      fromRef.current = target
      return
    }
    if (target === fromRef.current) return
    if (
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      setValue(target)
      fromRef.current = target
      return
    }
    const from = fromRef.current
    const to = target
    const start = performance.now()
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
    const tick = (now: number): void => {
      const t = Math.min(1, (now - start) / durationMs)
      const v = from + (to - from) * EASE(t)
      setValue(v)
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        fromRef.current = to
        rafRef.current = null
      }
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
  }, [target, durationMs])

  return value
}

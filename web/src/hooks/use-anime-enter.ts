import { useEffect, useRef, type DependencyList } from "react"
import { animate, type JSAnimation } from "animejs/animation"

export function useAnimeEnter<T extends HTMLElement>(
  deps: DependencyList = [],
) {
  const ref = useRef<T | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return

    let animation: JSAnimation | null = animate(el, {
      opacity: [0, 1],
      y: [4, 0],
      duration: 180,
      ease: "outCubic",
      composition: "replace",
    })
    return () => {
      animation?.revert()
      animation = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return ref
}

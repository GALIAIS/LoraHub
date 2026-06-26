import { animate, type JSAnimation } from "animejs/animation"
import { useLayoutEffect, useRef, type DependencyList } from "react"

function readDelayMs(el: Element): number {
  const raw = window
    .getComputedStyle(el)
    .getPropertyValue("--stagger-delay")
    .trim()
  if (!raw) return 0
  if (raw.endsWith("ms")) return Number.parseFloat(raw) || 0
  if (raw.endsWith("s")) return (Number.parseFloat(raw) || 0) * 1000
  return Number.parseFloat(raw) || 0
}

export function useAnimeAnalysisMotion<T extends HTMLElement>(
  deps: DependencyList = [],
) {
  const ref = useRef<T | null>(null)

  useLayoutEffect(() => {
    const root = ref.current
    if (!root) return
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return

    const animations: JSAnimation[] = []
    const enterTargets = root.querySelectorAll<HTMLElement>(
      ".analysis-fade-in, .analysis-fade-in-stagger",
    )
    enterTargets.forEach((el) => {
      el.style.opacity = "0"
      el.style.transform = "translate3d(0, 6px, 0)"
      animations.push(
        animate(el, {
          opacity: 1,
          y: 0,
          duration: 260,
          delay: readDelayMs(el),
          ease: "outCubic",
          composition: "replace",
        }),
      )
    })

    const barTargets = root.querySelectorAll<HTMLElement>(".analysis-bar-fill")
    barTargets.forEach((el) => {
      el.style.transformOrigin = "left center"
      el.style.transform = "scaleX(0)"
      animations.push(
        animate(el, {
          scaleX: 1,
          duration: 420,
          delay: readDelayMs(el),
          ease: "outCubic",
          composition: "replace",
        }),
      )
    })

    return () => {
      for (const animation of animations) animation.revert()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return ref
}

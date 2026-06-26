import { useEffect, useRef, type DependencyList } from "react"
import { animate, type JSAnimation } from "animejs/animation"

export function useAnimeThemeTransition(deps: DependencyList) {
  const firstRunRef = useRef(true)
  const animationRef = useRef<JSAnimation | null>(null)

  useEffect(() => {
    if (firstRunRef.current) {
      firstRunRef.current = false
      return
    }

    const root = document.documentElement
    if (root.dataset.viewTransitionInProgress === "true") return
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return

    root.style.setProperty("--theme-transition-opacity", "0.18")
    const state = { opacity: 0.18 }
    animationRef.current?.cancel()
    animationRef.current = animate(state, {
      opacity: 0,
      duration: 220,
      ease: "outCubic",
      onRender: () => {
        root.style.setProperty(
          "--theme-transition-opacity",
          String(state.opacity),
        )
      },
      onComplete: () => {
        root.style.removeProperty("--theme-transition-opacity")
        animationRef.current = null
      },
    })

    return () => {
      animationRef.current?.cancel()
      animationRef.current = null
      root.style.removeProperty("--theme-transition-opacity")
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}

import { useCallback, useEffect, useRef, useState } from "react"
import {
  PAD_LEFT,
  PAD_RIGHT,
  VIEW_W,
} from "./loss-chart-model"

type FullExtent = { xMin: number; xMax: number } | null
type SelectRect = { x0: number; x1: number } | null

export function useLossChartInteraction({
  persistKey,
  fullExtent,
  xMin,
  xMax,
  innerW,
}: {
  persistKey?: string | null
  fullExtent: FullExtent
  xMin: number
  xMax: number
  innerW: number
}) {
  const storageKey = persistKey ? `lorahub.loss.${persistKey}` : null
  const svgRef = useRef<SVGSVGElement | null>(null)
  const panRef = useRef<{ lastVX: number } | null>(null)
  const [hoverX, setHoverX] = useState<number | null>(null)
  const [selectMode, setSelectMode] = useState(false)
  const [selectRect, setSelectRect] = useState<SelectRect>(null)
  const [yLog, setYLog] = useState<boolean>(() => {
    if (!storageKey) return false
    try {
      const raw = window.sessionStorage.getItem(storageKey)
      if (raw) return JSON.parse(raw)?.yLog === true
    } catch {
      // Ignore corrupt storage.
    }
    return false
  })
  const [viewRange, setViewRange] = useState<[number, number] | null>(() => {
    if (!storageKey) return null
    try {
      const raw = window.sessionStorage.getItem(storageKey)
      if (!raw) return null
      const parsed = JSON.parse(raw)
      const xr = parsed?.xRange
      if (Array.isArray(xr) && xr.length === 2 && xr.every(Number.isFinite)) {
        return [xr[0], xr[1]]
      }
    } catch {
      // Ignore corrupt storage.
    }
    return null
  })
  const currentXMin = viewRange?.[0] ?? xMin
  const currentXMax = viewRange?.[1] ?? xMax

  useEffect(() => {
    if (!storageKey) return
    try {
      window.sessionStorage.setItem(
        storageKey,
        JSON.stringify({
          xRange: viewRange,
          yLog,
        }),
      )
    } catch {
      // Quota exceeded or disabled — silently skip.
    }
  }, [storageKey, viewRange, yLog])

  const clientToViewBox = useCallback(
    (event: React.PointerEvent | React.WheelEvent): number => {
      const svg = svgRef.current
      if (!svg) return 0
      const rect = svg.getBoundingClientRect()
      return ((event.clientX - rect.left) / rect.width) * VIEW_W
    },
    [],
  )

  const setRangeClamped = useCallback(
    (lo: number, hi: number) => {
      if (!fullExtent) return
      const span = hi - lo
      if (span <= 0) return

      const fullSpan = fullExtent.xMax - fullExtent.xMin
      const minSpan = fullSpan * 0.005
      if (span < minSpan) return

      let nextLo = Math.max(fullExtent.xMin, lo)
      let nextHi = Math.min(fullExtent.xMax, hi)
      if (nextHi - nextLo < minSpan) {
        const center = (nextLo + nextHi) / 2
        nextLo = center - minSpan / 2
        nextHi = center + minSpan / 2
      }

      if (nextLo === fullExtent.xMin && nextHi === fullExtent.xMax) {
        setViewRange(null)
      } else {
        setViewRange([nextLo, nextHi])
      }
    },
    [fullExtent],
  )

  const inverseX = useCallback(
    (px: number) =>
      currentXMin + ((px - PAD_LEFT) / innerW) * (currentXMax - currentXMin),
    [currentXMin, currentXMax, innerW],
  )

  const zoomBy = useCallback(
    (factor: number, anchorVX?: number) => {
      const anchor =
        anchorVX != null ? inverseX(anchorVX) : (currentXMin + currentXMax) / 2
      const span = (currentXMax - currentXMin) / factor
      setRangeClamped(
        anchor -
          span * ((anchor - currentXMin) / (currentXMax - currentXMin || 1)),
        anchor +
          span * ((currentXMax - anchor) / (currentXMax - currentXMin || 1)),
      )
    },
    [currentXMin, currentXMax, inverseX, setRangeClamped],
  )

  const reset = useCallback(() => {
    setViewRange(null)
    setSelectRect(null)
  }, [])

  const onWheel = useCallback(
    (event: React.WheelEvent<SVGSVGElement>) => {
      if (!fullExtent) return
      event.preventDefault()
      const factor = event.deltaY < 0 ? 1.25 : 1 / 1.25
      zoomBy(factor, clientToViewBox(event))
    },
    [clientToViewBox, fullExtent, zoomBy],
  )

  const onPointerDown = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (event.button !== 0) return
      const vx = clientToViewBox(event)
      const insideChart = vx >= PAD_LEFT && vx <= VIEW_W - PAD_RIGHT
      if (!insideChart) return
      if (selectMode || event.shiftKey) {
        setSelectRect({ x0: vx, x1: vx })
        event.currentTarget.setPointerCapture(event.pointerId)
        return
      }
      panRef.current = { lastVX: vx }
      event.currentTarget.setPointerCapture(event.pointerId)
    },
    [clientToViewBox, selectMode],
  )

  const onPointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      const vx = clientToViewBox(event)
      setHoverX(vx)
      if (selectRect) {
        setSelectRect({ x0: selectRect.x0, x1: vx })
        return
      }
      if (panRef.current) {
        const dx = vx - panRef.current.lastVX
        panRef.current.lastVX = vx
        const dataDx = -(dx / innerW) * (currentXMax - currentXMin)
        setRangeClamped(currentXMin + dataDx, currentXMax + dataDx)
      }
    },
    [
      clientToViewBox,
      currentXMin,
      currentXMax,
      innerW,
      selectRect,
      setRangeClamped,
    ],
  )

  const onPointerUp = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      panRef.current = null
      if (selectRect) {
        const x0 = Math.min(selectRect.x0, selectRect.x1)
        const x1 = Math.max(selectRect.x0, selectRect.x1)
        if (Math.abs(x1 - x0) > 4) {
          setRangeClamped(inverseX(x0), inverseX(x1))
        }
        setSelectRect(null)
      }
      event.currentTarget.releasePointerCapture(event.pointerId)
    },
    [inverseX, selectRect, setRangeClamped],
  )

  const onPointerLeave = useCallback(() => {
    setHoverX(null)
    panRef.current = null
  }, [])

  return {
    hoverX,
    onDoubleClick: reset,
    onPointerDown,
    onPointerLeave,
    onPointerMove,
    onPointerUp,
    onWheel,
    reset,
    selectMode,
    selectRect,
    setSelectMode,
    setYLog,
    svgRef,
    viewRange,
    viewXMax: currentXMax,
    viewXMin: currentXMin,
    yLog,
    zoomBy,
  }
}

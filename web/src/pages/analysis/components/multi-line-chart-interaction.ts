import { useCallback, useEffect, useRef, useState } from "react"

type FullExtent = { xMin: number; xMax: number } | null

export function useMultiLineChartInteraction({
  persistKey,
  fullExtent,
  innerW,
  padLeft,
  padRight,
  viewW,
}: {
  persistKey?: string | null
  fullExtent: FullExtent
  innerW: number
  padLeft: number
  padRight: number
  viewW: number
}) {
  const storageKey = persistKey ? `lorahub.multi.${persistKey}` : null
  const svgRef = useRef<SVGSVGElement | null>(null)
  const panRef = useRef<{ lastVX: number } | null>(null)
  const [hoverX, setHoverX] = useState<number | null>(null)
  const [viewRange, setViewRange] = useState<[number, number] | null>(() => {
    if (!storageKey) return null
    try {
      const raw = window.sessionStorage.getItem(storageKey)
      if (!raw) return null
      const parsed = JSON.parse(raw)
      if (
        Array.isArray(parsed?.xRange) &&
        parsed.xRange.length === 2 &&
        parsed.xRange.every(Number.isFinite)
      ) {
        return [parsed.xRange[0], parsed.xRange[1]]
      }
    } catch {
      // ignore corrupt storage
    }
    return null
  })

  const baseRange = fullExtent
    ? ([fullExtent.xMin, fullExtent.xMax] as [number, number])
    : ([0, 1] as [number, number])
  const xMin = viewRange?.[0] ?? baseRange[0]
  const xMax = viewRange?.[1] ?? baseRange[1]
  const xSpan = xMax - xMin || 1

  useEffect(() => {
    if (!storageKey) return
    try {
      window.sessionStorage.setItem(
        storageKey,
        JSON.stringify({ xRange: viewRange }),
      )
    } catch {
      // quota or disabled — silently skip
    }
  }, [storageKey, viewRange])

  const clientToViewBox = useCallback(
    (event: React.PointerEvent | React.WheelEvent): number => {
      const svg = svgRef.current
      if (!svg) return 0
      const rect = svg.getBoundingClientRect()
      return ((event.clientX - rect.left) / rect.width) * viewW
    },
    [viewW],
  )

  const inverseX = useCallback(
    (px: number) => xMin + ((px - padLeft) / innerW) * xSpan,
    [innerW, padLeft, xMin, xSpan],
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

  const zoomBy = useCallback(
    (factor: number, anchorVX?: number) => {
      const anchor = anchorVX != null ? inverseX(anchorVX) : (xMin + xMax) / 2
      const span = (xMax - xMin) / factor
      setRangeClamped(
        anchor - span * ((anchor - xMin) / (xMax - xMin || 1)),
        anchor + span * ((xMax - anchor) / (xMax - xMin || 1)),
      )
    },
    [inverseX, setRangeClamped, xMin, xMax],
  )

  const reset = useCallback(() => {
    setViewRange(null)
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
      if (vx < padLeft || vx > viewW - padRight) return
      panRef.current = { lastVX: vx }
      event.currentTarget.setPointerCapture(event.pointerId)
    },
    [clientToViewBox, padLeft, padRight, viewW],
  )

  const onPointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      const vx = clientToViewBox(event)
      setHoverX(vx)
      if (panRef.current) {
        const dx = vx - panRef.current.lastVX
        panRef.current.lastVX = vx
        const dataDx = -(dx / innerW) * xSpan
        setRangeClamped(xMin + dataDx, xMax + dataDx)
      }
    },
    [clientToViewBox, innerW, setRangeClamped, xMin, xMax, xSpan],
  )

  const onPointerUp = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      panRef.current = null
      event.currentTarget.releasePointerCapture(event.pointerId)
    },
    [],
  )

  const onPointerLeave = useCallback(() => {
    panRef.current = null
    setHoverX(null)
  }, [])

  return {
    hoverX,
    inverseX,
    onDoubleClick: reset,
    onPointerDown,
    onPointerLeave,
    onPointerMove,
    onPointerUp,
    onWheel,
    reset,
    svgRef,
    viewRange,
    xMax,
    xMin,
    xSpan,
    zoomBy,
  }
}

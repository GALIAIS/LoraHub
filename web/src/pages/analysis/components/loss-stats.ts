/**
 * Robust statistics for loss curves.
 *
 * Diffusion-model training loss is high-variance noise around a slowly
 * moving signal. Plain moving averages get tugged around by outliers;
 * sliding-window median + IQR is the right smoother. These helpers also
 * back the changepoint analysis (`pelt.ts`) and the effectiveness
 * panel's stage classifier.
 *
 * All functions take an array of `{ step, loss }` and return one or
 * more new arrays. They never mutate the input. The medians use a
 * sliding sorted window via insertion-sort because the windows are
 * small (≤ a few dozen points) and the constant factor beats heavier
 * data structures here.
 */

export interface XYPoint {
  step: number
  loss: number
}

export interface BandPoint {
  step: number
  lo: number
  hi: number
}

/**
 * Default window length, expressed as a fraction of the total sample
 * count, with a hard floor and ceiling. Tuned so a 200-step run gets
 * a 16-step window (smooth but responsive) and a 5000-step run gets a
 * 100-step window (stable, doesn't over-smooth).
 */
export function defaultRollingWindow(n: number): number {
  return Math.min(Math.max(8, Math.floor(n * 0.05)), 100)
}

/**
 * Sliding-window quantiles (Q25 / Q50 / Q75) over a series. Returns
 * three arrays of the same length as input — leading samples receive
 * partial-window estimates so the chart isn't blank at the start.
 */
export function rollingQuartiles(
  points: XYPoint[],
  window: number = defaultRollingWindow(points.length),
): { median: XYPoint[]; band: BandPoint[] } {
  const n = points.length
  if (n === 0) return { median: [], band: [] }
  const window_ = Math.max(1, Math.min(window, n))
  const sorted: number[] = []
  const queue: number[] = []
  const median: XYPoint[] = []
  const band: BandPoint[] = []
  for (let i = 0; i < n; i += 1) {
    const v = points[i].loss
    queue.push(v)
    insertSorted(sorted, v)
    if (queue.length > window_) {
      const removed = queue.shift() as number
      removeFromSorted(sorted, removed)
    }
    const q25 = quantile(sorted, 0.25)
    const q50 = quantile(sorted, 0.5)
    const q75 = quantile(sorted, 0.75)
    median.push({ step: points[i].step, loss: q50 })
    band.push({ step: points[i].step, lo: q25, hi: q75 })
  }
  return { median, band }
}

/**
 * Median absolute deviation — a robust dispersion measurement. Unlike
 * std, MAD doesn't blow up when a few high-loss steps appear. Returns
 * a sliding-window MAD aligned to the same X axis as the input.
 */
export function rollingMad(
  points: XYPoint[],
  window: number = defaultRollingWindow(points.length),
): XYPoint[] {
  const n = points.length
  if (n === 0) return []
  const window_ = Math.max(2, Math.min(window, n))
  const out: XYPoint[] = []
  for (let i = 0; i < n; i += 1) {
    const lo = Math.max(0, i - window_ + 1)
    const slice = points.slice(lo, i + 1).map((p) => p.loss)
    const med = quantile(sortedCopy(slice), 0.5)
    const dev = sortedCopy(slice.map((v) => Math.abs(v - med)))
    out.push({ step: points[i].step, loss: quantile(dev, 0.5) })
  }
  return out
}

/**
 * OLS slope of `loss` vs `step` on the trailing window. Negative means
 * loss is still falling. Used by the effectiveness panel's stability
 * verdict in place of CoV.
 */
export function trailingSlope(
  points: XYPoint[],
  window: number = defaultRollingWindow(points.length),
): { slope: number; intercept: number; rSquared: number; samples: number } | null {
  if (points.length < 4) return null
  const n = Math.min(window, points.length)
  const slice = points.slice(-n)
  const meanX =
    slice.reduce((a, p) => a + p.step, 0) / slice.length
  const meanY = slice.reduce((a, p) => a + p.loss, 0) / slice.length
  let num = 0
  let den = 0
  let sst = 0
  for (const p of slice) {
    const dx = p.step - meanX
    const dy = p.loss - meanY
    num += dx * dy
    den += dx * dx
    sst += dy * dy
  }
  if (den === 0 || sst === 0) return null
  const slope = num / den
  const intercept = meanY - slope * meanX
  let ssr = 0
  for (const p of slice) {
    const pred = slope * p.step + intercept
    ssr += (p.loss - pred) ** 2
  }
  const rSquared = 1 - ssr / sst
  return { slope, intercept, rSquared, samples: slice.length }
}

/* ----------------- internal helpers ----------------- */

function insertSorted(arr: number[], v: number): void {
  // Binary search for insertion point.
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (arr[mid] < v) lo = mid + 1
    else hi = mid
  }
  arr.splice(lo, 0, v)
}

function removeFromSorted(arr: number[], v: number): void {
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (arr[mid] < v) lo = mid + 1
    else hi = mid
  }
  if (arr[lo] === v) arr.splice(lo, 1)
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN
  if (sorted.length === 1) return sorted[0]
  const idx = (sorted.length - 1) * q
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

function sortedCopy(arr: number[]): number[] {
  return [...arr].sort((a, b) => a - b)
}

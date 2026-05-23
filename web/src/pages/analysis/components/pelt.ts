/**
 * PELT (Pruned Exact Linear Time) changepoint detection for univariate
 * series. Used to slice the loss curve into homogeneous segments —
 * each cut point typically lines up with a real training transition
 * (warm-up end, LR drop, schedule plateau, divergence onset), giving
 * the analysis page a principled answer to "where did the dynamics
 * actually change".
 *
 * Cost function: sum of squared deviations from the segment mean. We
 * precompute prefix sums of y and y² so any segment cost is O(1):
 *
 *   cost(i, j) = (Σy²)[j] − ((Σy)[j])² / (j − i + 1)
 *                 with a small +ε for numerical stability when a
 *                 segment is exactly constant.
 *
 * Penalty: β = 2·log(N)·σ² (BIC-flavoured) — robust to scale, doesn't
 * require the user to tune anything. Diffusion losses tend to have
 * very stable variance, so this rarely needs tweaking, but the helper
 * accepts a `penalty` override for callers that want to.
 *
 * The implementation lifts directly from Killick, Fearnhead, Eckley
 * (2012) "Optimal detection of changepoints with a linear computational
 * cost", with the standard pruning step.
 */

export interface ChangePointResult {
  /** Indices into the input series where a new segment starts (segment 0 always begins at 0). */
  starts: number[]
  /** Total cost of the optimal segmentation (mostly informational). */
  totalCost: number
}

const EPSILON = 1e-9

export function pelt(
  values: number[],
  options: {
    /** Override the BIC-style automatic penalty. */
    penalty?: number
    /** Refuse changepoints that would create a segment shorter than this. */
    minSegment?: number
  } = {},
): ChangePointResult {
  const n = values.length
  if (n < 4) return { starts: [0], totalCost: 0 }
  const minSeg = Math.max(2, options.minSegment ?? Math.max(4, Math.floor(n * 0.04)))

  // Prefix sums of y and y² so segment cost is O(1).
  const cumY = new Float64Array(n + 1)
  const cumY2 = new Float64Array(n + 1)
  for (let i = 0; i < n; i += 1) {
    cumY[i + 1] = cumY[i] + values[i]
    cumY2[i + 1] = cumY2[i] + values[i] * values[i]
  }
  // cost(i..j-1) — half-open interval [i, j).
  const cost = (i: number, j: number): number => {
    const len = j - i
    if (len <= 0) return 0
    const sumY = cumY[j] - cumY[i]
    const sumY2 = cumY2[j] - cumY2[i]
    return sumY2 - (sumY * sumY) / (len + EPSILON)
  }

  // Default penalty: 2·log(N)·σ̂². σ̂² is the variance of the residuals
  // assuming a single segment — a coarse estimator but fine for setting
  // the scale.
  let penalty = options.penalty
  if (penalty == null) {
    const meanAll = cumY[n] / n
    let varAll = 0
    for (let i = 0; i < n; i += 1) varAll += (values[i] - meanAll) ** 2
    varAll = varAll / Math.max(1, n - 1)
    penalty = 2 * Math.log(n) * Math.max(varAll, EPSILON)
  }

  // F[i] = optimal cost of segmenting values[0..i).
  const F = new Float64Array(n + 1)
  // bestPrev[i] = best previous changepoint that yields F[i].
  const bestPrev = new Int32Array(n + 1)
  // Pruning candidate set R for each i — start with the trivial "split
  // at 0" baseline.
  let R: number[] = [0]
  F[0] = -penalty // canonical PELT initialisation
  for (let i = minSeg; i <= n; i += 1) {
    let bestCost = Infinity
    let bestT = 0
    for (const t of R) {
      if (i - t < minSeg) continue
      const c = F[t] + cost(t, i) + penalty
      if (c < bestCost) {
        bestCost = c
        bestT = t
      }
    }
    F[i] = bestCost
    bestPrev[i] = bestT
    // Pruning step: drop candidates whose lower bound already exceeds
    // the current best.
    R = R.filter((t) => F[t] + cost(t, i) <= F[i])
    R.push(i)
  }

  // Backtrack from n to 0 collecting segment starts.
  const starts: number[] = []
  let cursor = n
  while (cursor > 0) {
    const t = bestPrev[cursor]
    starts.push(t)
    if (t === 0) break
    cursor = t
  }
  starts.reverse()
  if (starts[0] !== 0) starts.unshift(0)
  return { starts, totalCost: F[n] }
}

export interface PeltSegment {
  /** Index into the source series for the first sample of this segment. */
  startIndex: number
  /** Index of the last sample (inclusive). */
  endIndex: number
  /** Step value at startIndex. */
  startStep: number
  /** Step value at endIndex. */
  endStep: number
  /** OLS slope of loss vs step within this segment. */
  slope: number
  /** Mean loss within this segment. */
  meanLoss: number
  /** Stage classification — derived from slope + position in the run. */
  stage: "warmup" | "converging" | "plateau" | "diverging"
}

export interface PeltAnalysis {
  segments: PeltSegment[]
  /** Step values where a new segment begins (excluding 0). */
  changepointSteps: number[]
}

/**
 * Run PELT on the series and convert the index-based segmentation into
 * a step-aware structure with stage labels.
 */
export function analyseChangepoints(
  points: { step: number; loss: number }[],
): PeltAnalysis {
  if (points.length < 6) {
    return { segments: [], changepointSteps: [] }
  }
  const values = points.map((p) => p.loss)
  const { starts } = pelt(values)

  const segments: PeltSegment[] = []
  for (let i = 0; i < starts.length; i += 1) {
    const startIdx = starts[i]
    const endIdx = (i + 1 < starts.length ? starts[i + 1] : points.length) - 1
    if (endIdx <= startIdx) continue
    const slice = points.slice(startIdx, endIdx + 1)
    const meanLoss = slice.reduce((a, p) => a + p.loss, 0) / slice.length
    const meanStep = slice.reduce((a, p) => a + p.step, 0) / slice.length
    let num = 0
    let den = 0
    for (const p of slice) {
      const dx = p.step - meanStep
      num += dx * (p.loss - meanLoss)
      den += dx * dx
    }
    const slope = den === 0 ? 0 : num / den
    segments.push({
      startIndex: startIdx,
      endIndex: endIdx,
      startStep: points[startIdx].step,
      endStep: points[endIdx].step,
      slope,
      meanLoss,
      stage: "converging", // overwritten below
    })
  }

  // Stage assignment uses both segment-local slope and position. The
  // first segment of a long run is treated as warm-up if it shows a
  // strong negative slope (rapid initial fall). A late-run segment
  // with strongly positive slope is divergence.
  if (segments.length > 0) {
    // Normalise slope by the run's overall loss scale so the
    // thresholds are comparable across runs of any magnitude.
    const lossScale =
      Math.max(...values) - Math.min(...values) || Math.abs(values[0]) || 1
    for (let i = 0; i < segments.length; i += 1) {
      const seg = segments[i]
      const stepSpan = seg.endStep - seg.startStep || 1
      const normalised = (seg.slope * stepSpan) / lossScale
      if (normalised > 0.05) {
        seg.stage = "diverging"
      } else if (normalised < -0.1) {
        seg.stage = i === 0 ? "warmup" : "converging"
      } else if (normalised < -0.02) {
        seg.stage = "converging"
      } else {
        seg.stage = "plateau"
      }
    }
  }

  const changepointSteps = segments.slice(1).map((s) => s.startStep)
  return { segments, changepointSteps }
}

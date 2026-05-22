/**
 * Playwright SPA-aware perf test for the LoraHub workbench.
 *
 * The previous version used `page.goto()` for every page hop, which
 * is a full document reload — not what happens when a user clicks a
 * sidebar link. This version drives the SPA via in-app navigation
 * (sidebar clicks for the rapid-swap phase) and uses a one-time
 * mount measurement only for the cold-start tour.
 *
 * Boots a Chromium tab, captures console + page errors + failed
 * network requests, and walks the workbench through:
 *
 *   1. Cold start: load /, then visit each top-level route once via
 *      sidebar clicks. Records first-paint deltas.
 *   2. Rapid swap: 30 seconds of pseudo-random sidebar clicks at
 *      a click cadence the React Router can keep up with. Measures
 *      route-change render time per swap.
 *   3. Heap diff (start vs end) and DOM-node growth diff to surface
 *      slow leaks.
 *
 * ERR_ABORTED on requests is normal during route-driven cancellation
 * (TanStack Query cancels in-flight queries when their consumer
 * unmounts) and isn't counted as fatal.
 */
import { chromium } from "playwright"

const BASE = process.env.LORAHUB_URL ?? "http://127.0.0.1:18765"
const RAPID_DURATION_MS = 30_000
const CLICK_INTERVAL_MS = 200

// Sidebar entries — match the actual NAV in src/App.tsx (label is
// the visible text, urlPath is the route the link goes to). We
// intentionally omit /image-studio per the standing instruction
// that the image studio is out of scope for general perf work.
const ROUTES = [
  { label: "数据面板",     urlPath: "/" },
  { label: "训练任务",     urlPath: "/jobs" },
  { label: "训练分析",     urlPath: "/analysis" },
  { label: "超参 sweep",   urlPath: "/sweeps" },
  { label: "终端",        urlPath: "/terminal" },
  { label: "训练配置",     urlPath: "/configs" },
  { label: "数据集",      urlPath: "/datasets" },
  { label: "样图画廊",     urlPath: "/gallery" },
  { label: "设置",        urlPath: "/settings" },
  { label: "关于",        urlPath: "/about" },
]

function ms(t) { return `${t.toFixed(0)}ms` }

async function takeHeapMb(page) {
  return await page.evaluate(() => {
    if (!performance.memory) return null
    return {
      used: performance.memory.usedJSHeapSize,
      total: performance.memory.totalJSHeapSize,
    }
  })
}

async function domNodeCount(page) {
  return await page.evaluate(() => document.getElementsByTagName("*").length)
}

// Click a sidebar entry by label and wait for the route to actually
// commit. The path-prefix match handles redirects (e.g. /jobs vs
// /jobs/<latest>) — except for "/", where a startsWith match is
// always true; for that one we wait on equality.
async function clickRoute(page, route) {
  const t = performance.now()
  await page.getByRole("link", { name: route.label, exact: false }).first().click()
  await page.waitForFunction(
    (path) => {
      if (path === "/") return location.pathname === "/"
      return location.pathname.startsWith(path)
    },
    route.urlPath,
    { timeout: 5000 },
  )
  await page.evaluate(() => new Promise(r => requestAnimationFrame(() => r())))
  return performance.now() - t
}

// Counters
const consoleErrors = []
const pageErrors = []
const failedRequests = []
const abortedRequests = []
const requestsByPath = new Map()
let totalNetwork = 0

const browser = await chromium.launch({ headless: true })
const ctx = await browser.newContext()
const page = await ctx.newPage()

page.on("console", (msg) => {
  const type = msg.type()
  if (type === "error" || type === "warning") {
    const text = msg.text()
    if (text.includes("[HMR]") || text.includes("vite")) return
    consoleErrors.push(`[${type}] ${text}`)
  }
})
page.on("pageerror", (err) => pageErrors.push(`${err.name}: ${err.message}`))
page.on("requestfailed", (req) => {
  const f = req.failure()
  const text = f?.errorText ?? "unknown"
  // ERR_ABORTED on route swap is expected — TanStack Query cancels
  // in-flight queries whose consumer unmounted.
  if (text.includes("ABORTED")) {
    abortedRequests.push(req.url())
  } else {
    failedRequests.push(`${req.method()} ${req.url()} ${text}`)
  }
})
page.on("request", (req) => {
  if (!req.url().startsWith(BASE)) return
  totalNetwork++
  const u = new URL(req.url())
  const key = u.pathname
  requestsByPath.set(key, (requestsByPath.get(key) ?? 0) + 1)
})

// ---------- Cold start ----------
console.log(`Boot ${BASE}`)
const t0 = performance.now()
await page.goto(BASE, { waitUntil: "domcontentloaded" })
const tBoot = performance.now() - t0
console.log(`  domcontentloaded ${ms(tBoot)}`)

// Wait for the first route render to settle. We don't use
// `networkidle` because the SSE channel keeps the network "busy"
// indefinitely. Instead wait for the sidebar to render — that's the
// signal that the React tree has mounted.
await page.waitForSelector("a", { timeout: 10_000 })
const heapStart = await takeHeapMb(page)
const domStart = await domNodeCount(page)

// ---------- Step 1: cold tour ----------
console.log("\nStep 1 — cold tour, 1× each via sidebar click")
console.log(`  ${"page".padEnd(10)} switch     dom    heap-mb`)
const tourSummary = []
for (const route of ROUTES) {
  try {
    const elapsed = await clickRoute(page, route)
    const dom = await domNodeCount(page)
    const heap = await takeHeapMb(page)
    const heapMb = heap ? (heap.used / 1e6).toFixed(1) : "—"
    tourSummary.push({ label: route.label, ms: elapsed, dom, heapMb })
    console.log(`  ${route.label.padEnd(10)} ${ms(elapsed).padEnd(10)} ${String(dom).padEnd(6)} ${heapMb}`)
  } catch (e) {
    console.log(`  ${route.label.padEnd(10)} FAIL: ${e.message}`)
  }
}

// ---------- Step 2: rapid swap ----------
console.log(`\nStep 2 — rapid swap (${RAPID_DURATION_MS / 1000}s, ${CLICK_INTERVAL_MS}ms cadence)`)
const switchTimes = []
const deadline = Date.now() + RAPID_DURATION_MS
let i = 0
while (Date.now() < deadline) {
  const route = ROUTES[i++ % ROUTES.length]
  try {
    const elapsed = await clickRoute(page, route)
    switchTimes.push(elapsed)
  } catch (e) {
    console.log(`  switch to ${route.label} failed: ${e.message}`)
    break
  }
  await page.waitForTimeout(CLICK_INTERVAL_MS)
}
switchTimes.sort((a, b) => a - b)
const p50 = switchTimes[Math.floor(switchTimes.length * 0.5)] ?? 0
const p95 = switchTimes[Math.floor(switchTimes.length * 0.95)] ?? 0
const max = switchTimes[switchTimes.length - 1] ?? 0
console.log(`  ${switchTimes.length} switches`)
console.log(`  p50=${ms(p50)}  p95=${ms(p95)}  max=${ms(max)}`)

// ---------- Diagnostics ----------
const heapEnd = await takeHeapMb(page)
const domEnd = await domNodeCount(page)
console.log("\nResource diff")
if (heapStart && heapEnd) {
  const sMb = heapStart.used / 1e6
  const eMb = heapEnd.used / 1e6
  console.log(`  heap   start=${sMb.toFixed(1)} MB  end=${eMb.toFixed(1)} MB  Δ=${(eMb - sMb).toFixed(1)} MB`)
}
console.log(`  dom    start=${domStart}  end=${domEnd}  Δ=${domEnd - domStart}`)

console.log(`\nNetwork totals: ${totalNetwork} requests`)
const top = [...requestsByPath.entries()]
  .sort((a, b) => b[1] - a[1])
  .slice(0, 12)
for (const [path, n] of top) console.log(`  ${String(n).padStart(4)} × ${path}`)

console.log(`\nConsole warns/errors: ${consoleErrors.length}`)
for (const e of consoleErrors.slice(0, 8)) console.log(`  ${e}`)
console.log(`\nUncaught page errors: ${pageErrors.length}`)
for (const e of pageErrors) console.log(`  ${e}`)
console.log(`\nFailed (non-aborted) requests: ${failedRequests.length}`)
for (const e of failedRequests.slice(0, 5)) console.log(`  ${e}`)
console.log(`\nAborted (expected on swap): ${abortedRequests.length}`)

await browser.close()

const fatal = pageErrors.length > 0 || failedRequests.length > 0 || consoleErrors.length > 0
process.exit(fatal ? 1 : 0)

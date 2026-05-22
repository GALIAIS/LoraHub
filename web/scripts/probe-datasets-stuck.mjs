// Probe specifically the "click datasets, then navigate elsewhere"
// scenario the user reported. After clicking datasets, the URL changes
// on subsequent nav clicks but the page stays on datasets until reload.
//
// Strategy:
//   1. Visit /, confirm header reads 数据面板.
//   2. Visit /datasets, wait until the dataset scan UI is visible.
//   3. Click another nav item (training tasks). Capture URL + visible
//      content. Repeat for several targets.
//   4. Capture full console + page errors so we can correlate the
//      stuck-render with whatever React threw.

import { chromium } from "playwright"

const BASE = "http://127.0.0.1:6006"

const ts = () => new Date().toISOString().slice(11, 23)
const log = (...a) => console.log(`[${ts()}]`, ...a)

const consoleEvents = []
const pageErrors = []
const requestsFailed = []

const browser = await chromium.launch({ headless: false, devtools: true })
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } })
const page = await ctx.newPage()

page.on("console", async (msg) => {
  const text = msg.text()
  if (msg.type() === "error" || msg.type() === "warning") {
    // React passes the component stack as additional args. Resolve
    // them to strings so we can see *which* component is looping.
    let extras = []
    try {
      extras = await Promise.all(
        msg.args().map((a) =>
          a.evaluate((v) => (typeof v === "string" ? v : "")).catch(() => ""),
        ),
      )
    } catch {}
    consoleEvents.push({
      type: msg.type(),
      text,
      extras: extras.filter(Boolean).map((s) => s.slice(0, 1500)),
      at: ts(),
    })
  }
})
page.on("pageerror", (err) => {
  pageErrors.push({ message: err.message, stack: err.stack?.slice(0, 1500), at: ts() })
})
page.on("requestfailed", (req) => {
  requestsFailed.push({ url: req.url(), failure: req.failure()?.errorText, at: ts() })
})

await ctx.addInitScript(() => {
  const realError = console.error.bind(console)
  console.error = (...args) => {
    const text = String(args[0] ?? "")
    if (text.includes("Maximum update depth exceeded")) {
      // Throw a real Error so playwright captures the stack via
      // page.on('pageerror'). One throw is enough — set a flag so
      // the loop doesn't drown the listener.
      // eslint-disable-next-line no-undef
      if (!window.__throttledThrow) {
        // eslint-disable-next-line no-undef
        window.__throttledThrow = true
        const err = new Error(`Captured by probe: ${text}`)
        // Walk the call stack (Error stack) and tack on as much
        // context as we can find.
        setTimeout(() => {
          throw err
        }, 0)
      }
    }
    return realError(...args)
  }
})

await page.goto(BASE, { waitUntil: "domcontentloaded" })
await page.waitForSelector('[data-sidebar="menu-button"]')
log("loaded")

const titleSel = "header .truncate.text-base.font-semibold"

async function snapshot(tag) {
  const url = new URL(page.url()).pathname
  const headerTitle = (await page
    .locator(titleSel)
    .first()
    .textContent()
    .catch(() => null))?.trim() ?? ""
  // Probe the visible content of the route, not just the toolbar:
  //   - The dataset page renders a card titled "选择数据集"
  //   - The jobs page renders a column titled "训练任务" or shows the job list
  // We capture every distinct H1/H2/H3 we find so the report shows
  // *what* page is actually painted.
  const h1 = await page.locator("h1, h2, h3").allTextContents().catch(() => [])
  log(`[${tag}] url=${url} | header="${headerTitle}" | headings=${JSON.stringify(h1.slice(0, 6))}`)
  return { url, headerTitle, headings: h1.slice(0, 6) }
}

await snapshot("init /")

// Probe sequence — same scenario the user described.
async function clickAndProbe(label, tag) {
  await page.getByRole("link", { name: label, exact: true }).first().click()
  // Wait for the toolbar title to actually flip — that's the moment
  // React has committed the new route. 4s is generous: a normal flip
  // is <100ms; only a stuck render or a still-loading lazy chunk
  // takes longer than this.
  try {
    await page.waitForFunction(
      (expected) => {
        const el = document.querySelector(
          "header .truncate.text-base.font-semibold",
        )
        return (el?.textContent ?? "").trim() === expected
      },
      label,
      { timeout: 4_000 },
    )
  } catch {}
  return await snapshot(tag)
}

await clickAndProbe("数据面板", "click 数据面板")
await clickAndProbe("训练任务", "click 训练任务")
await clickAndProbe("数据集", "ENTER datasets")

// Wait for the datasets card to actually paint so any side-effect
// (queries, scan request) has the chance to start.
await page.waitForTimeout(1200)
await snapshot("after datasets settle")

// Now click around — these are the ones the user said get stuck.
await clickAndProbe("训练任务", "click jobs after datasets")
await clickAndProbe("训练分析", "click analysis after datasets")
await clickAndProbe("数据面板", "click dashboard after datasets")
await clickAndProbe("训练配置", "click configs after datasets")

console.log("\n========== ERRORS ==========")
console.log("page errors:", pageErrors.length)
for (const e of pageErrors) console.log(JSON.stringify(e, null, 2))
console.log("console errors/warnings:", consoleEvents.length)
for (const e of consoleEvents.slice(0, 40)) console.log(" ", e)
console.log("requests failed:", requestsFailed.length)
for (const e of requestsFailed.slice(0, 20)) console.log(" ", e)

await page.screenshot({ path: "scripts/datasets-stuck.png", fullPage: false })
log("screenshot saved")

await page.waitForTimeout(2000)
await browser.close()

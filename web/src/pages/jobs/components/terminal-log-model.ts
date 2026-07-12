import type { TrainingEvent } from "@/lib/api"
import { stripAnsi } from "./ansi"

export type LogTone =
  | "default"
  | "debug"
  | "error"
  | "warn"
  | "progress"
  | "gpu"
  | "checkpoint"
  | "sample"
  | "epoch"
  | "done"

export type LogFilter = "training" | "all" | "warning" | "error" | "gpu"

export interface LogLine {
  ts: number
  level: string
  message: string
  source: string | null
  stream: string | null
  tone: LogTone
  eventType: string
  key: string
  progressKey: string | null
  legacyRich: boolean
}

const LEGACY_RICH_RE = /^(?:\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+)?(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s+(.+?)(?:\s{2,}([^\s]+\.py:\d+))?$/
const PROGRESS_RE = /^(.{1,80}?):\s+(\d+)%\|.*?\|\s*(\d+)\/(\d+)\s*\[([^\]]+)\]/
const TRACEBACK_START_RE = /^(?:Exception in thread\b|Traceback \(most recent call last\):)/
const TRACEBACK_END_RE = /^(?:[A-Za-z_][\w.]*?(?:Error|Exception|Interrupt)|SystemExit):\s*/

function number(payload: Record<string, unknown>, key: string): number | null {
  const value = payload[key]
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function gib(mib: number | null): string {
  return mib === null ? "--" : (mib / 1024).toFixed(1)
}

function eventMessage(event: TrainingEvent, fallbackTotalSteps: number | null): string {
  const p = event.payload
  if (typeof p.message === "string") return p.message
  if (event.type === "gpu_sample") {
    const index = number(p, "gpu_index")
    const util = number(p, "util_percent")
    const used = number(p, "vram_used_mib")
    const total = number(p, "vram_total_mib")
    const temperature = number(p, "temperature_c")
    return `GPU ${index ?? "?"} · 利用率 ${util?.toFixed(0) ?? "--"}% · 显存 ${gib(used)} / ${gib(total)} GiB · ${temperature?.toFixed(0) ?? "--"}°C`
  }
  if (event.type === "step") {
    const total = number(p, "total_steps") ?? fallbackTotalSteps
    const loss = number(p, "loss")
    const lr = number(p, "lr")
    return `第 ${number(p, "step") ?? "?"} / ${total ?? "?"} 步${loss === null ? "" : ` · loss ${loss.toFixed(4)}`}${lr === null ? "" : ` · lr ${lr.toExponential(2)}`}`
  }
  if (event.type === "epoch_end") return `第 ${number(p, "epoch") ?? "?"} 回合结束`
  if (event.type === "checkpoint_saved") return `检查点已保存 · ${String(p.path ?? "")}`
  if (event.type === "sample_ready") return `预览图已生成 · ${String(p.path ?? "")}`
  if (event.type === "done") return `任务结束 · 返回码 ${String(p.returncode ?? "?")}`
  if (event.type === "start") return "训练已启动"
  if (event.type === "cancel") return "已请求取消"
  return JSON.stringify(p)
}

function toneFor(event: TrainingEvent, level: string, message: string): LogTone {
  if (event.type === "gpu_sample") return "gpu"
  if (event.type === "step") return "progress"
  if (event.type === "checkpoint_saved") return "checkpoint"
  if (event.type === "sample_ready") return "sample"
  if (event.type === "epoch_end") return "epoch"
  if (event.type === "done") return "done"
  if (level === "DEBUG") return "debug"
  if (level === "WARNING" || level === "WARN") return "warn"

  const canceled = /\b(?:keyboardinterrupt|killing subprocess|exits with return code = -(?:2|9|15))\b/i.test(message)
  if (
    !canceled &&
    (event.type === "error" || level === "ERROR" || level === "CRITICAL" ||
      /\b(error|fail(?:ed|ure)?|fatal|exception)\b/i.test(message))
  ) return "error"
  return "default"
}

function toLine(
  event: TrainingEvent,
  index: number,
  fallbackTotalSteps: number | null,
): LogLine {
  const p = event.payload
  const raw = eventMessage(event, fallbackTotalSteps)
  const plain = stripAnsi(raw)
  const legacy = event.type === "log" ? LEGACY_RICH_RE.exec(plain) : null
  const explicitLevel = typeof p.level === "string" ? p.level.toUpperCase() : null
  let level = legacy?.[1] ?? explicitLevel ?? event.type.toUpperCase()
  const message = legacy?.[2]?.trim() ?? raw
  const source = legacy?.[3] ?? (typeof p.location === "string" ? p.location : null)
  const stream = typeof p.source === "string" ? p.source : null
  const progress = PROGRESS_RE.exec(stripAnsi(message))
  const normalizedMessage = progress
    ? `${progress[1]} · ${progress[3]} / ${progress[4]} · ${progress[2]}% · ${progress[5]}`
    : message
  const tone = progress ? "progress" : toneFor(event, level, normalizedMessage)
  if (tone === "error" && level !== "CRITICAL") level = "ERROR"
  if (tone === "progress" && event.type === "log") level = "PROGRESS"
  if (tone === "gpu") level = "GPU"

  return {
    ts: event.timestamp,
    level,
    message: normalizedMessage,
    source,
    stream,
    tone,
    eventType: event.type,
    key: `${index}-${event.timestamp}-${event.type}`,
    progressKey: progress?.[1] ?? null,
    legacyRich: legacy !== null,
  }
}

function mergeWrappedRich(lines: LogLine[]): LogLine[] {
  const merged: LogLine[] = []
  for (const line of lines) {
    const previous = merged[merged.length - 1]
    const continuation =
      previous?.legacyRich === true &&
      line.eventType === "log" &&
      !line.legacyRich &&
      line.source === null &&
      line.stream === previous.stream &&
      line.ts - previous.ts <= 0.2 &&
      line.progressKey === null
    if (continuation) {
      previous.message = `${previous.message} ${line.message}`.replace(/\s+/g, " ").trim()
      previous.ts = line.ts
      continue
    }
    merged.push({ ...line })
  }
  return merged
}

function groupTracebacks(lines: LogLine[]): LogLine[] {
  const grouped: LogLine[] = []
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    if (!TRACEBACK_START_RE.test(stripAnsi(line.message))) {
      grouped.push(line)
      continue
    }
    const parts = [line.message]
    let cursor = index + 1
    for (; cursor < lines.length && cursor <= index + 80; cursor += 1) {
      const next = lines[cursor]
      if (next.eventType !== "log") break
      parts.push(next.message)
      if (TRACEBACK_END_RE.test(stripAnsi(next.message))) {
        cursor += 1
        break
      }
    }
    grouped.push({
      ...line,
      level: "ERROR",
      tone: "error",
      message: parts.join("\n"),
    })
    index = cursor - 1
  }
  return grouped
}

function collapseProgress(lines: LogLine[]): LogLine[] {
  const output: LogLine[] = []
  const positions = new Map<string, number>()
  for (const line of lines) {
    if (line.progressKey) {
      const position = positions.get(line.progressKey)
      if (position !== undefined) {
        output[position] = line
        continue
      }
      positions.set(line.progressKey, output.length)
    }
    output.push(line)
  }
  return output
}

export function buildLogLines(
  events: TrainingEvent[],
  fallbackTotalSteps: number | null,
): LogLine[] {
  const lines = events.map((event, index) => toLine(event, index, fallbackTotalSteps))
  return collapseProgress(groupTracebacks(mergeWrappedRich(lines)))
}

export function matchesLogFilter(line: LogLine, filter: LogFilter): boolean {
  if (filter === "all") return true
  if (filter === "training") return line.tone !== "gpu"
  if (filter === "warning") return line.tone === "warn"
  if (filter === "error") return line.tone === "error"
  return line.tone === "gpu"
}

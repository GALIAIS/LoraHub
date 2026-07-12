import type { TrainingEvent } from "@/lib/api"
import { renderInlineSummary } from "./event-timeline-model"

export type OperationStage =
  | "prepare"
  | "data"
  | "train"
  | "validate"
  | "artifact"
  | "diagnostic"
  | "complete"

export type OperationSeverity = "info" | "success" | "warning" | "critical"

export type OperationFilterId =
  | "all"
  | "critical"
  | "warning"
  | "artifact"
  | "validate"
  | "diagnostic"

export interface OperationEvent {
  id: string
  event: TrainingEvent
  index: number
  stage: OperationStage
  severity: OperationSeverity
  status: string
  title: string
  summary: string
  step: number | null
  epoch: number | null
  artifactPath: string | null
}

export interface OperationSummary {
  latest: OperationEvent | null
  latestStage: OperationStage | null
  latestSummary: string | undefined
  step: number | null
  totalSteps: number | null
  progressPercent: number | null
  loss: number | null
  critical: number
  warning: number
  checkpoints: number
  samples: number
  validations: number
  diagnostics: number
  eventCount: number
}

export const STAGE_LABEL: Record<OperationStage, string> = {
  prepare: "准备",
  data: "数据",
  train: "训练",
  validate: "验证",
  artifact: "产物",
  diagnostic: "诊断",
  complete: "完成",
}

export const SEVERITY_LABEL: Record<OperationSeverity, string> = {
  info: "信息",
  success: "正常",
  warning: "关注",
  critical: "严重",
}

export const FILTERS: Array<{ id: OperationFilterId; label: string }> = [
  { id: "all", label: "全部" },
  { id: "critical", label: "严重" },
  { id: "warning", label: "关注" },
  { id: "artifact", label: "产物" },
  { id: "validate", label: "验证" },
  { id: "diagnostic", label: "诊断" },
]

export function fmtClock(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString()
}

function numberPayload(
  payload: Record<string, unknown>,
  key: string,
): number | null {
  return typeof payload[key] === "number" ? (payload[key] as number) : null
}

function stringPayload(
  payload: Record<string, unknown>,
  key: string,
): string | null {
  return typeof payload[key] === "string" ? (payload[key] as string) : null
}

function cachePhaseLabel(phase: string): string {
  if (phase === "latents") return "潜空间缓存"
  if (phase === "text_encoder") return "文本编码缓存"
  return phase || "缓存"
}

export function classifyEvent(
  event: TrainingEvent,
  index: number,
  fallbackTotalSteps: number | null,
): OperationEvent | null {
  const p = event.payload ?? {}
  const step = numberPayload(p, "step")
  const epoch = numberPayload(p, "epoch")
  const artifactPath = stringPayload(p, "path") ?? stringPayload(p, "checkpoint")
  const logLevel = String(p.level ?? "").toLowerCase()
  const message = String(p.message ?? p.error ?? p.traceback ?? "")
  const isWarningLog = logLevel === "warning" || logLevel === "warn"
  const isErrorLog = ["error", "critical", "fatal"].includes(logLevel)

  switch (event.type) {
    case "step":
    case "gpu_sample":
    case "lora_spectrum":
    case "forgetting_probe":
      return null
    case "cache_progress": {
      const done = numberPayload(p, "done")
      const total = numberPayload(p, "total")
      const completed = total !== null && done !== null && done >= total
      return {
        id: `${index}-${event.timestamp}-cache`,
        event,
        index,
        stage: "data",
        severity: completed ? "success" : "info",
        status: completed ? "已完成" : "运行中",
        title: cachePhaseLabel(String(p.phase ?? "")),
        summary: `${done ?? "?"} / ${total ?? "?"}`,
        step,
        epoch,
        artifactPath: null,
      }
    }
    case "epoch_end":
      return {
        id: `${index}-${event.timestamp}-epoch`,
        event,
        index,
        stage: "train",
        severity: "success",
        status: "已完成",
        title: "回合结束",
        summary: renderInlineSummary(event, fallbackTotalSteps),
        step,
        epoch,
        artifactPath: null,
      }
    case "epoch_start":
      return {
        id: `${index}-${event.timestamp}-epoch-start`,
        event,
        index,
        stage: "train",
        severity: "info",
        status: "运行中",
        title: "回合开始",
        summary: renderInlineSummary(event, fallbackTotalSteps),
        step,
        epoch,
        artifactPath: null,
      }
    case "validation":
      return {
        id: `${index}-${event.timestamp}-validation`,
        event,
        index,
        stage: "validate",
        severity: "info",
        status: "已记录",
        title: "验证结果",
        summary: renderInlineSummary(event, fallbackTotalSteps),
        step,
        epoch,
        artifactPath: null,
      }
    case "checkpoint_saved":
      return {
        id: `${index}-${event.timestamp}-checkpoint`,
        event,
        index,
        stage: "artifact",
        severity: "success",
        status: "已保存",
        title: "检查点",
        summary: artifactPath ?? "已保存检查点",
        step,
        epoch,
        artifactPath,
      }
    case "sample_ready":
      return {
        id: `${index}-${event.timestamp}-sample`,
        event,
        index,
        stage: "artifact",
        severity: "success",
        status: "已生成",
        title: "采样图片",
        summary: artifactPath ?? "已生成样本",
        step,
        epoch,
        artifactPath,
      }
    case "preview_unavailable":
      return {
        id: `${index}-${event.timestamp}-preview`,
        event,
        index,
        stage: "diagnostic",
        severity: "warning",
        status: "已降级",
        title: "预览不可用",
        summary: renderInlineSummary(event, fallbackTotalSteps),
        step,
        epoch,
        artifactPath: null,
      }
    case "diagnostic_warning":
      return {
        id: `${index}-${event.timestamp}-diagnostic`,
        event,
        index,
        stage: "diagnostic",
        severity: p.severity === "error" ? "critical" : "warning",
        status: "需处理",
        title: String(p.message ?? p.category ?? "诊断告警"),
        summary: String(p.remediation ?? p.evidence ?? ""),
        step,
        epoch,
        artifactPath: null,
      }
    case "oom":
      return {
        id: `${index}-${event.timestamp}-oom`,
        event,
        index,
        stage: "diagnostic",
        severity: "critical",
        status: "已失败",
        title: "CUDA 显存溢出",
        summary: message || "CUDA out of memory",
        step,
        epoch,
        artifactPath: null,
      }
    case "error":
      return {
        id: `${index}-${event.timestamp}-error`,
        event,
        index,
        stage: "diagnostic",
        severity: "critical",
        status: "已失败",
        title: "训练错误",
        summary: message || "训练进程报告错误",
        step,
        epoch,
        artifactPath: null,
      }
    case "done": {
      const returncode = numberPayload(p, "returncode")
      return {
        id: `${index}-${event.timestamp}-done`,
        event,
        index,
        stage: "complete",
        severity: returncode === 0 ? "success" : "critical",
        status: returncode === 0 ? "成功" : "失败",
        title: "进程结束",
        summary: renderInlineSummary(event, fallbackTotalSteps),
        step,
        epoch,
        artifactPath: null,
      }
    }
    case "log":
      if (!isWarningLog && !isErrorLog) return null
      return {
        id: `${index}-${event.timestamp}-log`,
        event,
        index,
        stage: "diagnostic",
        severity: isErrorLog ? "critical" : "warning",
        status: isErrorLog ? "需处理" : "关注",
        title: isErrorLog ? "日志错误" : "日志警告",
        summary: message,
        step,
        epoch,
        artifactPath: null,
      }
    default:
      return null
  }
}

export function buildOperations(
  events: TrainingEvent[],
  fallbackTotalSteps: number | null,
): OperationEvent[] {
  const out: OperationEvent[] = []
  let first: OperationEvent | null = null
  for (let i = 0; i < events.length; i += 1) {
    const op = classifyEvent(events[i], i, fallbackTotalSteps)
    if (op) out.push(op)
    if (!first && events[i]) {
      first = {
        id: `spawn-${events[i].timestamp}`,
        event: events[i],
        index: i,
        stage: "prepare",
        severity: "info",
        status: "已启动",
        title: "训练进程",
        summary: "事件流已开始",
        step: null,
        epoch: null,
        artifactPath: null,
      }
    }
  }
  return first ? [first, ...out] : out
}

export function buildSummary(
  operations: OperationEvent[],
  events: TrainingEvent[],
  fallbackTotalSteps: number | null,
): OperationSummary {
  const latest = operations[operations.length - 1] ?? null
  const latestRaw = events[events.length - 1] ?? null
  const latestStep = [...events].reverse().find((event) => event.type === "step")
  const latestStepValue =
    latestStep && typeof latestStep.payload.step === "number"
      ? latestStep.payload.step
      : null
  const latestTotalSteps =
    latestStep && typeof latestStep.payload.total_steps === "number"
      ? (latestStep.payload.total_steps as number)
      : fallbackTotalSteps
  const latestLoss =
    latestStep && typeof latestStep.payload.loss === "number"
      ? (latestStep.payload.loss as number)
      : null
  const progressPercent =
    latestStepValue !== null &&
    typeof latestTotalSteps === "number" &&
    latestTotalSteps > 0
      ? Math.min(100, Math.max(0, (latestStepValue / latestTotalSteps) * 100))
      : null
  const latestStage =
    latestRaw?.type === "step"
      ? "train"
      : latest
        ? latest.stage
        : null
  const latestSummary =
    latestRaw?.type === "step"
      ? renderInlineSummary(latestRaw, fallbackTotalSteps)
      : latest?.summary
  return {
    latest,
    latestStage,
    latestSummary,
    step: latestStepValue,
    totalSteps: latestTotalSteps,
    progressPercent,
    loss: latestLoss,
    critical: operations.filter((op) => op.severity === "critical").length,
    warning: operations.filter((op) => op.severity === "warning").length,
    checkpoints: operations.filter((op) => op.event.type === "checkpoint_saved").length,
    samples: operations.filter((op) => op.event.type === "sample_ready").length,
    validations: operations.filter((op) => op.event.type === "validation").length,
    diagnostics: operations.filter((op) => op.stage === "diagnostic").length,
    eventCount: events.length,
  }
}

export function filterOperation(
  op: OperationEvent,
  filter: OperationFilterId,
): boolean {
  if (filter === "all") return true
  if (filter === "critical") return op.severity === "critical"
  if (filter === "warning") return op.severity === "warning"
  if (filter === "artifact") return op.stage === "artifact"
  if (filter === "validate") return op.stage === "validate"
  return op.stage === "diagnostic"
}

/**
 * LoRA 测试台纯函数 helper。
 *
 * session 结果提取、LoRA/轴参数构造、轴标签格式化等无副作用逻辑。
 * 放在独立文件便于单测与复用，避免混进组件。
 */
import type { Dispatch, SetStateAction } from "react"
import type { LoraTestAxisInput, LoraTestJob } from "@/lib/api"
import type { TaskSessionRecord } from "@/lib/api/tasks"
import type { LoraRow, ResultImage } from "./types"

export function getResultImages(session: TaskSessionRecord | undefined): ResultImage[] {
  const raw = session?.result?.images
  return Array.isArray(raw) ? (raw as ResultImage[]) : []
}

export function getResultGridPath(session: TaskSessionRecord | undefined): string | null {
  const raw = session?.result?.grid
  return typeof raw === "string" && raw ? raw : null
}

export function buildLoras(
  rows: LoraRow[],
  jobId: string,
  checkpointPath: string,
  weight: number,
) {
  if (rows.length === 0) return undefined
  return [
    { job_id: jobId, checkpoint_path: checkpointPath, weight },
    ...rows
      .filter((row) => row.jobId && row.checkpointPath && Number.isFinite(row.weight))
      .map((row) => ({
        job_id: row.jobId,
        checkpoint_path: row.checkpointPath,
        weight: row.weight,
      })),
  ]
}

export function buildAxis(
  field: LoraTestAxisInput["field"],
  raw: string,
): LoraTestAxisInput | null {
  const separator =
    field === "prompt" || field === "negative_prompt" || field === "checkpoint"
      ? /\n/
      : /[\n,]/
  const values = raw
    .split(separator)
    .map((item) => item.trim())
    .filter(Boolean)
  return values.length > 0 ? { field, values } : null
}

export function updateLoraRow(
  setRows: Dispatch<SetStateAction<LoraRow[]>>,
  id: string,
  patch: Partial<LoraRow>,
) {
  setRows((rows) =>
    rows.map((row) => (row.id === id ? { ...row, ...patch } : row)),
  )
}

export function buildPromptGeneralizationValues(prompt: string): string {
  const base = prompt.trim()
  if (!base) return ""
  return [
    base,
    `${base}, different outfit`,
    `${base}, outdoor scene`,
    `${base}, close-up portrait`,
  ].join("\n")
}

export function buildCheckpointAxisValues(
  selectedJob: LoraTestJob | null,
  current: string,
): string {
  const checkpoints = selectedJob?.checkpoints.map((item) => item.path) ?? []
  const unique = Array.from(new Set([current, ...checkpoints].filter(Boolean)))
  return unique.slice(0, 6).join("\n")
}

export function formatAxisLabel(image: ResultImage): string | null {
  return [image.x_label, image.y_label].filter(Boolean).join(" / ") || null
}

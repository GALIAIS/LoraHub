/**
 * 效果矩阵卡片 — 多 LoRA 叠加 + XY 轴编辑 + 预设。
 *
 * 从 index 抽出的大块内联 JSX。状态仍由 index 持有，通过 props 下发，
 * 保持单一数据源；本组件只负责渲染与回调。
 */
import type { Dispatch, SetStateAction } from "react"
import type { LoraTestAxisInput, LoraTestJob } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { AxisEditor, PresetButton } from "./axis-editor"
import {
  buildCheckpointAxisValues,
  buildPromptGeneralizationValues,
  updateLoraRow,
} from "./helpers"
import { NEGATIVE_STRESS_VALUES, QUALITY_NEGATIVE } from "./types"
import type { LoraRow } from "./types"

export function MatrixCard({
  loraRows,
  setLoraRows,
  jobId,
  checkpointPath,
  jobs,
  xField,
  setXField,
  xValues,
  setXValues,
  yField,
  setYField,
  yValues,
  setYValues,
  seed,
  prompt,
  selectedJob,
  setNegative,
}: {
  loraRows: LoraRow[]
  setLoraRows: Dispatch<SetStateAction<LoraRow[]>>
  jobId: string
  checkpointPath: string
  jobs: LoraTestJob[]
  xField: LoraTestAxisInput["field"]
  setXField: (field: LoraTestAxisInput["field"]) => void
  xValues: string
  setXValues: (values: string) => void
  yField: LoraTestAxisInput["field"]
  setYField: (field: LoraTestAxisInput["field"]) => void
  yValues: string
  setYValues: (values: string) => void
  seed: number
  prompt: string
  selectedJob: LoraTestJob | null
  setNegative: Dispatch<SetStateAction<string>>
}) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>效果矩阵</CardTitle>
        <CardDescription>
          多 LoRA 叠加与 XY 轴扫描，用同一 seed 对比权重、CFG、steps、sampler 或 checkpoint。
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs font-medium">叠加 LoRA</div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() =>
                setLoraRows((rows) => [
                  ...rows,
                  {
                    id: crypto.randomUUID(),
                    jobId,
                    checkpointPath,
                    weight: 1,
                  },
                ])
              }
              disabled={!jobId || !checkpointPath}
            >
              添加
            </Button>
          </div>
          {loraRows.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              默认只用上方选择的主 LoRA。添加后会按顺序叠加多个 LoRA。
            </p>
          ) : (
            <div className="flex flex-col gap-2">
              {loraRows.map((row, index) => {
                const rowJob = jobs.find((item) => item.job_id === row.jobId)
                return (
                  <div
                    key={row.id}
                    className="grid gap-2 rounded-[6px] border border-border/60 bg-muted/20 p-2 md:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_6rem_auto]"
                  >
                    <Select
                      value={row.jobId}
                      onValueChange={(value) => {
                        if (!value) return
                        const nextJob = jobs.find((item) => item.job_id === value)
                        updateLoraRow(setLoraRows, row.id, {
                          jobId: value,
                          checkpointPath: nextJob?.checkpoints[0]?.path ?? "",
                        })
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder={`LoRA ${index + 1}`} />
                      </SelectTrigger>
                      <SelectContent>
                        {jobs.map((item) => (
                          <SelectItem key={item.job_id} value={item.job_id}>
                            {item.output_name ?? item.job_id.slice(-8)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      value={row.checkpointPath}
                      onValueChange={(value) =>
                        value && updateLoraRow(setLoraRows, row.id, { checkpointPath: value })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="checkpoint" />
                      </SelectTrigger>
                      <SelectContent>
                        {(rowJob?.checkpoints ?? []).map((ckpt) => (
                          <SelectItem key={ckpt.path} value={ckpt.path}>
                            {ckpt.path}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Input
                      type="number"
                      step={0.05}
                      min={-2}
                      max={2}
                      value={row.weight}
                      onChange={(e) =>
                        updateLoraRow(setLoraRows, row.id, {
                          weight: Number(e.target.value),
                        })
                      }
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        setLoraRows((rows) => rows.filter((item) => item.id !== row.id))
                      }
                    >
                      删除
                    </Button>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <AxisEditor
            title="X 轴"
            field={xField}
            values={xValues}
            onFieldChange={setXField}
            onValuesChange={setXValues}
          />
          <AxisEditor
            title="Y 轴"
            field={yField}
            values={yValues}
            onFieldChange={setYField}
            onValuesChange={setYValues}
          />
        </div>
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          <PresetButton
            label="Base 对照"
            onClick={() => {
              setXField("variant")
              setXValues("base, lora")
              setYValues("")
            }}
          />
          <PresetButton
            label="权重扫描"
            onClick={() => {
              setXField("lora_weight")
              setXValues("0.4, 0.6, 0.8, 1.0, 1.2")
              setYValues("")
            }}
          />
          <PresetButton
            label="Prompt 泛化"
            onClick={() => {
              setXField("prompt")
              setXValues(buildPromptGeneralizationValues(prompt))
              setYValues("")
            }}
          />
          <PresetButton
            label="Seed 稳定性"
            onClick={() => {
              const base = seed >= 0 ? seed : 1001
              setXField("seed")
              setXValues(`${base}, ${base + 1}, ${base + 2}, ${base + 3}`)
              setYValues("")
            }}
          />
          <PresetButton
            label="CFG x 权重"
            onClick={() => {
              setXField("lora_weight")
              setXValues("0.6, 0.8, 1.0, 1.2")
              setYField("cfg")
              setYValues("3.5, 4.5, 5.5")
            }}
          />
          <PresetButton
            label="Steps x Sampler"
            onClick={() => {
              setXField("steps")
              setXValues("16, 24, 32, 40")
              setYField("sampler")
              setYValues("euler, er_sde")
            }}
          />
          <PresetButton
            label="负面词压力"
            onClick={() => {
              setXField("negative_prompt")
              setXValues(NEGATIVE_STRESS_VALUES.join("\n"))
              setYValues("")
            }}
          />
          <PresetButton
            label="尺寸鲁棒性"
            onClick={() => {
              setXField("size")
              setXValues("768x1344, 896x1632, 1024x1024")
              setYValues("")
            }}
          />
          <PresetButton
            label="Checkpoint 回放"
            onClick={() => {
              setXField("checkpoint")
              setXValues(buildCheckpointAxisValues(selectedJob, checkpointPath))
              setYValues("")
            }}
          />
          <PresetButton
            label="质量诊断矩阵"
            onClick={() => {
              setNegative((current) => current || QUALITY_NEGATIVE)
              setXField("lora_weight")
              setXValues("0.6, 0.8, 1.0, 1.2")
              setYField("steps")
              setYValues("20, 28, 36")
            }}
          />
        </div>
      </CardContent>
    </Card>
  )
}

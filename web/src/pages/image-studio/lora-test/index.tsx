import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Copy,
  Download,
  ImageIcon,
  Loader2,
  Play,
  RefreshCw,
  Wand2,
} from "lucide-react"
import { toast } from "sonner"
import { useSearchParams } from "react-router-dom"
import { api, type LoraTestAxisInput, type LoraTestJob } from "@/lib/api"
import type { TaskSessionRecord } from "@/lib/api/tasks"
import { Badge } from "@/components/ui/badge"
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
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { QueuePanel } from "./queue-panel"

const SIZE_PRESETS = [
  { label: "896 x 1632", width: 896, height: 1632 },
  { label: "768 x 1344", width: 768, height: 1344 },
  { label: "832 x 1216", width: 832, height: 1216 },
  { label: "1024 x 1024", width: 1024, height: 1024 },
] as const

interface ResultImage {
  path: string
  seed: number
  prompt: string
  negative_prompt: string
  width: number
  height: number
  steps: number
  cfg: number
  sampler: string
  lora_weight: number
  loras?: Array<{
    job_id: string
    checkpoint_path: string
    checkpoint_name: string
    weight: number
  }>
  checkpoint_path: string
  job_id: string
  x_label?: string | null
  y_label?: string | null
}

interface LoraRow {
  id: string
  jobId: string
  checkpointPath: string
  weight: number
}

const AXIS_FIELDS = [
  { value: "variant", label: "Base/LoRA" },
  { value: "prompt", label: "Prompt" },
  { value: "negative_prompt", label: "Negative" },
  { value: "seed", label: "Seed" },
  { value: "lora_weight", label: "LoRA 权重" },
  { value: "cfg", label: "CFG" },
  { value: "steps", label: "Steps" },
  { value: "sampler", label: "Sampler" },
  { value: "size", label: "尺寸" },
  { value: "checkpoint", label: "Checkpoint" },
] as const

const NEGATIVE_STRESS_VALUES = [
  "empty",
  "low quality, worst quality, blurry, bad anatomy",
  "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, watermark, text",
]

const QUALITY_NEGATIVE =
  "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, watermark, text"

export function LoraTestPage() {
  const [params, setParams] = useSearchParams()
  const qc = useQueryClient()
  const models = useQuery({
    queryKey: ["lora-test-models"],
    queryFn: api.listLoraTestModels,
    staleTime: 5_000,
  })
  const jobs = models.data?.jobs ?? []
  const urlJob = params.get("job") ?? ""
  const urlCheckpoint = params.get("checkpoint") ?? ""
  const urlSession = params.get("session") ?? ""
  const [jobId, setJobId] = useState(urlJob)
  const [checkpointPath, setCheckpointPath] = useState(urlCheckpoint)
  const [prompt, setPrompt] = useState("")
  const [negative, setNegative] = useState("")
  const [width, setWidth] = useState(896)
  const [height, setHeight] = useState(1632)
  const [seed, setSeed] = useState(-1)
  const [batchCount, setBatchCount] = useState(4)
  const [steps, setSteps] = useState(28)
  const [cfg, setCfg] = useState(4.5)
  const [sampler, setSampler] = useState("euler")
  const [loraWeight, setLoraWeight] = useState(1)
  const [loraRows, setLoraRows] = useState<LoraRow[]>([])
  const [xField, setXField] = useState<LoraTestAxisInput["field"]>("lora_weight")
  const [xValues, setXValues] = useState("")
  const [yField, setYField] = useState<LoraTestAxisInput["field"]>("cfg")
  const [yValues, setYValues] = useState("")
  const [sessionId, setSessionId] = useState(urlSession)
  const [detail, setDetail] = useState<ResultImage | null>(null)

  const selectedJob = useMemo(
    () => jobs.find((job) => job.job_id === jobId) ?? null,
    [jobs, jobId],
  )
  const session = useQuery({
    queryKey: ["lora-test-session", sessionId],
    queryFn: () => api.getLoraTestSession(sessionId),
    enabled: Boolean(sessionId),
    refetchInterval: (q) => {
      const status = q.state.data?.status
      return status === "queued" ||
        status === "running" ||
        status === "stop_requested"
        ? 1500
        : false
    },
  })
  const resultImages = getResultImages(session.data)
  const canGenerate =
    Boolean(jobId) && Boolean(checkpointPath) && prompt.trim().length > 0

  useEffect(() => {
    if (!jobId && jobs.length > 0) {
      const first = jobs[0]
      setJobId(first.job_id)
      setCheckpointPath(first.checkpoints[0]?.path ?? "")
    }
  }, [jobId, jobs])

  useEffect(() => {
    setJobId(urlJob)
    setCheckpointPath(urlCheckpoint)
    setSessionId(urlSession)
  }, [urlJob, urlCheckpoint, urlSession])

  const start = useMutation({
    mutationFn: () =>
      api.startLoraTestGeneration({
        job_id: jobId,
        checkpoint_path: checkpointPath,
        prompt: prompt.trim(),
        negative_prompt: negative.trim(),
        width,
        height,
        seed,
        batch_count: batchCount,
        steps,
        cfg,
        sampler,
        lora_weight: loraWeight,
        loras: buildLoras(loraRows, jobId, checkpointPath, loraWeight),
        x_axis: buildAxis(xField, xValues),
        y_axis: buildAxis(yField, yValues),
        output_format: "png",
      }),
    onSuccess: (data) => {
      setSessionId(data.session_id)
      const next = new URLSearchParams(params)
      next.set("stage", "lora-test")
      next.set("job", jobId)
      next.set("checkpoint", checkpointPath)
      next.set("session", data.session_id)
      setParams(next, { replace: true })
      qc.invalidateQueries({ queryKey: ["lora-test-session", data.session_id] })
      toast.success("LoRA 测试任务已开始")
    },
    onError: (e) =>
      toast.error("启动失败", {
        description: e instanceof Error ? e.message : String(e),
      }),
  })

  const cancel = useMutation({
    mutationFn: () => api.cancelLoraTestSession(sessionId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lora-test-session", sessionId] })
      toast.info("已请求停止")
    },
    onError: (error) =>
      toast.error("停止失败", {
        description: error instanceof Error ? error.message : String(error),
      }),
  })

  function updateJob(nextJobId: string) {
    const nextJob = jobs.find((job) => job.job_id === nextJobId)
    setJobId(nextJobId)
    setCheckpointPath(nextJob?.checkpoints[0]?.path ?? "")
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <header className="border-b border-border/60 px-4 py-3 md:px-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Wand2 className="size-4 text-muted-foreground" />
              <h1 className="text-sm font-semibold">LoRA 生图测试</h1>
              <Badge variant="outline">anima_lora</Badge>
            </div>
            <p className="mt-1 truncate text-xs text-muted-foreground">
              选择训练产物，快速验证 LoRA 权重、prompt 和采样参数。
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => models.refetch()}
              disabled={models.isFetching}
            >
              {models.isFetching ? <Loader2 className="animate-spin" /> : <RefreshCw />}
              刷新产物
            </Button>
            <Button
              size="sm"
              disabled={!canGenerate || start.isPending}
              onClick={() => start.mutate()}
            >
              {start.isPending ? <Loader2 className="animate-spin" /> : <Play />}
              生成
            </Button>
          </div>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[18rem_minmax(0,1fr)_18rem]">
        <aside className="min-h-0 overflow-y-auto border-b border-border/60 p-3 lg:border-b-0 lg:border-r">
          <ModelPanel
            loading={models.isLoading}
            jobs={jobs}
            jobId={jobId}
            checkpointPath={checkpointPath}
            onJobChange={updateJob}
            onCheckpointChange={setCheckpointPath}
            selectedJob={selectedJob}
          />
        </aside>

        <main className="min-h-0 overflow-y-auto p-3">
          <div className="flex flex-col gap-3">
            <Card size="sm">
              <CardHeader>
                <CardTitle>Prompt</CardTitle>
                <CardDescription>
                  常用参数直接展开，高级项先收敛到必要字段。
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <label className="flex flex-col gap-1.5 text-xs font-medium">
                  正向 prompt
                  <Textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="masterpiece, best quality, ..."
                    className="min-h-28"
                  />
                </label>
                <label className="flex flex-col gap-1.5 text-xs font-medium">
                  负向 prompt
                  <Textarea
                    value={negative}
                    onChange={(e) => setNegative(e.target.value)}
                    placeholder="low quality, blurry, ..."
                    className="min-h-20"
                  />
                </label>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                  <Field label="尺寸预设">
                    <Select
                      value={`${width}x${height}`}
                      onValueChange={(value) => {
                        if (!value) return
                        const preset = SIZE_PRESETS.find(
                          (item) => `${item.width}x${item.height}` === value,
                        )
                        if (preset) {
                          setWidth(preset.width)
                          setHeight(preset.height)
                        }
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="选择尺寸" />
                      </SelectTrigger>
                      <SelectContent>
                        {SIZE_PRESETS.map((item) => (
                          <SelectItem
                            key={`${item.width}x${item.height}`}
                            value={`${item.width}x${item.height}`}
                          >
                            {item.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                  <NumberField label="宽" value={width} min={256} max={2048} onChange={setWidth} />
                  <NumberField label="高" value={height} min={256} max={2048} onChange={setHeight} />
                  <NumberField label="Seed (-1 随机)" value={seed} min={-1} max={2147483647} onChange={setSeed} />
                  <NumberField label="张数" value={batchCount} min={1} max={32} onChange={setBatchCount} />
                  <NumberField label="Steps" value={steps} min={1} max={150} onChange={setSteps} />
                  <NumberField label="CFG" value={cfg} min={0} max={30} step={0.1} onChange={setCfg} />
                  <NumberField label="LoRA 权重" value={loraWeight} min={-2} max={2} step={0.05} onChange={setLoraWeight} />
                  <Field label="Sampler">
                    <Select value={sampler} onValueChange={(value) => value && setSampler(value)}>
                      <SelectTrigger>
                        <SelectValue placeholder="sampler" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="euler">euler</SelectItem>
                        <SelectItem value="er_sde">er_sde</SelectItem>
                        <SelectItem value="lcm">lcm</SelectItem>
                      </SelectContent>
                    </Select>
                  </Field>
                </div>
              </CardContent>
            </Card>

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

            <ResultGrid
              sessionId={sessionId}
              images={resultImages}
              gridPath={getResultGridPath(session.data)}
              loading={session.isFetching && !session.data}
              onOpen={setDetail}
            />
          </div>
        </main>

        <aside className="min-h-0 overflow-y-auto border-t border-border/60 p-3 lg:border-l lg:border-t-0">
          <QueuePanel
            session={session.data ?? null}
            loading={session.isFetching}
            onCancel={() => cancel.mutate()}
            canceling={cancel.isPending}
          />
        </aside>
      </div>

      <ResultSheet
        sessionId={sessionId}
        image={detail}
        onClose={() => setDetail(null)}
      />
    </div>
  )
}

function ModelPanel({
  loading,
  jobs,
  jobId,
  checkpointPath,
  onJobChange,
  onCheckpointChange,
  selectedJob,
}: {
  loading: boolean
  jobs: LoraTestJob[]
  jobId: string
  checkpointPath: string
  onJobChange: (value: string) => void
  onCheckpointChange: (value: string) => void
  selectedJob: LoraTestJob | null
}) {
  if (loading) return <Skeleton className="h-48" />
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>模型</CardTitle>
        <CardDescription>从训练产物中选择 LoRA checkpoint。</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Field label="训练任务">
          <Select value={jobId} onValueChange={(value) => value && onJobChange(value)}>
            <SelectTrigger>
              <SelectValue placeholder="选择任务" />
            </SelectTrigger>
            <SelectContent>
              {jobs.map((job) => (
                <SelectItem key={job.job_id} value={job.job_id}>
                  {job.output_name ?? job.job_id.slice(-8)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="Checkpoint">
          <Select
            value={checkpointPath}
            onValueChange={(value) => value && onCheckpointChange(value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="选择 LoRA" />
            </SelectTrigger>
            <SelectContent>
              {(selectedJob?.checkpoints ?? []).map((ckpt) => (
                <SelectItem key={ckpt.path} value={ckpt.path}>
                  {ckpt.path}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        {selectedJob ? (
          <div className="rounded-[6px] border border-border/60 bg-muted/30 p-2 text-[11px] text-muted-foreground">
            <div className="font-mono text-foreground">{selectedJob.backend ?? "unknown"}</div>
            <div className="mt-1 break-all">{selectedJob.workspace}</div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">没有可测试的 LoRA 产物。</p>
        )}
      </CardContent>
    </Card>
  )
}

function ResultGrid({
  sessionId,
  images,
  gridPath,
  loading,
  onOpen,
}: {
  sessionId: string
  images: ResultImage[]
  gridPath: string | null
  loading: boolean
  onOpen: (image: ResultImage) => void
}) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>结果</CardTitle>
        <CardDescription>固定比例网格展示，参数写入 sidecar 可追溯。</CardDescription>
      </CardHeader>
      <CardContent>
        {loading && <Skeleton className="h-64" />}
        {!loading && images.length === 0 && (
          <div className="grid min-h-64 place-items-center rounded-[6px] border border-dashed border-border/70 bg-muted/25 text-center">
            <div>
              <ImageIcon className="mx-auto size-8 text-muted-foreground" />
              <p className="mt-2 text-sm font-medium">还没有生成结果</p>
              <p className="mt-1 text-xs text-muted-foreground">
                生成完成后会显示图片、seed 和核心参数。
              </p>
            </div>
          </div>
        )}
        {gridPath && (
          <button
            type="button"
            className="mb-3 block w-full overflow-hidden rounded-[6px] border border-border/60 bg-background text-left"
            onClick={() =>
              window.open(
                api.loraTestResultFileUrl(sessionId, gridPath),
                "_blank",
                "noopener,noreferrer",
              )
            }
          >
            <img
              src={api.loraTestResultFileUrl(sessionId, gridPath)}
              alt="XY grid"
              className="max-h-[70vh] w-full object-contain"
            />
          </button>
        )}
        {images.length > 0 && (
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {images.map((image) => (
              <button
                key={image.path}
                type="button"
                className="group overflow-hidden rounded-[6px] border border-border/60 bg-muted/20 text-left"
                onClick={() => onOpen(image)}
              >
                <div className="grid aspect-[9/13] place-items-center bg-background">
                  <img
                    src={api.loraTestResultFileUrl(sessionId, image.path)}
                    alt={`seed ${image.seed}`}
                    className="max-h-full max-w-full object-contain"
                  />
                </div>
                <div className="flex items-center justify-between gap-2 px-2 py-2 text-[11px]">
                  <span className="font-mono">seed {image.seed}</span>
                  <span className="text-muted-foreground">
                    {formatAxisLabel(image) ?? `${image.steps} / cfg ${image.cfg}`}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function ResultSheet({
  sessionId,
  image,
  onClose,
}: {
  sessionId: string
  image: ResultImage | null
  onClose: () => void
}) {
  return (
    <Sheet open={image !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full sm:max-w-3xl" side="right">
        <SheetHeader>
          <SheetTitle>生成结果</SheetTitle>
          <SheetDescription>
            图片和完整参数，可复制到下一轮测试。
          </SheetDescription>
        </SheetHeader>
        {image && (
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_18rem]">
              <div className="grid min-h-[24rem] place-items-center rounded-[6px] border bg-background">
                <img
                  src={api.loraTestResultFileUrl(sessionId, image.path)}
                  alt={`seed ${image.seed}`}
                  className="max-h-[70vh] max-w-full object-contain"
                />
              </div>
              <div className="flex flex-col gap-3 text-xs">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    navigator.clipboard.writeText(JSON.stringify(image, null, 2))
                  }
                >
                  <Copy />
                  复制参数
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    window.open(
                      api.loraTestResultFileUrl(sessionId, image.path),
                      "_blank",
                      "noopener,noreferrer",
                    )
                  }
                >
                  <Download />
                  打开图片
                </Button>
                <Param label="seed" value={String(image.seed)} />
                <Param label="尺寸" value={`${image.width} x ${image.height}`} />
                <Param label="steps / cfg" value={`${image.steps} / ${image.cfg}`} />
                <Param label="sampler" value={image.sampler} />
                <Param
                  label="LoRA"
                  value={
                    image.loras?.length
                      ? image.loras
                          .map((item) => `${item.checkpoint_name} x ${item.weight}`)
                          .join("\n")
                      : String(image.lora_weight)
                  }
                  block
                />
                <Param label="checkpoint" value={image.checkpoint_path} />
                <Param label="prompt" value={image.prompt} block />
                <Param label="negative" value={image.negative_prompt || "-"} block />
              </div>
            </div>
          </div>
        )}
      </SheetContent>
    </Sheet>
  )
}

function Field({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <label className="flex flex-col gap-1.5 text-xs font-medium">
      {label}
      {children}
    </label>
  )
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
}) {
  return (
    <Field label={label}>
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const next = Number(e.target.value)
          if (Number.isFinite(next)) onChange(next)
        }}
      />
    </Field>
  )
}

function Param({
  label,
  value,
  block = false,
}: {
  label: string
  value: string
  block?: boolean
}) {
  return (
    <div className="rounded-[6px] border border-border/60 bg-muted/25 p-2">
      <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </div>
      <div className={block ? "mt-1 whitespace-pre-wrap" : "mt-1 break-all font-mono"}>
        {value}
      </div>
    </div>
  )
}

function getResultImages(session: TaskSessionRecord | undefined): ResultImage[] {
  const raw = session?.result?.images
  return Array.isArray(raw) ? (raw as ResultImage[]) : []
}

function getResultGridPath(session: TaskSessionRecord | undefined): string | null {
  const raw = session?.result?.grid
  return typeof raw === "string" && raw ? raw : null
}

function buildLoras(
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

function buildAxis(
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

function updateLoraRow(
  setRows: React.Dispatch<React.SetStateAction<LoraRow[]>>,
  id: string,
  patch: Partial<LoraRow>,
) {
  setRows((rows) =>
    rows.map((row) => (row.id === id ? { ...row, ...patch } : row)),
  )
}

function AxisEditor({
  title,
  field,
  values,
  onFieldChange,
  onValuesChange,
}: {
  title: string
  field: LoraTestAxisInput["field"]
  values: string
  onFieldChange: (field: LoraTestAxisInput["field"]) => void
  onValuesChange: (values: string) => void
}) {
  return (
    <div className="rounded-[6px] border border-border/60 bg-muted/20 p-3">
      <div className="mb-2 text-xs font-medium">{title}</div>
      <div className="grid gap-2">
        <Select
          value={field}
          onValueChange={(value) => onFieldChange(value as LoraTestAxisInput["field"])}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {AXIS_FIELDS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {field === "prompt" || field === "negative_prompt" || field === "checkpoint" ? (
          <Textarea
            value={values}
            onChange={(e) => onValuesChange(e.target.value)}
            placeholder={axisPlaceholder(field)}
            className="min-h-24 font-mono text-xs"
          />
        ) : (
          <Input
            value={values}
            onChange={(e) => onValuesChange(e.target.value)}
            placeholder={axisPlaceholder(field)}
            className="font-mono"
          />
        )}
      </div>
    </div>
  )
}

function PresetButton({
  label,
  onClick,
}: {
  label: string
  onClick: () => void
}) {
  return (
    <Button type="button" variant="outline" size="sm" onClick={onClick}>
      {label}
    </Button>
  )
}

function buildPromptGeneralizationValues(prompt: string): string {
  const base = prompt.trim()
  if (!base) return ""
  return [
    base,
    `${base}, different outfit`,
    `${base}, outdoor scene`,
    `${base}, close-up portrait`,
  ].join("\n")
}

function buildCheckpointAxisValues(
  selectedJob: LoraTestJob | null,
  current: string,
): string {
  const checkpoints = selectedJob?.checkpoints.map((item) => item.path) ?? []
  const unique = Array.from(new Set([current, ...checkpoints].filter(Boolean)))
  return unique.slice(0, 6).join("\n")
}

function axisPlaceholder(field: LoraTestAxisInput["field"]): string {
  if (field === "checkpoint") return "每行一个 checkpoint 相对路径"
  if (field === "variant") return "base, lora"
  if (field === "prompt") return "每行一个 prompt"
  if (field === "negative_prompt") return "每行一个 negative；empty 表示空负面词"
  if (field === "size") return "768x1344, 896x1632, 1024x1024"
  return "0.6, 0.8, 1.0, 1.2"
}

function formatAxisLabel(image: ResultImage): string | null {
  return [image.x_label, image.y_label].filter(Boolean).join(" / ") || null
}

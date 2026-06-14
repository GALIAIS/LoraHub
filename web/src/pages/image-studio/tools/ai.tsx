/**
 * AI 类工具页面集合。
 *
 *   - ai-smart-caption     : 复用现成的 batch + 全局任务条 (smart-caption 走 sessioned poll)
 *   - ai-caption           : 直接 VLM 写 caption（无 WD14 预处理）
 *   - ai-quality           : 给图打质量分（sessioned poll）
 *   - ai-trigger-words     : 让 VLM 推荐触发词候选（in-flight task）
 *   - ai-wd14-prefilter    : 单图测试 - 拿到 WD14 标签 + 拼装好的 prompt
 *   - ai-vlm-anima-rewrite : 单图测试 - 把上一个的产出喂给 VLM 写 caption
 *
 * 批量类工具入队到全局 studio-task-store, banner 在 tool-page 顶部由
 * 调用方渲染（这里通过 GlobalTaskBanner 自渲染，跨 tool 切换都可见）。
 */
import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  BookOpen,
  Gauge,
  ImageIcon,
  Loader2,
  Play,
  Sparkles,
  Tags,
  Wand2,
  X,
} from "lucide-react"
import { toast } from "sonner"
import {
  api,
  imageStudioBatchCaption,
  imageStudioBatchTriggerWords,
  imageStudioVlmAnimaRewrite,
  imageStudioWd14Prefilter,
  startQualitySession,
  startSmartCaptionSession,
  startTaggingSession,
  type ImageStudioTriggerWordsResult,
  type Wd14PrefilterResult,
} from "@/lib/api"
import {
  addTask,
  removeTask,
  updateTask,
} from "@/lib/studio-task-store"
import { useStudioTasksFor } from "@/hooks/use-studio-tasks"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { TriggerPicker } from "../components/library/trigger-picker"

// ai-bulk-modal.tsx 里同款 fallback。WD14 模型清单从 /tagging/wd14/models
// 来；首次还没拿到时先用一个真实存在的默认值，避免 401。
const FALLBACK_DEFAULT_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"

const newTaskId = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `studio-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

// --------------------------------------------------------------------------- //
// GlobalTaskBanner — 与 dataset-detail 里的 StudioTaskBanner 等价的极简版
// --------------------------------------------------------------------------- //

function TaskBanner({ datasetPath }: { datasetPath: string }) {
  const tasks = useStudioTasksFor(datasetPath)
  if (tasks.length === 0) return null
  const newest = [...tasks].sort((a, b) => b.startedAt - a.startedAt)[0]
  if (!newest) return null
  const running = newest.status === "running"
  const lastImageName = newest.lastImage
    ? newest.lastImage.split(/[/\\]/).pop() ?? ""
    : ""
  const label = running
    ? newest.kind === "smart-caption"
      ? `${newest.label}中…${lastImageName ? ` · ${lastImageName}` : ""}`
      : newest.kind === "quality-score"
        ? `${newest.label}中…${lastImageName ? ` · ${lastImageName}` : ""}`
      : newest.kind === "wd14"
        ? `${newest.label}中… ${newest.processed ?? 0}/${newest.total ?? "?"}`
        : `${newest.label}中…`
    : newest.label
  return (
    <div className="flex items-center gap-3 rounded-md border bg-muted/30 px-3 py-2 mb-3">
      {running && <Loader2 className="size-4 animate-spin text-primary" />}
      <span className="text-xs font-medium">{label}</span>
      {newest.processed != null && newest.kind !== "wd14" && (
        <span className="text-xs text-muted-foreground">
          {newest.processed}
          {newest.total ? ` / ${newest.total}` : ""} 张
        </span>
      )}
      {newest.errorMsg && (
        <span className="text-xs text-destructive truncate flex-1">
          {newest.errorMsg}
        </span>
      )}
      {!running && (
        <button
          type="button"
          onClick={() => removeTask(newest.id)}
          className="ml-auto text-xs text-muted-foreground hover:text-foreground"
          title="关闭"
        >
          <X className="size-3.5" />
        </button>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------- //
// ai-smart-caption — WD14 + VLM 两步式
// --------------------------------------------------------------------------- //

export function AiSmartCaptionTool({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [mergeStrategy, setMergeStrategy] = useState("replace")
  const [device, setDevice] = useState("auto")
  const [captionMode, setCaptionMode] =
    useState<"general" | "style" | "character">("style")
  const [captionSource, setCaptionSource] = useState<"vlm" | "tags">("vlm")
  const [triggerWord, setTriggerWord] = useState("")
  const [triggerPickerOpen, setTriggerPickerOpen] = useState(false)
  const [stripStyleTags, setStripStyleTags] = useState(true)
  const [skipExisting, setSkipExisting] = useState(true)
  const [recursive, setRecursive] = useState(true)

  const start = async () => {
    try {
      const submit = await startSmartCaptionSession({
        path: datasetPath,
        recursive,
        device,
        mergeStrategy,
        captionMode,
        captionSource,
        triggerWord: triggerWord.trim() || undefined,
        stripStyleTags,
        skipExisting,
      })
      addTask({
        id: submit.session_id,
        kind: "smart-caption",
        datasetPath,
        label:
          captionSource === "tags"
            ? "智能标注（WD14 + LLM 文本）"
            : "智能标注（WD14 + VLM 视觉）",
        total: submit.total,
      })
      toast.success("已启动智能标注", {
        description: `共 ${submit.total} 张 · 后台进行 · 切到其他 tool 也不会中断`,
      })
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    } catch (err) {
      toast.error("启动失败", {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  return (
    <div className="h-full overflow-y-auto p-4 max-w-2xl space-y-3">
      <TaskBanner datasetPath={datasetPath} />
      <section className="rounded-md border border-border/60 bg-card flex flex-col">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Wand2 className="size-3.5" />
          <span className="text-xs font-medium">智能 caption（WD14 + VLM/LLM）</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            两步流水：先 WD14 拿到 booru 标签，再喂给 VLM/LLM 合成 Anima 风 caption。
            想拆开测试的话用「WD14 单步出标签」+「VLM Anima 重写」单图工具。
          </p>
          <Row label="LLM 输入">
            <Select
              value={captionSource}
              onValueChange={(v) => v && setCaptionSource(v as typeof captionSource)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="vlm">视觉模型（看图） · 质量最高</SelectItem>
                <SelectItem value="tags">仅 WD14 标签 · 不上传图片</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Row label="训练用途">
            <Select
              value={captionMode}
              onValueChange={(v) => v && setCaptionMode(v as typeof captionMode)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="style">风格 LoRA</SelectItem>
                <SelectItem value="character">角色 LoRA</SelectItem>
                <SelectItem value="general">通用</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Row label="触发词">
            <div className="flex items-center gap-1">
              <Input
                value={triggerWord}
                onChange={(e) => setTriggerWord(e.target.value)}
                placeholder="anima style / @charA"
                className="h-8 text-xs"
              />
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-8 px-2 gap-1 text-[11px]"
                onClick={() => setTriggerPickerOpen((v) => !v)}
                title="从工具库选触发词"
              >
                <BookOpen className="size-3" />
                工具库
              </Button>
              {triggerPickerOpen && (
                <TriggerPicker
                  onSelect={(t) => {
                    setTriggerWord(t)
                    setTriggerPickerOpen(false)
                  }}
                  onClose={() => setTriggerPickerOpen(false)}
                />
              )}
            </div>
          </Row>
          <Row label="合并策略">
            <Select
              value={mergeStrategy}
              onValueChange={(v) => v && setMergeStrategy(v)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="replace">替换（推荐）</SelectItem>
                <SelectItem value="append">追加</SelectItem>
                <SelectItem value="prepend">前置</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Row label="设备">
            <Select value={device} onValueChange={(v) => v && setDevice(v)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">自动</SelectItem>
                <SelectItem value="cuda">CUDA</SelectItem>
                <SelectItem value="cpu">CPU</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <div className="flex items-center gap-3 flex-wrap pt-1">
            {captionMode === "style" && (
              <label className="inline-flex items-center gap-1.5 select-none">
                <Switch checked={stripStyleTags} onCheckedChange={setStripStyleTags} />
                剔除画风标签
              </label>
            )}
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={skipExisting} onCheckedChange={setSkipExisting} />
              跳过已有 caption
            </label>
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={recursive} onCheckedChange={setRecursive} />
              递归子目录
            </label>
          </div>
          <Button size="sm" onClick={start} className="w-full gap-1">
            <Play className="size-3" />
            启动智能标注
          </Button>
        </div>
      </section>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// ai-caption — VLM 直出，无 WD14
// --------------------------------------------------------------------------- //

export function AiCaptionTool({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [mergeStrategy, setMergeStrategy] = useState("replace")
  const [task, setTask] = useState("")
  const [skipAnnotated, setSkipAnnotated] = useState(true)
  const [recursive, setRecursive] = useState(true)

  const start = async () => {
    const id = newTaskId()
    addTask({
      id,
      kind: "smart-caption",
      datasetPath,
      label: "VLM 批量 caption",
      sessioned: false,
    })
    try {
      const res = await imageStudioBatchCaption({
        path: datasetPath,
        recursive,
        task: task.trim() || undefined,
        mergeStrategy,
        skipAnnotated,
      })
      updateTask(id, {
        status: "completed",
        processed: res.processed,
        label: res.skipped
          ? `VLM caption 完成（跳过 ${res.skipped} 已有）`
          : "VLM caption 完成",
      })
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    } catch (err) {
      updateTask(id, {
        status: "failed",
        errorMsg: err instanceof Error ? err.message : String(err),
        label: "VLM caption 失败",
      })
    }
  }

  return (
    <div className="h-full overflow-y-auto p-4 max-w-2xl space-y-3">
      <TaskBanner datasetPath={datasetPath} />
      <section className="rounded-md border border-border/60 bg-card flex flex-col">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Sparkles className="size-3.5" />
          <span className="text-xs font-medium">VLM 批量 caption</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            把图直接喂给 VLM，让它写一句 caption。和「智能 caption」的差别：
            这里不调 WD14，结果完全由 VLM 决定，适合通用风格数据集。
            想加结构化标签的话用智能 caption。
          </p>
          <Row label="任务模板">
            <Input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="留空 = 用全局默认 vision task"
              className="h-8 text-xs"
            />
          </Row>
          <Row label="合并策略">
            <Select
              value={mergeStrategy}
              onValueChange={(v) => v && setMergeStrategy(v)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="replace">替换</SelectItem>
                <SelectItem value="append">追加</SelectItem>
                <SelectItem value="prepend">前置</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <div className="flex items-center gap-3 flex-wrap">
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={skipAnnotated} onCheckedChange={setSkipAnnotated} />
              跳过已有 .txt
            </label>
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={recursive} onCheckedChange={setRecursive} />
              递归子目录
            </label>
          </div>
          <Button size="sm" onClick={start} className="w-full gap-1">
            <Play className="size-3" />
            启动 VLM caption
          </Button>
        </div>
      </section>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// ai-quality — 0-10 质量分
// --------------------------------------------------------------------------- //

export function AiQualityTool({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [task, setTask] = useState("")
  const [skipScored, setSkipScored] = useState(true)
  const [recursive, setRecursive] = useState(true)

  const start = async () => {
    try {
      const submit = await startQualitySession({
        path: datasetPath,
        recursive,
        task: task.trim() || undefined,
        skipScored,
      })
      addTask({
        id: submit.session_id,
        kind: "quality-score",
        datasetPath,
        label: submit.skipped
          ? `质量评分（跳过 ${submit.skipped} 已评分）`
          : "质量评分",
        total: submit.total,
      })
      toast.success("已启动质量评分", {
        description: `共 ${submit.total} 张 · 后台进行 · 刷新后可恢复进度`,
      })
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    } catch (err) {
      toast.error("启动失败", {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  return (
    <div className="h-full overflow-y-auto p-4 max-w-xl space-y-3">
      <TaskBanner datasetPath={datasetPath} />
      <section className="rounded-md border border-border/60 bg-card flex flex-col">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Gauge className="size-3.5" />
          <span className="text-xs font-medium">AI 质量评分</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            对每张图打 0-10 分 + 标签 + 原因，结果存到图片的 store 里（不写 .txt）。
            可在「整理总览」工具里按分数过滤剔除低质。
          </p>
          <Row label="任务模板">
            <Input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="留空 = 默认质量评估 prompt"
              className="h-8 text-xs"
            />
          </Row>
          <div className="flex items-center gap-3 flex-wrap">
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={skipScored} onCheckedChange={setSkipScored} />
              跳过已评分
            </label>
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={recursive} onCheckedChange={setRecursive} />
              递归子目录
            </label>
          </div>
          <Button size="sm" onClick={start} className="w-full gap-1">
            <Play className="size-3" />
            启动质量评分
          </Button>
        </div>
      </section>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// ai-trigger-words
// --------------------------------------------------------------------------- //

export function AiTriggerWordsTool({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [task, setTask] = useState("")
  const [skipAnalyzed, setSkipAnalyzed] = useState(true)
  const [recursive, setRecursive] = useState(true)
  const [topN, setTopN] = useState<ImageStudioTriggerWordsResult["dataset_top"] | null>(
    null,
  )

  const start = async () => {
    const id = newTaskId()
    setTopN(null)
    addTask({
      id,
      kind: "trigger-words",
      datasetPath,
      label: "分析触发词",
      sessioned: false,
    })
    try {
      const res = await imageStudioBatchTriggerWords({
        path: datasetPath,
        recursive,
        task: task.trim() || undefined,
        skipAnalyzed,
      })
      setTopN(res.dataset_top)
      updateTask(id, {
        status: "completed",
        processed: res.processed,
        label: res.skipped
          ? `触发词分析完成（跳过 ${res.skipped} 已分析）`
          : "触发词分析完成",
      })
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    } catch (err) {
      updateTask(id, {
        status: "failed",
        errorMsg: err instanceof Error ? err.message : String(err),
        label: "触发词分析失败",
      })
    }
  }

  return (
    <div className="h-full overflow-y-auto p-4 max-w-2xl space-y-3">
      <TaskBanner datasetPath={datasetPath} />
      <section className="rounded-md border border-border/60 bg-card flex flex-col">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Tags className="size-3.5" />
          <span className="text-xs font-medium">AI 触发词抽取</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            逐图让 VLM 推荐 1-3 个适合作触发词的短语。完成后会汇总数据集层面的
            top-N 候选 — 点击复制。
          </p>
          <Row label="任务模板">
            <Input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="留空 = 默认触发词抽取 prompt"
              className="h-8 text-xs"
            />
          </Row>
          <div className="flex items-center gap-3 flex-wrap">
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={skipAnalyzed} onCheckedChange={setSkipAnalyzed} />
              跳过已分析
            </label>
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={recursive} onCheckedChange={setRecursive} />
              递归子目录
            </label>
          </div>
          <Button size="sm" onClick={start} className="w-full gap-1">
            <Play className="size-3" />
            启动触发词分析
          </Button>
        </div>
      </section>
      {topN && topN.length > 0 && (
        <section className="rounded-md border border-border/60 bg-muted/20 p-3">
          <div className="text-xs font-medium mb-2">数据集触发词候选（点击复制）</div>
          <div className="flex flex-wrap gap-2">
            {topN.map((t) => (
              <button
                key={t.trigger}
                type="button"
                onClick={() => void navigator.clipboard.writeText(t.trigger)}
                className="inline-flex items-center gap-1 rounded border bg-background px-2 py-0.5 text-[11px] font-mono hover:bg-muted"
                title={`${t.count} 张含此触发词 · 点击复制`}
              >
                <span>{t.trigger}</span>
                <span className="text-muted-foreground">·{t.count}</span>
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------- //
// ai-wd14-prefilter — 单图测试入口（调试用）
// --------------------------------------------------------------------------- //

export function AiWd14PrefilterTool({ datasetPath }: { datasetPath: string }) {
  const [imagePath, setImagePath] = useState("")
  const [taggerModel, setTaggerModel] = useState<string>(FALLBACK_DEFAULT_MODEL)
  const [device, setDevice] = useState("auto")
  const [generalThreshold, setGeneralThreshold] = useState(0.35)
  const [characterThreshold, setCharacterThreshold] = useState(0.85)
  const [captionMode, setCaptionMode] =
    useState<"general" | "style" | "character">("style")
  const [captionSource, setCaptionSource] = useState<"vlm" | "tags">("vlm")
  const [triggerWord, setTriggerWord] = useState("")
  const [stripStyleTags, setStripStyleTags] = useState(true)
  const [result, setResult] = useState<Wd14PrefilterResult | null>(null)

  // 与 ai-bulk-modal 同源的真实模型列表
  const wd14Models = useQuery({
    queryKey: ["wd14-models"],
    queryFn: api.listWd14Models,
    staleTime: 60 * 60 * 1000,
  })
  useEffect(() => {
    if (wd14Models.data?.default && taggerModel === FALLBACK_DEFAULT_MODEL) {
      setTaggerModel(wd14Models.data.default)
    }
  }, [wd14Models.data?.default, taggerModel])
  const modelOptions = wd14Models.data?.models ?? [
    { id: FALLBACK_DEFAULT_MODEL, label: "v3 · EvaCLIP-Large(推荐)" },
  ]

  const mutation = useMutation({
    mutationFn: () =>
      imageStudioWd14Prefilter({
        path: imagePath.trim(),
        taggerModel,
        device,
        generalThreshold,
        characterThreshold,
        captionMode,
        captionSource,
        triggerWord: triggerWord.trim() || undefined,
        stripStyleTags,
      }),
    onSuccess: (data) => setResult(data),
    onError: (err) =>
      toast.error("prefilter 失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  return (
    <div className="h-full overflow-y-auto p-4 max-w-3xl grid gap-3 lg:grid-cols-2">
      <section className="rounded-md border border-border/60 bg-card">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Tags className="size-3.5" />
          <span className="text-xs font-medium">WD14 单步出标签 · 单图测试</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            智能 caption 第一步独立出来：给一张图，拿到 WD14 标签 +
            assemble 好的 prompt。结果可直接给「VLM Anima 重写」工具用。
          </p>
          <Row label="图片路径">
            <Input
              value={imagePath}
              onChange={(e) => setImagePath(e.target.value)}
              placeholder={`${datasetPath}/sample.png`}
              className="h-8 text-xs font-mono"
            />
          </Row>
          <Row label="模型">
            <Select value={taggerModel} onValueChange={(v) => v && setTaggerModel(v)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {modelOptions.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Row>
          <Row label="设备">
            <Select value={device} onValueChange={(v) => v && setDevice(v)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">自动</SelectItem>
                <SelectItem value="cuda">CUDA</SelectItem>
                <SelectItem value="cpu">CPU</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <div className="grid grid-cols-2 gap-2">
            <Row label="通用阈值">
              <Input
                type="number"
                step="0.05"
                min={0}
                max={1}
                value={generalThreshold}
                onChange={(e) => setGeneralThreshold(Number(e.target.value))}
                className="h-8 text-xs font-mono"
              />
            </Row>
            <Row label="角色阈值">
              <Input
                type="number"
                step="0.05"
                min={0}
                max={1}
                value={characterThreshold}
                onChange={(e) => setCharacterThreshold(Number(e.target.value))}
                className="h-8 text-xs font-mono"
              />
            </Row>
          </div>
          <Row label="LLM 输入">
            <Select
              value={captionSource}
              onValueChange={(v) => v && setCaptionSource(v as typeof captionSource)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="vlm">视觉模型</SelectItem>
                <SelectItem value="tags">仅 WD14 标签</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Row label="训练用途">
            <Select
              value={captionMode}
              onValueChange={(v) => v && setCaptionMode(v as typeof captionMode)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="style">风格</SelectItem>
                <SelectItem value="character">角色</SelectItem>
                <SelectItem value="general">通用</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Row label="触发词">
            <Input
              value={triggerWord}
              onChange={(e) => setTriggerWord(e.target.value)}
              placeholder="anima style / @charA"
              className="h-8 text-xs"
            />
          </Row>
          {captionMode === "style" && (
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={stripStyleTags} onCheckedChange={setStripStyleTags} />
              剔除画风类标签
            </label>
          )}
          <Button
            size="sm"
            disabled={!imagePath.trim() || mutation.isPending}
            onClick={() => mutation.mutate()}
            className="w-full gap-1"
          >
            {mutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Play className="size-3" />
            )}
            运行 WD14 prefilter
          </Button>
        </div>
      </section>
      <PrefilterResultCard result={result} />
    </div>
  )
}

function PrefilterResultCard({ result }: { result: Wd14PrefilterResult | null }) {
  if (!result) {
    return (
      <section className="rounded-md border border-dashed border-border/60 p-3 text-xs text-muted-foreground flex items-center justify-center min-h-32">
        点左侧「运行」后结果会显示在这里。可直接拷给「VLM Anima 重写」。
      </section>
    )
  }
  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <ImageIcon className="size-3.5" />
        <span className="text-xs font-medium">prefilter 结果</span>
      </div>
      <div className="p-3 space-y-2 text-xs">
        <Field label="rating">{result.ratingName ?? "—"}</Field>
        <Field label="general tags">
          <span className="font-mono break-words">
            {result.generalTags.join(", ")}
          </span>
        </Field>
        <Field label="character tags">
          <span className="font-mono break-words">
            {result.characterTags.join(", ") || "—"}
          </span>
        </Field>
        <Field label="prompt">
          <pre className="font-mono whitespace-pre-wrap text-[11px] bg-muted/30 rounded p-2 max-h-64 overflow-auto">
            {result.promptText}
          </pre>
        </Field>
        <Button
          size="sm"
          variant="outline"
          className="w-full text-[11px]"
          onClick={() => {
            void navigator.clipboard.writeText(JSON.stringify(result, null, 2))
            toast.success("已拷贝 prefilter JSON")
          }}
        >
          复制 JSON
        </Button>
      </div>
    </section>
  )
}

// --------------------------------------------------------------------------- //
// ai-vlm-anima-rewrite — 单图测试，把 prefilter 的产出送到 VLM
// --------------------------------------------------------------------------- //

export function AiVlmAnimaRewriteTool({ datasetPath }: { datasetPath: string }) {
  const [json, setJson] = useState("")
  const [mergeStrategy, setMergeStrategy] = useState("replace")
  const [result, setResult] = useState<{
    path: string
    wd14Tags: string
    caption: string
  } | null>(null)

  const mutation = useMutation({
    mutationFn: () => {
      let parsed: Wd14PrefilterResult
      try {
        parsed = JSON.parse(json) as Wd14PrefilterResult
      } catch {
        throw new Error("JSON 解析失败 — 请粘贴 prefilter 工具的完整输出")
      }
      return imageStudioVlmAnimaRewrite({
        path: parsed.path,
        mergeStrategy,
        captionMode: "style",
        captionSource: parsed.captionSource,
        stripStyleTags: parsed.stripStyleTags,
        ratingName: parsed.ratingName,
        generalTags: parsed.generalTags,
        characterTags: parsed.characterTags,
        promptText: parsed.promptText,
        dataUrl: parsed.dataUrl,
        skipLlm: parsed.skipLlm,
      })
    },
    onSuccess: (data) =>
      setResult({ path: data.path, wd14Tags: data.wd14Tags, caption: data.caption }),
    onError: (err) =>
      toast.error("VLM 重写失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  return (
    <div className="h-full overflow-y-auto p-4 max-w-3xl space-y-3">
      <section className="rounded-md border border-border/60 bg-card">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Sparkles className="size-3.5" />
          <span className="text-xs font-medium">VLM Anima 重写 · 单图测试</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            把「WD14 单步出标签」工具复制出来的 JSON 粘到这里，调 VLM 写 caption
            并写入 <code>.txt</code>。 主要用来对比不同 prompt / merge 策略下的产出。
            数据集 <code>{datasetPath}</code> 仅用于校验路径前缀。
          </p>
          <textarea
            value={json}
            onChange={(e) => setJson(e.target.value)}
            placeholder='{"path":"...","ratingName":"...","generalTags":[...]...}'
            className="w-full h-40 rounded border bg-background px-2 py-1.5 text-[11px] font-mono resize-y"
          />
          <Row label="合并策略">
            <Select
              value={mergeStrategy}
              onValueChange={(v) => v && setMergeStrategy(v)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="replace">替换</SelectItem>
                <SelectItem value="append">追加</SelectItem>
                <SelectItem value="prepend">前置</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Button
            size="sm"
            disabled={!json.trim() || mutation.isPending}
            onClick={() => mutation.mutate()}
            className="w-full gap-1"
          >
            {mutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Play className="size-3" />
            )}
            写入 caption
          </Button>
        </div>
      </section>
      {result && (
        <section className="rounded-md border border-border/60 bg-card">
          <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
            <ImageIcon className="size-3.5" />
            <span className="text-xs font-medium">写入结果</span>
          </div>
          <div className="p-3 space-y-2 text-xs">
            <Field label="path">
              <code className="font-mono">{result.path}</code>
            </Field>
            <Field label="wd14 tags">
              <pre className="font-mono whitespace-pre-wrap text-[11px] bg-muted/30 rounded p-2 max-h-32 overflow-auto">
                {result.wd14Tags}
              </pre>
            </Field>
            <Field label="caption">
              <pre className="font-mono whitespace-pre-wrap text-[11px] bg-muted/30 rounded p-2 max-h-48 overflow-auto">
                {result.caption}
              </pre>
            </Field>
          </div>
        </section>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

function Row({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="grid grid-cols-[5rem_1fr] items-center gap-2">
      <span className="text-muted-foreground">{label}</span>
      <div className="min-w-0">{children}</div>
    </div>
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
    <div className="space-y-0.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div>{children}</div>
    </div>
  )
}

// 占位避免 lint 警告 — startTaggingSession 暂未直接使用，但保留 import 以便后续
// 给 tagging-wd14 工具搬过来用。
void startTaggingSession

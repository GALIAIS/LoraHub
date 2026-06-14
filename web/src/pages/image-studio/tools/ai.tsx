/**
 * AI 类工具页面集合。
 *
 *   - ai-smart-caption     : 复用现成的 batch + 全局任务条 (smart-caption 走 sessioned poll)
 *   - ai-caption           : 直接 VLM 写 caption（sessioned poll）
 *   - ai-quality           : 给图打质量分（sessioned poll）
 *   - ai-trigger-words     : 让 VLM 推荐触发词候选（sessioned poll）
 *   - ai-wd14-prefilter    : 单图测试 - 拿到 WD14 标签 + 拼装好的 prompt
 *   - ai-vlm-anima-rewrite : 单图测试 - 把上一个的产出喂给 VLM 写 caption
 *
 * 批量类工具入队到全局 studio-task-store, banner 在 tool-page 顶部由
 * 调用方渲染（这里通过 GlobalTaskBanner 自渲染，跨 tool 切换都可见）。
 */
import { useEffect, useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import {
  BookOpen,
  Gauge,
  Play,
  Sparkles,
  Tags,
  Wand2,
} from "lucide-react"
import { toast } from "sonner"
import {
  getTriggerWordsSession,
  startCaptionSession,
  startQualitySession,
  startSmartCaptionSession,
  startTaggingSession,
  startTriggerWordsSession,
  type ImageStudioTriggerWordsResult,
} from "@/lib/api"
import { addTask } from "@/lib/studio-task-store"
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
import { Row, TaskBanner } from "./ai-shared"
export {
  AiVlmAnimaRewriteTool,
  AiWd14PrefilterTool,
} from "./ai-single-image"

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
    try {
      const submit = await startCaptionSession({
        path: datasetPath,
        recursive,
        task: task.trim() || undefined,
        mergeStrategy,
        skipAnnotated,
      })
      addTask({
        id: submit.session_id,
        kind: "caption",
        datasetPath,
        label: submit.skipped
          ? `VLM 批量 caption（跳过 ${submit.skipped} 已有）`
          : "VLM 批量 caption",
        total: submit.total,
      })
      toast.success("已启动 VLM caption", {
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
  const terminalSig = useStudioTasksFor(datasetPath)
    .filter((t) => t.kind === "trigger-words" && t.status === "completed")
    .map((t) => `${t.id}:${t.status}`)
    .join(",")

  useEffect(() => {
    if (!terminalSig) return
    const id = terminalSig.split(":")[0]
    let cancelled = false
    getTriggerWordsSession(id)
      .then((snap) => {
        if (!cancelled) setTopN(snap.dataset_top)
      })
      .catch(() => {
        // The banner covers task failures; the top-N panel is optional.
      })
    return () => {
      cancelled = true
    }
  }, [terminalSig])

  const start = async () => {
    setTopN(null)
    try {
      const submit = await startTriggerWordsSession({
        path: datasetPath,
        recursive,
        task: task.trim() || undefined,
        skipAnalyzed,
      })
      addTask({
        id: submit.session_id,
        kind: "trigger-words",
        datasetPath,
        label: submit.skipped
          ? `分析触发词（跳过 ${submit.skipped} 已分析）`
          : "分析触发词",
        total: submit.total,
      })
      toast.success("已启动触发词分析", {
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

// 占位避免 lint 警告 — startTaggingSession 暂未直接使用，但保留 import 以便后续
// 给 tagging-wd14 工具搬过来用。
void startTaggingSession

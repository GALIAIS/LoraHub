import { useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  CheckCircle2,
  Loader2,
  Play,
  RefreshCw,
  Replace,
  Tag as TagIcon,
  Trash2,
  Wand2,
} from "lucide-react"
import { toast } from "sonner"
import {
  api,
  imageStudioCaptionsBlacklist,
  imageStudioCaptionsFindReplace,
  imageStudioCaptionsInjectTrigger,
  imageStudioCaptionsVocab,
  startCaptionSession,
  startSmartCaptionSession,
  startTaggingSession,
  type CaptionDiff,
  type CaptionVocabRow,
} from "@/lib/api"
import { addTask } from "@/lib/studio-task-store"
import { useStudioTasksFor } from "@/hooks/use-studio-tasks"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import {
  AiQualityTool,
  AiTriggerWordsTool,
} from "./annotate-ai-tools"
import {
  useRefreshCaptionViewsOnTaskDone,
  useTaggerDownloadDisplay,
} from "./annotate-ai-shared"
import { CaptionsBlacklistTool } from "./annotate-caption-tools"
import {
  CaptionPromptPicker,
  type CaptionPromptValue,
} from "../caption-prompt-picker"

interface Props {
  datasetPath: string
}

const FALLBACK_DEFAULT_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"

type AnnotationMode = "tag" | "nl" | "tag-llm" | "tag-vlm" | "toriigate"
type DetailMode = "vocab" | "replace" | "trigger"

const TOOL_TO_ANNOTATE_VIEW: Record<
  string,
  { mode?: AnnotationMode; detail?: DetailMode }
> = {
  "tagging-wd14": { mode: "tag" },
  "captions-vocab": { detail: "vocab" },
  "captions-find-replace": { detail: "replace" },
  "captions-inject-trigger": { detail: "trigger" },
  "captions-blacklist": { detail: "vocab" },
  "ai-caption": { mode: "nl" },
  "ai-smart-caption": { mode: "tag-vlm" },
  "ai-wd14-prefilter": { mode: "tag" },
  "ai-vlm-anima-rewrite": { mode: "tag-vlm" },
  "ai-quality": { mode: "nl" },
  "ai-trigger-words": { detail: "trigger" },
}

export function AnnotateStage({ datasetPath }: Props) {
  const [params] = useSearchParams()
  const tool = params.get("tool")
  if (tool === "ai-quality") {
    return <AiQualityTool datasetPath={datasetPath} />
  }
  if (tool === "ai-trigger-words") {
    return <AiTriggerWordsTool datasetPath={datasetPath} />
  }
  if (tool === "captions-blacklist") {
    return <CaptionsBlacklistTool datasetPath={datasetPath} />
  }
  return <AnnotateMainPanel datasetPath={datasetPath} />
}

function AnnotateMainPanel({ datasetPath }: Props) {
  const qc = useQueryClient()
  const [params] = useSearchParams()
  useRefreshCaptionViewsOnTaskDone(datasetPath)
  const tasks = useStudioTasksFor(datasetPath)
  const latestTask = [...tasks].sort((a, b) => b.startedAt - a.startedAt)[0]
  const [mode, setMode] = useState<AnnotationMode>("tag")
  const [detail, setDetail] = useState<DetailMode>("vocab")
  const [model, setModel] = useState(FALLBACK_DEFAULT_MODEL)
  const [device, setDevice] = useState<"auto" | "cuda" | "cpu">("auto")
  const [recursive, setRecursive] = useState(true)
  const [overwrite, setOverwrite] = useState(false)
  const [general, setGeneral] = useState(0.35)
  const [character, setCharacter] = useState(0.85)
  const [captionMode, setCaptionMode] =
    useState<"general" | "style" | "character">("style")
  const [captionPrompt, setCaptionPrompt] =
    useState<CaptionPromptValue>("style")
  const [promptTemplate, setPromptTemplate] = useState<string | undefined>()
  const [useWd14, setUseWd14] = useState(true)
  const [skipExisting, setSkipExisting] = useState(true)
  const [triggerWord, setTriggerWord] = useState("")

  useEffect(() => {
    const tool = params.get("tool")
    if (!tool) return
    const view = TOOL_TO_ANNOTATE_VIEW[tool]
    if (!view) return
    if (view.mode) setMode(view.mode)
    if (view.detail) setDetail(view.detail)
  }, [params])

  const vocabQuery = useQuery({
    queryKey: ["image-studio-captions-vocab", datasetPath],
    queryFn: () =>
      imageStudioCaptionsVocab(datasetPath, { recursive: true, limit: 200 }),
    enabled: Boolean(datasetPath),
  })

  const wd14Models = useQuery({
    queryKey: ["wd14-models"],
    queryFn: api.listWd14Models,
    staleTime: 60 * 60 * 1000,
  })

  useEffect(() => {
    if (wd14Models.data?.default && model === FALLBACK_DEFAULT_MODEL) {
      setModel(wd14Models.data.default)
    }
  }, [model, wd14Models.data?.default])

  const modelOptions = wd14Models.data?.models ?? [
    { id: FALLBACK_DEFAULT_MODEL, label: "v3 · EvaCLIP-Large" },
  ]

  const onMutated = () => {
    qc.invalidateQueries({ queryKey: ["image-studio-captions-vocab", datasetPath] })
    qc.invalidateQueries({ queryKey: ["image-studio-audit-report", datasetPath] })
    qc.invalidateQueries({ queryKey: ["image-studio"] })
  }

  const startMutation = useMutation({
    mutationFn: async () => {
      if (mode === "tag") {
        const session = await startTaggingSession({
          path: datasetPath,
          model_id: model,
          device,
          general,
          character,
          overwrite,
          recursive,
        })
        addTask({
          id: session.session_id,
          kind: "wd14",
          datasetPath,
          label: "TAG 模式",
        })
        return "TAG 模式已启动"
      }
      if (mode === "nl") {
        const session = await startCaptionSession({
          path: datasetPath,
          recursive,
          skipAnnotated: skipExisting,
          mergeStrategy: "replace",
        })
        addTask({
          id: session.session_id,
          kind: "caption",
          datasetPath,
          label: "NL 模式",
          total: session.total,
        })
        return `NL 模式已启动，共 ${session.total} 张`
      }
      const captionSource =
        mode === "tag-llm" ? "tags" : mode === "toriigate" ? "toriigate" : "vlm"
      const session = await startSmartCaptionSession({
        path: datasetPath,
        recursive,
        device,
        captionMode,
        promptTemplate,
        captionSource,
        triggerWord: triggerWord.trim() || undefined,
        stripStyleTags: true,
        useWd14: mode === "toriigate" ? useWd14 : undefined,
        skipExisting,
      })
      addTask({
        id: session.session_id,
        kind: "smart-caption",
        datasetPath,
        label:
          mode === "tag-llm"
            ? "TAG+LLM 模式"
            : mode === "toriigate"
              ? "ToriiGate 模式"
              : "TAG+VLM 模式",
        total: session.total,
      })
      return `${mode === "tag-llm" ? "TAG+LLM" : mode === "toriigate" ? "ToriiGate" : "TAG+VLM"} 模式已启动，共 ${session.total} 张`
    },
    onSuccess: (message) => {
      toast.success(message)
      onMutated()
    },
    onError: (err) =>
      toast.error("启动失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto grid max-w-6xl gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
          <section className="rounded-md border border-border/60 bg-card">
            <div className="border-b border-border/60 px-4 py-3">
              <div className="flex items-center gap-2">
                <TagIcon className="size-4" />
                <h2 className="text-sm font-semibold">打标任务</h2>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                选择一种模式后启动，结果统一写入图片同名 txt。
              </p>
            </div>

            <div className="space-y-4 p-4">
              <Field label="模式">
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value as AnnotationMode)}
                  className="h-9 w-full rounded border bg-background px-2 text-sm"
                >
                  <option value="tag">TAG 模式</option>
                  <option value="nl">NL 模式</option>
                  <option value="tag-llm">TAG+LLM 模式</option>
                  <option value="tag-vlm">TAG+VLM 模式</option>
                  <option value="toriigate">ToriiGate 模式</option>
                </select>
              </Field>

              {mode === "tag" ? (
                <div className="grid gap-3 md:grid-cols-2">
                  <Field label="模型">
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="h-8 w-full rounded border bg-background px-2 text-xs"
                    >
                      {modelOptions.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.label}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <DeviceSelect value={device} onChange={setDevice} />
                  <NumberField label="通用阈值" value={general} onChange={setGeneral} />
                  <NumberField label="角色阈值" value={character} onChange={setCharacter} />
                </div>
              ) : mode === "nl" ? (
                <div className="grid gap-3 md:grid-cols-2">
                  <Field label="输出">
                    <div className="flex h-8 items-center rounded border bg-muted/25 px-2 text-xs text-muted-foreground">
                      VLM 自然语言描述
                    </div>
                  </Field>
                </div>
              ) : (
                <div className="grid gap-3 md:grid-cols-2">
                  <DeviceSelect value={device} onChange={setDevice} />
                  <Field label="Caption 类型">
                    <CaptionPromptPicker
                      value={captionPrompt}
                      onChange={(next) => {
                        setCaptionPrompt(next.value)
                        setCaptionMode(next.captionMode)
                        setPromptTemplate(next.promptTemplate)
                      }}
                    />
                  </Field>
                  <Field label="来源">
                    <div className="flex h-8 items-center rounded border bg-muted/25 px-2 text-xs text-muted-foreground">
                      {mode === "tag-llm"
                        ? "TAG → LLM 文本重写"
                        : mode === "toriigate"
                          ? "ToriiGate 官方 short 格式"
                          : "TAG → VLM 视觉重写"}
                    </div>
                  </Field>
                  <Field label="触发词">
                    <Input
                      value={triggerWord}
                      onChange={(e) => setTriggerWord(e.target.value)}
                      placeholder="可选，如 shinken_gomi"
                      className="h-8 text-xs"
                    />
                  </Field>
                </div>
              )}

              <div className="flex flex-wrap items-center gap-4 border-t border-border/50 pt-3 text-xs">
                <Toggle label="递归子目录" checked={recursive} onChange={setRecursive} />
                {mode === "tag" ? (
                  <Toggle label="覆盖已有" checked={overwrite} onChange={setOverwrite} />
                ) : (
                  <Toggle label="跳过已有 caption" checked={skipExisting} onChange={setSkipExisting} />
                )}
                {mode === "toriigate" ? (
                  <Toggle label="使用 WD14 参考标签" checked={useWd14} onChange={setUseWd14} />
                ) : null}
                <Button
                  type="button"
                  disabled={startMutation.isPending || latestTask?.status === "running"}
                  onClick={() => startMutation.mutate()}
                  className="ml-auto h-8 gap-1.5"
                >
                  {startMutation.isPending ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <Play className="size-3.5" />
                  )}
                  开始
                </Button>
              </div>
            </div>
          </section>

          <TaskCard task={latestTask} />

          <section className="rounded-md border border-border/60 bg-card xl:col-span-2">
            <div className="flex flex-wrap items-center gap-1 border-b border-border/60 px-3 py-2">
              <Segment active={detail === "vocab"} onClick={() => setDetail("vocab")}>
                词表
              </Segment>
              <Segment active={detail === "replace"} onClick={() => setDetail("replace")}>
                查找替换
              </Segment>
              <Segment active={detail === "trigger"} onClick={() => setDetail("trigger")}>
                触发词
              </Segment>
              <Button
                size="sm"
                variant="ghost"
                className="ml-auto h-7 px-2 text-[11px]"
                onClick={() => vocabQuery.refetch()}
                disabled={vocabQuery.isLoading}
              >
                {vocabQuery.isLoading ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <RefreshCw className="size-3" />
                )}
              </Button>
            </div>
            <div className="min-h-[360px]">
              {detail === "vocab" && (
                <VocabPanel
                  vocab={vocabQuery.data?.vocab ?? []}
                  loading={vocabQuery.isLoading}
                  totalFiles={vocabQuery.data?.files_seen ?? 0}
                  tagCount={vocabQuery.data?.tag_count ?? 0}
                  datasetPath={datasetPath}
                  onMutated={onMutated}
                />
              )}
              {detail === "replace" && (
                <FindReplacePanel datasetPath={datasetPath} onMutated={onMutated} />
              )}
              {detail === "trigger" && (
                <TriggerInjectPanel datasetPath={datasetPath} onMutated={onMutated} />
              )}
            </div>
          </section>
        </div>
      </div>
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
    <label className="grid gap-1 text-xs">
      <span className="text-muted-foreground">{label}</span>
      {children}
    </label>
  )
}

function DeviceSelect({
  value,
  onChange,
}: {
  value: "auto" | "cuda" | "cpu"
  onChange: (value: "auto" | "cuda" | "cpu") => void
}) {
  return (
    <Field label="设备">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as "auto" | "cuda" | "cpu")}
        className="h-8 w-full rounded border bg-background px-2 text-xs"
      >
        <option value="auto">自动</option>
        <option value="cuda">CUDA</option>
        <option value="cpu">CPU</option>
      </select>
    </Field>
  )
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (value: number) => void
}) {
  return (
    <Field label={label}>
      <Input
        type="number"
        min="0"
        max="1"
        step="0.05"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-8 text-xs"
      />
    </Field>
  )
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <label className="inline-flex items-center gap-2">
      <Switch checked={checked} onCheckedChange={onChange} />
      {label}
    </label>
  )
}

function Segment({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <Button
      type="button"
      variant={active ? "secondary" : "ghost"}
      size="sm"
      className="h-7 px-3 text-xs"
      onClick={onClick}
    >
      {children}
    </Button>
  )
}

function TaskCard({
  task,
}: {
  task:
        | {
        kind?: string
        label: string
        status: string
        processed?: number
        total?: number
        lastImage?: string
        errorMsg?: string
      }
    | undefined
}) {
  const download = useTaggerDownloadDisplay(
    task?.status === "running" && (task.kind === "wd14" || task.kind === "smart-caption"),
  )
  const total = task?.total ?? 0
  const processed = task?.processed ?? 0
  const percent = download?.percent ?? (total > 0 ? Math.round((processed / total) * 100) : 0)
  return (
    <section className="rounded-md border border-border/60 bg-card p-4">
      <div className="flex items-center gap-2">
        {task?.status === "running" ? (
          <Loader2 className="size-4 animate-spin" />
        ) : (
          <CheckCircle2 className="size-4 text-muted-foreground" />
        )}
        <h3 className="text-sm font-semibold">任务状态</h3>
      </div>
      {!task ? (
        <p className="mt-3 text-xs text-muted-foreground">
          暂无标注任务。选择左侧任务类型后开始。
        </p>
      ) : (
        <div className="mt-3 space-y-2 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="font-medium">{download?.label ?? task.label}</span>
            <span className="text-muted-foreground">{task.status}</span>
          </div>
          <div className="shiro-progress-track h-1.5 border-0 bg-muted">
            <div
              className={cn(
                "shiro-progress-fill",
                task.status === "failed" ? "bg-destructive" : "bg-primary",
              )}
              style={{ width: `${percent}%` }}
            />
          </div>
          {total > 0 && (
            <p className="text-muted-foreground">
              {processed} / {total}
            </p>
          )}
          {task.lastImage && (
            <p className="truncate text-muted-foreground">{task.lastImage}</p>
          )}
          {task.errorMsg && <p className="text-destructive">{task.errorMsg}</p>}
        </div>
      )}
    </section>
  )
}

function VocabPanel({
  vocab,
  loading,
  totalFiles,
  tagCount,
  datasetPath,
  onMutated,
}: {
  vocab: CaptionVocabRow[]
  loading: boolean
  totalFiles: number
  tagCount: number
  datasetPath: string
  onMutated: () => void
}) {
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set())
  const toggleTag = (tag: string) =>
    setSelectedTags((prev) => {
      const next = new Set(prev)
      if (next.has(tag)) next.delete(tag)
      else next.add(tag)
      return next
    })

  const blacklistMutation = useMutation({
    mutationFn: () =>
      imageStudioCaptionsBlacklist({
        dataset_path: datasetPath,
        tags: Array.from(selectedTags),
      }),
    onSuccess: (data) => {
      toast.success(`已删除 ${data.removed_count} 处 tag`)
      setSelectedTags(new Set())
      onMutated()
    },
    onError: (err) =>
      toast.error("黑名单删除失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  const max = vocab[0]?.count ?? 1
  return (
    <div className="flex h-full min-h-[360px] flex-col">
      <div className="flex items-center gap-2 border-b border-border/40 px-3 py-2 text-xs">
        <span className="font-medium">标签词表</span>
        <span className="text-muted-foreground">
          {tagCount} 种 · {totalFiles} 文件
        </span>
        {selectedTags.size > 0 && (
          <Button
            size="sm"
            variant="outline"
            className="ml-auto h-7 gap-1 px-2 text-[11px]"
            disabled={blacklistMutation.isPending}
            onClick={() => {
              if (!window.confirm(`确认删除 ${selectedTags.size} 个 tag？`)) return
              blacklistMutation.mutate()
            }}
          >
            {blacklistMutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Trash2 className="size-3" />
            )}
            删除选中
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex h-32 items-center justify-center text-xs text-muted-foreground">
            <Loader2 className="mr-2 size-4 animate-spin" />
            扫描标签...
          </div>
        )}
        {!loading && vocab.length === 0 && (
          <div className="flex h-32 items-center justify-center text-xs text-muted-foreground">
            未找到 caption 标签
          </div>
        )}
        {!loading && vocab.length > 0 && (
          <ul className="divide-y divide-border/30">
            {vocab.map((row) => {
              const checked = selectedTags.has(row.tag)
              return (
                <li
                  key={row.tag}
                  className={cn(
                    "relative flex cursor-pointer items-center gap-2 px-3 py-1 text-[11px] hover:bg-muted/40",
                    checked && "bg-muted/50",
                  )}
                  onClick={() => toggleTag(row.tag)}
                >
                  <span
                    aria-hidden
                    className="absolute inset-y-0 left-0 bg-primary/10"
                    style={{ width: `${(row.count / max) * 100}%` }}
                  />
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleTag(row.tag)}
                    onClick={(e) => e.stopPropagation()}
                    className="relative size-3"
                  />
                  <span className="relative flex-1 truncate" title={row.tag}>
                    {row.tag}
                  </span>
                  <span className="relative tabular-nums text-muted-foreground">
                    {row.count}
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </div>
  )
}

function FindReplacePanel({
  datasetPath,
  onMutated,
}: {
  datasetPath: string
  onMutated: () => void
}) {
  const [pattern, setPattern] = useState("")
  const [replacement, setReplacement] = useState("")
  const [isRegex, setIsRegex] = useState(false)
  const [caseSensitive, setCaseSensitive] = useState(false)
  const [wholeCaption, setWholeCaption] = useState(false)
  const [dryRunResult, setDryRunResult] = useState<{
    matched_files: number
    matched_count: number
    diffs: CaptionDiff[]
    diffs_truncated: boolean
  } | null>(null)

  const dryRunMutation = useMutation({
    mutationFn: () =>
      imageStudioCaptionsFindReplace({
        dataset_path: datasetPath,
        pattern,
        replacement,
        is_regex: isRegex,
        case_sensitive: caseSensitive,
        whole_caption: wholeCaption,
        dry_run: true,
      }),
    onSuccess: (data) => setDryRunResult(data),
    onError: (err) =>
      toast.error("预演失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  const applyMutation = useMutation({
    mutationFn: () =>
      imageStudioCaptionsFindReplace({
        dataset_path: datasetPath,
        pattern,
        replacement,
        is_regex: isRegex,
        case_sensitive: caseSensitive,
        whole_caption: wholeCaption,
        dry_run: false,
      }),
    onSuccess: (data) => {
      toast.success(`已应用替换：${data.matched_count} 处`)
      setDryRunResult(null)
      onMutated()
    },
    onError: (err) =>
      toast.error("应用失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  return (
    <div className="space-y-3 p-4 text-xs">
      <div className="flex items-center gap-2">
        <Replace className="size-3.5" />
        <span className="font-medium">查找替换</span>
      </div>
      <Input
        placeholder="查找"
        value={pattern}
        onChange={(e) => setPattern(e.target.value)}
        className="h-8 text-xs font-mono"
      />
      <Input
        placeholder="替换，留空则删除"
        value={replacement}
        onChange={(e) => setReplacement(e.target.value)}
        className="h-8 text-xs font-mono"
      />
      <div className="flex flex-wrap items-center gap-3">
        <Toggle label="正则" checked={isRegex} onChange={setIsRegex} />
        <Toggle label="大小写敏感" checked={caseSensitive} onChange={setCaseSensitive} />
        <Toggle label="整段匹配" checked={wholeCaption} onChange={setWholeCaption} />
      </div>
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          disabled={!pattern || dryRunMutation.isPending}
          onClick={() => dryRunMutation.mutate()}
          className="h-7"
        >
          预演
        </Button>
        <Button
          size="sm"
          disabled={!dryRunResult || dryRunResult.matched_count === 0 || applyMutation.isPending}
          onClick={() => {
            if (!dryRunResult) return
            if (!window.confirm(`确认应用 ${dryRunResult.matched_count} 处替换？`)) return
            applyMutation.mutate()
          }}
          className="h-7"
        >
          应用
        </Button>
        {dryRunResult && (
          <span className="text-muted-foreground">
            {dryRunResult.matched_count} 处 / {dryRunResult.matched_files} 文件
          </span>
        )}
      </div>
      {dryRunResult && dryRunResult.diffs.length > 0 && (
        <div className="max-h-56 overflow-y-auto rounded border border-border/50">
          {dryRunResult.diffs.slice(0, 100).map((d, i) => (
            <div key={i} className="border-t border-border/30 px-3 py-1.5 first:border-t-0">
              <div className="truncate font-mono text-muted-foreground" title={d.path}>
                {d.path.split(/[\\/]/).pop()}
              </div>
              <div className="truncate font-mono text-red-600/80 line-through" title={d.before}>
                {d.before}
              </div>
              <div className="truncate font-mono text-emerald-700/80" title={d.after}>
                {d.after}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function TriggerInjectPanel({
  datasetPath,
  onMutated,
}: {
  datasetPath: string
  onMutated: () => void
}) {
  const [trigger, setTrigger] = useState("")
  const [position, setPosition] = useState<"prepend" | "append">("prepend")
  const [skipExisting, setSkipExisting] = useState(true)

  const mutation = useMutation({
    mutationFn: () =>
      imageStudioCaptionsInjectTrigger({
        dataset_path: datasetPath,
        trigger_word: trigger.trim(),
        position,
        skip_existing: skipExisting,
      }),
    onSuccess: (data) => {
      toast.success(`已注入到 ${data.injected_count} 个 caption`)
      onMutated()
    },
    onError: (err) =>
      toast.error("注入失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  return (
    <div className="space-y-3 p-4 text-xs">
      <div className="flex items-center gap-2">
        <Wand2 className="size-3.5" />
        <span className="font-medium">触发词注入</span>
      </div>
      <Input
        placeholder="触发词，如 shinken_gomi"
        value={trigger}
        onChange={(e) => setTrigger(e.target.value)}
        className="h-8 text-xs font-mono"
      />
      <div className="flex flex-wrap items-center gap-3">
        <label className="inline-flex items-center gap-1.5">
          <input
            type="radio"
            checked={position === "prepend"}
            onChange={() => setPosition("prepend")}
          />
          前置
        </label>
        <label className="inline-flex items-center gap-1.5">
          <input
            type="radio"
            checked={position === "append"}
            onChange={() => setPosition("append")}
          />
          后置
        </label>
        <Toggle label="跳过已存在" checked={skipExisting} onChange={setSkipExisting} />
      </div>
      <Button
        size="sm"
        disabled={!trigger.trim() || mutation.isPending}
        onClick={() => mutation.mutate()}
        className="h-8"
      >
        {mutation.isPending ? <Loader2 className="mr-1 size-3 animate-spin" /> : null}
        注入
      </Button>
    </div>
  )
}

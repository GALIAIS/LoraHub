import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  Loader2,
  RefreshCw,
  Replace,
  Tag as TagIcon,
  Trash2,
  Wand2,
} from "lucide-react"
import { toast } from "sonner"
import {
  imageStudioCaptionsBlacklist,
  imageStudioCaptionsFindReplace,
  imageStudioCaptionsInjectTrigger,
  imageStudioCaptionsVocab,
  type CaptionDiff,
  type CaptionVocabRow,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

interface Props {
  datasetPath: string
}

export function AnnotateStage({ datasetPath }: Props) {
  const qc = useQueryClient()

  const vocabQuery = useQuery({
    queryKey: ["image-studio-captions-vocab", datasetPath],
    queryFn: () =>
      imageStudioCaptionsVocab(datasetPath, { recursive: true, limit: 200 }),
    enabled: Boolean(datasetPath),
  })

  // Selection state for the "blacklist selected tags" workflow.
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set())
  const toggleTag = (tag: string) =>
    setSelectedTags((prev) => {
      const next = new Set(prev)
      if (next.has(tag)) next.delete(tag)
      else next.add(tag)
      return next
    })
  const clearTags = () => setSelectedTags(new Set())

  const onMutated = () => {
    qc.invalidateQueries({ queryKey: ["image-studio-captions-vocab", datasetPath] })
    qc.invalidateQueries({ queryKey: ["image-studio-audit-report", datasetPath] })
    qc.invalidateQueries({ queryKey: ["image-studio"] })
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 grid gap-4 lg:grid-cols-2">
        <VocabPanel
          vocab={vocabQuery.data?.vocab ?? []}
          loading={vocabQuery.isLoading}
          totalFiles={vocabQuery.data?.files_seen ?? 0}
          tagCount={vocabQuery.data?.tag_count ?? 0}
          selectedTags={selectedTags}
          onToggleTag={toggleTag}
          onClearSelection={clearTags}
          onRefresh={() => vocabQuery.refetch()}
          datasetPath={datasetPath}
          onMutated={onMutated}
        />
        <FindReplacePanel datasetPath={datasetPath} onMutated={onMutated} />
        <TriggerInjectPanel datasetPath={datasetPath} onMutated={onMutated} />
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Vocab panel
// --------------------------------------------------------------------------- //

function VocabPanel({
  vocab,
  loading,
  totalFiles,
  tagCount,
  selectedTags,
  onToggleTag,
  onClearSelection,
  onRefresh,
  datasetPath,
  onMutated,
}: {
  vocab: CaptionVocabRow[]
  loading: boolean
  totalFiles: number
  tagCount: number
  selectedTags: Set<string>
  onToggleTag: (tag: string) => void
  onClearSelection: () => void
  onRefresh: () => void
  datasetPath: string
  onMutated: () => void
}) {
  const blacklistMutation = useMutation({
    mutationFn: () =>
      imageStudioCaptionsBlacklist({
        dataset_path: datasetPath,
        tags: Array.from(selectedTags),
      }),
    onSuccess: (data) => {
      toast.success(
        `已删除 ${data.removed_count} 处 tag(${data.edited_count} 个文件)`,
        { description: `黑名单: ${data.blacklisted_tags.join(", ")}` },
      )
      onClearSelection()
      onMutated()
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("黑名单删除失败", { description: msg })
    },
  })

  const max = vocab[0]?.count ?? 1
  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col min-h-0 lg:row-span-2">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <TagIcon className="size-3.5" />
        <span className="text-xs font-medium">标签词表</span>
        <span className="text-[11px] text-muted-foreground tabular-nums">
          {tagCount} 种 · {totalFiles} 文件
        </span>
        <div className="ml-auto flex items-center gap-1">
          <Button
            size="sm"
            variant="ghost"
            className="h-6 px-2 text-[11px]"
            onClick={onRefresh}
            disabled={loading}
          >
            {loading ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <RefreshCw className="size-3" />
            )}
          </Button>
        </div>
      </div>
      {selectedTags.size > 0 && (
        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border/40 bg-amber-50/50 dark:bg-amber-950/20 text-[11px]">
          <span>已选中 {selectedTags.size} 个 tag</span>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 px-2 text-[11px]"
            onClick={onClearSelection}
          >
            清空
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-6 px-2 text-[11px] gap-1 ml-auto hover:text-red-600"
            disabled={blacklistMutation.isPending}
            onClick={() => {
              if (
                !window.confirm(
                  `确认从全部 caption 中删除这 ${selectedTags.size} 个 tag?\n` +
                    `(原文件备份到 .workbench/backups/, 可恢复)`,
                )
              )
                return
              blacklistMutation.mutate()
            }}
          >
            {blacklistMutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Trash2 className="size-3" />
            )}
            黑名单删除
          </Button>
        </div>
      )}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center h-32 text-muted-foreground">
            <Loader2 className="size-4 animate-spin mr-2" />
            扫描标签...
          </div>
        )}
        {!loading && vocab.length === 0 && (
          <div className="flex items-center justify-center h-32 text-muted-foreground text-xs">
            未找到任何 caption.txt 标签
          </div>
        )}
        {!loading && vocab.length > 0 && (
          <ul className="divide-y divide-border/30">
            {vocab.map((row) => {
              const pct = (row.count / max) * 100
              const checked = selectedTags.has(row.tag)
              return (
                <li
                  key={row.tag}
                  className={cn(
                    "relative flex items-center gap-2 px-3 py-1 text-[11px] cursor-pointer hover:bg-muted/40",
                    checked && "bg-amber-50/40 dark:bg-amber-950/15",
                  )}
                  onClick={() => onToggleTag(row.tag)}
                >
                  <span
                    aria-hidden
                    className="absolute inset-y-0 left-0 bg-primary/10"
                    style={{ width: `${pct}%` }}
                  />
                  <input
                    type="checkbox"
                    className="relative size-3"
                    checked={checked}
                    onChange={() => onToggleTag(row.tag)}
                    onClick={(e) => e.stopPropagation()}
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
    </section>
  )
}

// --------------------------------------------------------------------------- //
// Find-replace panel
// --------------------------------------------------------------------------- //

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
    onSuccess: (data) => {
      setDryRunResult(data)
      if (data.matched_count === 0) {
        toast.info("未匹配到任何 caption")
      }
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("预演失败", { description: msg })
    },
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
      toast.success(`已应用替换: ${data.matched_count} 处 / ${data.matched_files} 文件`)
      setDryRunResult(null)
      onMutated()
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("应用失败", { description: msg })
    },
  })

  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <Replace className="size-3.5" />
        <span className="text-xs font-medium">查找替换</span>
      </div>
      <div className="p-3 space-y-2 text-xs">
        <Input
          placeholder="查找(默认整个 tag 精确匹配)"
          value={pattern}
          onChange={(e) => setPattern(e.target.value)}
          className="h-8 text-xs font-mono"
        />
        <Input
          placeholder="替换(留空 = 删除该 tag)"
          value={replacement}
          onChange={(e) => setReplacement(e.target.value)}
          className="h-8 text-xs font-mono"
        />
        <div className="flex items-center gap-3 flex-wrap">
          <label className="inline-flex items-center gap-1.5 select-none">
            <Switch checked={isRegex} onCheckedChange={setIsRegex} />
            正则
          </label>
          <label className="inline-flex items-center gap-1.5 select-none">
            <Switch checked={caseSensitive} onCheckedChange={setCaseSensitive} />
            大小写敏感
          </label>
          <label className="inline-flex items-center gap-1.5 select-none">
            <Switch checked={wholeCaption} onCheckedChange={setWholeCaption} />
            按整 caption 匹配
          </label>
        </div>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            disabled={!pattern || dryRunMutation.isPending}
            onClick={() => dryRunMutation.mutate()}
            className="h-7 gap-1"
          >
            {dryRunMutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : null}
            预演
          </Button>
          <Button
            size="sm"
            disabled={
              !dryRunResult ||
              dryRunResult.matched_count === 0 ||
              applyMutation.isPending
            }
            onClick={() => {
              if (!dryRunResult) return
              if (
                !window.confirm(
                  `确认应用替换 ${dryRunResult.matched_count} 处?(${dryRunResult.matched_files} 个文件)`,
                )
              )
                return
              applyMutation.mutate()
            }}
            className="h-7 gap-1"
          >
            {applyMutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : null}
            应用
          </Button>
          {dryRunResult && (
            <span className="text-[11px] text-muted-foreground">
              匹配 {dryRunResult.matched_count} 处 / {dryRunResult.matched_files}{" "}
              文件
            </span>
          )}
        </div>
      </div>
      {dryRunResult && dryRunResult.diffs.length > 0 && (
        <div className="border-t border-border/40 max-h-64 overflow-y-auto">
          {dryRunResult.diffs.slice(0, 100).map((d, i) => (
            <div
              key={i}
              className="px-3 py-1.5 text-[11px] border-t first:border-t-0 border-border/30"
            >
              <div className="font-mono truncate text-muted-foreground" title={d.path}>
                {d.path.split(/[\\/]/).pop()}
              </div>
              <div className="text-red-600/80 line-through truncate font-mono" title={d.before}>
                {d.before}
              </div>
              <div className="text-emerald-700/80 truncate font-mono" title={d.after}>
                {d.after}
              </div>
            </div>
          ))}
          {dryRunResult.diffs_truncated && (
            <div className="px-3 py-1 text-[10px] text-muted-foreground border-t border-border/30">
              ...(diff 仅展示前 100 项,实际匹配更多)
            </div>
          )}
        </div>
      )}
    </section>
  )
}

// --------------------------------------------------------------------------- //
// Trigger inject panel
// --------------------------------------------------------------------------- //

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
      toast.success(
        `已注入触发词到 ${data.injected_count} 个 caption`,
        { description: data.skipped_count ? `跳过 ${data.skipped_count}` : undefined },
      )
      onMutated()
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("注入失败", { description: msg })
    },
  })

  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <Wand2 className="size-3.5" />
        <span className="text-xs font-medium">触发词注入</span>
      </div>
      <div className="p-3 space-y-2 text-xs">
        <Input
          placeholder="触发词,如 @charA"
          value={trigger}
          onChange={(e) => setTrigger(e.target.value)}
          className="h-8 text-xs font-mono"
        />
        <div className="flex items-center gap-3 flex-wrap">
          <label className="inline-flex items-center gap-1.5 select-none">
            <input
              type="radio"
              name="trigger-pos"
              checked={position === "prepend"}
              onChange={() => setPosition("prepend")}
            />
            前置
          </label>
          <label className="inline-flex items-center gap-1.5 select-none">
            <input
              type="radio"
              name="trigger-pos"
              checked={position === "append"}
              onChange={() => setPosition("append")}
            />
            后置
          </label>
          <label className="inline-flex items-center gap-1.5 select-none ml-auto">
            <Switch checked={skipExisting} onCheckedChange={setSkipExisting} />
            跳过已含触发词
          </label>
        </div>
        <Button
          size="sm"
          disabled={!trigger.trim() || mutation.isPending}
          onClick={() => mutation.mutate()}
          className="h-7 gap-1 w-full"
        >
          {mutation.isPending ? <Loader2 className="size-3 animate-spin" /> : null}
          注入
        </Button>
      </div>
    </section>
  )
}

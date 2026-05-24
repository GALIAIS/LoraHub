import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  CheckCircle2,
  CircleAlert,
  CircleX,
  Copy,
  Download,
  Loader2,
  RefreshCw,
  Rocket,
} from "lucide-react"
import { toast } from "sonner"
import {
  imageStudioShipExport,
  imageStudioShipLint,
  imageStudioShipSaveAs,
  type ShipLintReport,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

interface Props {
  datasetPath: string
}

export function ShipStage({ datasetPath }: Props) {
  const lintQuery = useQuery({
    queryKey: ["image-studio-ship-lint", datasetPath],
    queryFn: () => imageStudioShipLint(datasetPath),
    enabled: Boolean(datasetPath),
    staleTime: 0,
  })

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <LintCard
          report={lintQuery.data}
          loading={lintQuery.isLoading}
          onRefresh={() => lintQuery.refetch()}
          datasetPath={datasetPath}
        />
        <div className="grid gap-4 lg:grid-cols-2">
          <ExportPanel datasetPath={datasetPath} />
          <SaveAsPanel datasetPath={datasetPath} />
        </div>
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Lint card
// --------------------------------------------------------------------------- //

function LintCard({
  report,
  loading,
  onRefresh,
  datasetPath,
}: {
  report: ShipLintReport | undefined
  loading: boolean
  onRefresh: () => void
  datasetPath: string
}) {
  const navigate = useNavigate()

  if (loading) {
    return (
      <section className="rounded-md border border-border/60 bg-card p-4 flex items-center gap-2 text-muted-foreground text-sm">
        <Loader2 className="size-4 animate-spin" />
        正在评估训练就绪状态...
      </section>
    )
  }
  if (!report) {
    return (
      <section className="rounded-md border border-border/60 bg-card p-4 text-sm text-muted-foreground">
        无法加载就绪报告。
      </section>
    )
  }

  const { state, label, tone, icon: Icon } = (() => {
    if (report.stale) {
      return {
        state: "stale" as const,
        label: "审计已过期",
        tone: "amber",
        icon: CircleAlert,
      }
    }
    if (report.blockers > 0) {
      return {
        state: "block" as const,
        label: "存在阻塞问题",
        tone: "red",
        icon: CircleX,
      }
    }
    if (report.warnings > 0 && report.ready) {
      return {
        state: "warn" as const,
        label: "可训练（有警告）",
        tone: "amber",
        icon: CircleAlert,
      }
    }
    if (report.ready) {
      return {
        state: "ready" as const,
        label: "已就绪",
        tone: "emerald",
        icon: CheckCircle2,
      }
    }
    return {
      state: "block" as const,
      label: "无法训练",
      tone: "red",
      icon: CircleX,
    }
  })()

  const toneRing =
    tone === "emerald"
      ? "border-emerald-600/50 bg-emerald-50 dark:bg-emerald-950/20"
      : tone === "amber"
        ? "border-amber-600/50 bg-amber-50 dark:bg-amber-950/20"
        : "border-red-600/50 bg-red-50 dark:bg-red-950/20"
  const toneText =
    tone === "emerald"
      ? "text-emerald-700 dark:text-emerald-400"
      : tone === "amber"
        ? "text-amber-700 dark:text-amber-400"
        : "text-red-700 dark:text-red-400"

  const goToScan = () => {
    const url = new URL(window.location.href)
    url.searchParams.set("stage", "audit")
    navigate(url.pathname + url.search)
  }
  const goToCurate = () => {
    const url = new URL(window.location.href)
    url.searchParams.set("stage", "curate")
    navigate(url.pathname + url.search)
  }
  const goToAnnotate = () => {
    const url = new URL(window.location.href)
    url.searchParams.set("stage", "annotate")
    navigate(url.pathname + url.search)
  }
  const launchTraining = () => {
    // Hand off to the configs page with the dataset path pre-filled.
    // Configs page reads `?dataset=` to seed the new-recipe flow.
    navigate(`/configs?dataset=${encodeURIComponent(datasetPath)}`)
  }

  return (
    <section className={cn("rounded-md border p-4", toneRing)}>
      <div className="flex items-center gap-2">
        <Icon className={cn("size-5", toneText)} />
        <span className={cn("text-base font-medium", toneText)}>{label}</span>
        {report.stale && report.stale_reason && (
          <span className="text-xs text-muted-foreground">
            ({report.stale_reason})
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-[11px] gap-1"
            onClick={onRefresh}
          >
            <RefreshCw className="size-3" />
            重新检查
          </Button>
          {state === "stale" && (
            <Button
              size="sm"
              onClick={goToScan}
              className="h-7 px-3 text-[11px] gap-1"
            >
              去 审计 重新扫描
            </Button>
          )}
          {report.ready && (
            <Button
              size="sm"
              onClick={launchTraining}
              className="h-7 px-3 text-[11px] gap-1"
            >
              <Rocket className="size-3" />
              新建训练任务
            </Button>
          )}
        </div>
      </div>

      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-sm">
        <Stat label="总图数" value={report.image_count} />
        <Stat
          label="已 caption"
          value={`${report.captioned_count} / ${report.image_count}`}
        />
        {report.trigger_word && (
          <Stat
            label={`触发词 ${report.trigger_word}`}
            value={`${report.trigger_word_hits} / ${report.image_count}`}
          />
        )}
        <Stat label="阻塞 / 警告" value={`${report.blockers} / ${report.warnings}`} />
      </dl>

      {report.issues.length > 0 && (
        <ul className="mt-3 space-y-1.5">
          {report.issues.map((iss) => {
            const isBlocker = iss.severity === "block"
            return (
              <li
                key={iss.code}
                className="flex items-start gap-2 text-xs"
              >
                {isBlocker ? (
                  <CircleX className="size-3.5 shrink-0 mt-0.5 text-red-600" />
                ) : (
                  <AlertTriangle className="size-3.5 shrink-0 mt-0.5 text-amber-600" />
                )}
                <span className={cn(isBlocker && "text-red-700 dark:text-red-400")}>
                  <strong>{iss.count}</strong> {iss.message}
                </span>
                {/* Quick jump-to-fix shortcuts. */}
                {(iss.code === "no_caption" ||
                  iss.code === "missing_trigger") && (
                  <button
                    type="button"
                    className="ml-auto text-[11px] underline text-muted-foreground hover:text-foreground"
                    onClick={goToAnnotate}
                  >
                    去 标注
                  </button>
                )}
                {(iss.code === "tiny" ||
                  iss.code === "blurry" ||
                  iss.code === "exif_rotation" ||
                  iss.code === "corrupt") && (
                  <button
                    type="button"
                    className="ml-auto text-[11px] underline text-muted-foreground hover:text-foreground"
                    onClick={goToScan}
                  >
                    去 审计 处理
                  </button>
                )}
                {iss.code === "empty" && (
                  <button
                    type="button"
                    className="ml-auto text-[11px] underline text-muted-foreground hover:text-foreground"
                    onClick={goToCurate}
                  >
                    去 整理 添加
                  </button>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div>
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className="text-base font-medium tabular-nums">{value}</dd>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Export panel
// --------------------------------------------------------------------------- //

function ExportPanel({ datasetPath }: { datasetPath: string }) {
  const [includeBackups, setIncludeBackups] = useState(false)
  const [includeQuarantine, setIncludeQuarantine] = useState(false)

  const exportMutation = useMutation({
    mutationFn: async () => {
      const resp = await imageStudioShipExport({
        dataset_path: datasetPath,
        include_backups: includeBackups,
        include_quarantine: includeQuarantine,
      })
      const blob = await resp.blob()
      // Trigger browser download.
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${datasetPath.split(/[\\/]/).pop()}.zip`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      return blob.size
    },
    onSuccess: (size) => {
      toast.success(
        `已导出 ${(size / 1024 / 1024).toFixed(1)} MB`,
        { description: "浏览器已开始下载 zip" },
      )
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("导出失败", { description: msg })
    },
  })

  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <Download className="size-3.5" />
        <span className="text-xs font-medium">导出 zip</span>
      </div>
      <div className="p-3 space-y-2 text-xs">
        <p className="text-muted-foreground">
          打包数据集为 zip，默认排除 <code>.workbench/</code>（隔离区 / 备份）。
        </p>
        <label className="inline-flex items-center gap-1.5 select-none">
          <Switch checked={includeBackups} onCheckedChange={setIncludeBackups} />
          包含 backups/
        </label>
        <label className="inline-flex items-center gap-1.5 select-none">
          <Switch
            checked={includeQuarantine}
            onCheckedChange={setIncludeQuarantine}
          />
          包含 quarantine/
        </label>
        <Button
          size="sm"
          onClick={() => exportMutation.mutate()}
          disabled={exportMutation.isPending}
          className="w-full gap-1 mt-2"
        >
          {exportMutation.isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Download className="size-3" />
          )}
          导出 zip
        </Button>
      </div>
    </section>
  )
}

// --------------------------------------------------------------------------- //
// Save-as panel
// --------------------------------------------------------------------------- //

function SaveAsPanel({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [newName, setNewName] = useState("")
  const [includeBackups, setIncludeBackups] = useState(false)

  const saveMutation = useMutation({
    mutationFn: () =>
      imageStudioShipSaveAs({
        source_path: datasetPath,
        new_name: newName.trim(),
        include_backups: includeBackups,
      }),
    onSuccess: (data) => {
      toast.success(
        `已另存为「${data.meta.name as string}」`,
        { description: `${data.images_copied} 张图 · ${data.path}` },
      )
      qc.invalidateQueries({ queryKey: ["image-studio-datasets"] })
      setNewName("")
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("另存失败", { description: msg })
    },
  })

  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <Copy className="size-3.5" />
        <span className="text-xs font-medium">另存为新数据集</span>
      </div>
      <div className="p-3 space-y-2 text-xs">
        <p className="text-muted-foreground">
          复制当前数据集到 <code>datasets/&lt;新名&gt;/</code>。
          实验性变体常用 · 改一份不影响原数据集。
        </p>
        <Input
          placeholder="新数据集名（只能字母数字+下划线/连字符）"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          className="h-8 text-xs"
        />
        <label className="inline-flex items-center gap-1.5 select-none">
          <Switch checked={includeBackups} onCheckedChange={setIncludeBackups} />
          包含 backups/（用于回滚链）
        </label>
        <Button
          size="sm"
          onClick={() => saveMutation.mutate()}
          disabled={!newName.trim() || saveMutation.isPending}
          className="w-full gap-1 mt-2"
        >
          {saveMutation.isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Copy className="size-3" />
          )}
          复制
        </Button>
      </div>
    </section>
  )
}

import { useState, useEffect } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Download, Pencil, Play } from "lucide-react"
import { api, ApiError, type ConfigListEntry } from "@/lib/api"
import { toastApiError } from "@/lib/toast-api-error"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ConfigForm, type ConfigFormValue } from "@/components/config-form"
import { applyOverrides, emptyOverrides, extractOverrides } from "../utils"
import type { LaunchOverrides } from "../types"
import { ErrorBanner } from "./error-banner"
import { LaunchOverrideDialog } from "./launch-override-dialog"
import { RawConfigFallback } from "./raw-config-fallback"

export function ConfigPreview({
  name,
  entry,
  onEdit,
  pendingDataset,
  autoOpenLaunch,
  onLaunchHandled,
}: {
  name: string
  entry: ConfigListEntry | null
  onEdit: () => void
  pendingDataset: string | null
  autoOpenLaunch: boolean
  onLaunchHandled: () => void
}) {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const detail = useQuery({
    queryKey: ["config", name],
    queryFn: () => api.getConfig(name),
  })

  const [dialogOpen, setDialogOpen] = useState(false)
  const [overrides, setOverrides] = useState<LaunchOverrides>(emptyOverrides())
  const [launchError, setLaunchError] = useState<string | null>(null)

  const data = detail.data
  const canLaunch = !!data?.parsed && !data.error
  const errorMsg = data?.error ?? entry?.error ?? null

  // Whenever the dialog opens (or the underlying config changes), seed the
  // form with the config's own defaults so the inputs read like placeholders
  // until the user actually changes them.
  useEffect(() => {
    if (!dialogOpen) return
    setOverrides(extractOverrides(data?.parsed ?? null, pendingDataset))
    setLaunchError(null)
  }, [dialogOpen, data?.parsed, pendingDataset])

  // Auto-open the dialog when the user navigated in from the Datasets page
  // with an override. Wait for the config detail to load so the dialog has
  // real defaults to display, otherwise opening would flash empty fields.
  useEffect(() => {
    if (!autoOpenLaunch) return
    if (!data?.parsed) return
    setDialogOpen(true)
    onLaunchHandled()
  }, [autoOpenLaunch, data?.parsed, onLaunchHandled])

  const launch = useMutation({
    mutationFn: () => {
      if (!data?.parsed) throw new Error("config is not valid")
      const merged = applyOverrides(data.parsed, overrides)
      return api.createJob(merged, undefined)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] })
      setDialogOpen(false)
      navigate("/jobs")
    },
    onError: (err) => {
      // For preflight 422s, render the structured findings as a toast
      // *and* keep the inline banner so the user can scroll back to the
      // root cause after dismissing the toast. Other errors only get
      // the banner — no point doubling generic "fetch failed" noise.
      if (err instanceof ApiError && err.preflightFindings) {
        toastApiError(err, { title: "训练前自检未通过" })
      }
      setLaunchError(err instanceof Error ? err.message : String(err))
    },
  })

  return (
    <div className="flex flex-col min-h-0 h-full">
      <header className="px-7 py-5 border-b border-border/60 flex items-start gap-4 shrink-0">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            {entry?.arch && (
              <Badge variant="outline" className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]">
                {entry.arch}
              </Badge>
            )}
            <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
              {entry?.filename ?? `${name}.yaml`}
            </span>
          </div>
          <div className="text-base font-semibold tracking-tight font-mono truncate">{name}</div>
          {entry?.valid && entry.summary && (
            <div className="text-xs text-muted-foreground mt-1">{entry.summary}</div>
          )}
        </div>
        <Button size="sm" variant="outline" onClick={onEdit}>
          <Pencil className="size-3" /> 编辑
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            const content = data?.content
            if (!content) return
            const blob = new Blob([content], { type: "text/yaml" })
            const url = URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.href = url
            a.download = `${name}.yaml`
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)
          }}
          disabled={!data?.content}
        >
          <Download className="size-3" /> 导出
        </Button>
        <Button
          size="sm"
          disabled={!canLaunch || launch.isPending}
          onClick={() => setDialogOpen(true)}
        >
          <Play className="size-3" />
          {launch.isPending ? "启动中…" : "训练"}
        </Button>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="px-4 py-4 space-y-3">
          {errorMsg && <ErrorBanner title="配置错误" message={errorMsg} />}

          {detail.isLoading ? (
            <div className="text-sm text-muted-foreground px-2 py-6">加载中…</div>
          ) : data?.parsed ? (
            <ConfigForm
              value={data.parsed as unknown as ConfigFormValue}
              onChange={() => {
                /* read-only — the form is rendered as a structured overview */
              }}
              readOnly
            />
          ) : data?.content ? (
            <RawConfigFallback content={data.content} />
          ) : (
            <div className="text-sm text-muted-foreground px-2 py-6">配置无法解析。</div>
          )}
        </div>
      </div>

      <LaunchOverrideDialog
        open={dialogOpen}
        onOpenChange={(next) => {
          setDialogOpen(next)
          if (!next) setLaunchError(null)
        }}
        configName={name}
        overrides={overrides}
        setOverrides={setOverrides}
        defaults={extractOverrides(data?.parsed ?? null, null)}
        launching={launch.isPending}
        errorMessage={launchError}
        onSubmit={() => launch.mutate()}
      />
    </div>
  )
}

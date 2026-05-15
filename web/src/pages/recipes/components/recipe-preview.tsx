import { useState, useEffect } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Pencil, Play } from "lucide-react"
import { api, type RecipeListEntry } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { applyOverrides, emptyOverrides, extractOverrides } from "../utils"
import type { LaunchOverrides } from "../types"
import { ErrorBanner } from "./error-banner"
import { LaunchOverrideDialog } from "./launch-override-dialog"

export function RecipePreview({
  name,
  entry,
  onEdit,
  pendingDataset,
  autoOpenLaunch,
  onLaunchHandled,
}: {
  name: string
  entry: RecipeListEntry | null
  onEdit: () => void
  pendingDataset: string | null
  autoOpenLaunch: boolean
  onLaunchHandled: () => void
}) {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const detail = useQuery({
    queryKey: ["recipe", name],
    queryFn: () => api.getRecipe(name),
  })

  const [dialogOpen, setDialogOpen] = useState(false)
  const [overrides, setOverrides] = useState<LaunchOverrides>(emptyOverrides())
  const [launchError, setLaunchError] = useState<string | null>(null)

  const data = detail.data
  const canLaunch = !!data?.parsed && !data.error
  const errorMsg = data?.error ?? entry?.error ?? null

  // Whenever the dialog opens (or the underlying recipe changes), seed the
  // form with the recipe's own defaults so the inputs read like placeholders
  // until the user actually changes them.
  useEffect(() => {
    if (!dialogOpen) return
    setOverrides(extractOverrides(data?.parsed ?? null, pendingDataset))
    setLaunchError(null)
  }, [dialogOpen, data?.parsed, pendingDataset])

  // Auto-open the dialog when the user navigated in from the Datasets page
  // with an override. Wait for the recipe detail to load so the dialog has
  // real defaults to display, otherwise opening would flash empty fields.
  useEffect(() => {
    if (!autoOpenLaunch) return
    if (!data?.parsed) return
    setDialogOpen(true)
    onLaunchHandled()
  }, [autoOpenLaunch, data?.parsed, onLaunchHandled])

  const launch = useMutation({
    mutationFn: () => {
      if (!data?.parsed) throw new Error("recipe is not valid")
      const merged = applyOverrides(data.parsed, overrides)
      return api.createJob(merged, undefined)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] })
      setDialogOpen(false)
      navigate("/jobs")
    },
    onError: (err) => {
      setLaunchError(err instanceof Error ? err.message : String(err))
    },
  })

  return (
    <div className="flex flex-col min-h-0 h-full">
      <header className="px-7 py-5 border-b border-border/60 flex items-start gap-4">
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
          disabled={!canLaunch || launch.isPending}
          onClick={() => setDialogOpen(true)}
        >
          <Play className="size-3" />
          {launch.isPending ? "启动中…" : "训练"}
        </Button>
      </header>

      {errorMsg && (
        <ErrorBanner title="配方错误" message={errorMsg} />
      )}

      <Card className="m-4 mb-0 rounded-[6px] border-border/60 shadow-[var(--panel-shadow)] overflow-hidden flex-1 min-h-0 flex flex-col">
        <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40">
          <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
            recipe.yaml
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0 flex-1 min-h-0">
          <ScrollArea className="h-full">
            <pre className="font-mono text-[12px] leading-relaxed px-4 py-3 whitespace-pre">
              {detail.isLoading ? "加载中…" : data?.content ?? ""}
            </pre>
          </ScrollArea>
        </CardContent>
      </Card>

      <LaunchOverrideDialog
        open={dialogOpen}
        onOpenChange={(next) => {
          setDialogOpen(next)
          if (!next) setLaunchError(null)
        }}
        recipeName={name}
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

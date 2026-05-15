import { useState, useEffect, useRef } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useLocation, useNavigate } from "react-router-dom"
import {
  Play,
  FileWarning,
  FileCheck2,
  Folder,
  Pencil,
  Plus,
  Save,
  CheckCheck,
  XCircle,
  ArrowLeft,
  AlertTriangle,
  Gauge,
} from "lucide-react"
import {
  api,
  type RecipeListEntry,
  type ValidationFieldError,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { RecipeForm, type RecipeFormValue } from "@/components/recipe-form"
import { cn } from "@/lib/utils"

type Mode = { kind: "preview"; name: string } | { kind: "edit"; name: string } | { kind: "new" }

type LaunchOverrides = {
  datasetSource: string
  outputName: string
  batchSize: string
  epochs: string
  maxSteps: string
}

type LocationState = {
  overrideDataset?: string
} | null

export function RecipesPage() {
  const list = useQuery({ queryKey: ["recipes"], queryFn: api.listRecipes })
  const recipes = list.data?.recipes ?? []

  const [mode, setMode] = useState<Mode | null>(null)
  // Pre-populated dataset path that flows in from the Datasets page via
  // router state. Once the launch dialog opens we hand it off and clear.
  const [pendingDataset, setPendingDataset] = useState<string | null>(null)
  const [autoOpenLaunch, setAutoOpenLaunch] = useState(false)

  const location = useLocation()
  const navigate = useNavigate()
  const consumedNavStateRef = useRef(false)

  // Pick up overrideDataset handed in from the Datasets page exactly once,
  // then strip it from history so a refresh does not reopen the dialog.
  useEffect(() => {
    if (consumedNavStateRef.current) return
    const navState = location.state as LocationState
    const override = navState?.overrideDataset
    if (typeof override === "string" && override.trim().length > 0) {
      consumedNavStateRef.current = true
      setPendingDataset(override)
      setAutoOpenLaunch(true)
      navigate(location.pathname, { replace: true, state: null })
    }
  }, [location, navigate])

  // Default selection: first recipe in preview mode. When we arrived with a
  // pending dataset override, force-select the first recipe even if a mode
  // was already chosen, so the dialog opens against a real recipe.
  useEffect(() => {
    if (recipes.length === 0) return
    if (mode === null) {
      setMode({ kind: "preview", name: recipes[0].name })
    } else if (autoOpenLaunch && mode.kind !== "preview") {
      setMode({ kind: "preview", name: recipes[0].name })
    }
  }, [mode, recipes, autoOpenLaunch])

  return (
    <div className="grid grid-cols-[minmax(320px,380px)_1fr] h-full">
      <aside className="border-r border-border/60 flex flex-col min-h-0">
        <header className="px-5 py-4 border-b border-border/60 flex items-start gap-2">
          <div className="flex-1 min-w-0">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
              Recipes
            </div>
            <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-1.5">
              <Folder className="size-3" />
              <span className="font-mono truncate" title={list.data?.dir ?? ""}>
                {list.data?.dir ? shortenPath(list.data.dir) : "loading…"}
              </span>
            </div>
          </div>
          <Button size="sm" variant="outline" onClick={() => setMode({ kind: "new" })}>
            <Plus className="size-3" /> New
          </Button>
        </header>
        <ScrollArea className="flex-1">
          <ul className="divide-y divide-border/40">
            {list.isLoading && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                Loading…
              </li>
            )}
            {!list.isLoading && recipes.length === 0 && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                No recipes yet. Click <span className="text-foreground font-medium">+ New</span> to
                create one, or run <code className="text-foreground">lorahub init</code>.
              </li>
            )}
            {recipes.map((r) => {
              const active =
                (mode?.kind === "preview" || mode?.kind === "edit") && mode.name === r.name
              return (
                <RecipeRow
                  key={r.name}
                  recipe={r}
                  active={active}
                  onSelect={() => setMode({ kind: "preview", name: r.name })}
                />
              )
            })}
          </ul>
        </ScrollArea>
      </aside>

      <section className="min-w-0 flex flex-col bg-background/60">
        {mode === null ? (
          <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
            Select a recipe to preview, or create a new one.
          </div>
        ) : mode.kind === "preview" ? (
          <RecipePreview
            name={mode.name}
            entry={recipes.find((r) => r.name === mode.name) ?? null}
            onEdit={() => setMode({ kind: "edit", name: mode.name })}
            pendingDataset={pendingDataset}
            autoOpenLaunch={autoOpenLaunch}
            onLaunchHandled={() => {
              setPendingDataset(null)
              setAutoOpenLaunch(false)
            }}
          />
        ) : (
          <RecipeEditor mode={mode} setMode={setMode} />
        )}
      </section>
    </div>
  )
}

function RecipeRow({
  recipe,
  active,
  onSelect,
}: {
  recipe: RecipeListEntry
  active: boolean
  onSelect: () => void
}) {
  return (
    <li
      onClick={onSelect}
      className={cn(
        "px-5 py-3 cursor-pointer transition-colors",
        active
          ? "bg-accent/70 border-l-2 border-l-primary"
          : "border-l-2 border-l-transparent hover:bg-muted/40",
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        {recipe.valid ? (
          <FileCheck2 className="size-3.5 text-emerald-600 dark:text-emerald-400" />
        ) : (
          <FileWarning className="size-3.5 text-destructive" />
        )}
        <span className="text-sm font-medium truncate">{recipe.name}</span>
        {recipe.arch && (
          <Badge variant="outline" className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]">
            {recipe.arch}
          </Badge>
        )}
      </div>
      <div className="text-xs text-muted-foreground truncate">
        {recipe.valid ? recipe.summary : recipe.error}
      </div>
    </li>
  )
}

function RecipePreview({
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
          <Pencil className="size-3" /> Edit
        </Button>
        <Button
          size="sm"
          disabled={!canLaunch || launch.isPending}
          onClick={() => setDialogOpen(true)}
        >
          <Play className="size-3" />
          {launch.isPending ? "Launching…" : "Train"}
        </Button>
      </header>

      {errorMsg && (
        <ErrorBanner title="Recipe error" message={errorMsg} />
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
              {detail.isLoading ? "Loading…" : data?.content ?? ""}
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

function LaunchOverrideDialog({
  open,
  onOpenChange,
  recipeName,
  overrides,
  setOverrides,
  defaults,
  launching,
  errorMessage,
  onSubmit,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  recipeName: string
  overrides: LaunchOverrides
  setOverrides: (next: LaunchOverrides) => void
  defaults: LaunchOverrides
  launching: boolean
  errorMessage: string | null
  onSubmit: () => void
}) {
  const update = <K extends keyof LaunchOverrides>(key: K, value: LaunchOverrides[K]) => {
    setOverrides({ ...overrides, [key]: value })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[min(calc(100%-2rem),34rem)]">
        <DialogHeader>
          <DialogTitle>Launch override</DialogTitle>
          <DialogDescription>
            Tweak any field for this run only. Empty fields fall back to the recipe value.
            The recipe file on disk is not touched.
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 gap-3">
          <OverrideField
            label="dataset.source"
            placeholder={defaults.datasetSource || "./datasets/my_character"}
            value={overrides.datasetSource}
            onChange={(v) => update("datasetSource", v)}
            description="Image folder to train on."
          />
          <OverrideField
            label="output.name"
            placeholder={defaults.outputName || "my_character_v1"}
            value={overrides.outputName}
            onChange={(v) => update("outputName", v)}
            description="Filename stem for the saved LoRA."
          />
          <div className="grid grid-cols-3 gap-3">
            <OverrideField
              label="batch_size"
              placeholder={defaults.batchSize || "1"}
              value={overrides.batchSize}
              onChange={(v) => update("batchSize", v)}
              type="number"
              min={1}
            />
            <OverrideField
              label="epochs"
              placeholder={defaults.epochs || "10"}
              value={overrides.epochs}
              onChange={(v) => update("epochs", v)}
              type="number"
              min={1}
            />
            <OverrideField
              label="max_steps"
              placeholder={defaults.maxSteps || "(unset)"}
              value={overrides.maxSteps}
              onChange={(v) => update("maxSteps", v)}
              type="number"
              min={1}
            />
          </div>
        </div>

        {errorMessage && (
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
            {errorMessage}
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={launching}>
            Cancel
          </Button>
          <Button onClick={onSubmit} disabled={launching}>
            <Play className="size-3" />
            {launching ? "Launching…" : `Train ${recipeName}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function OverrideField({
  label,
  value,
  onChange,
  placeholder,
  description,
  type = "text",
  min,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  description?: string
  type?: "text" | "number"
  min?: number
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </Label>
      <Input
        type={type}
        value={value}
        placeholder={placeholder}
        min={min}
        onChange={(event) => onChange(event.target.value)}
        className="font-mono"
      />
      {description && (
        <span className="text-[11px] text-muted-foreground">{description}</span>
      )}
    </div>
  )
}

function RecipeEditor({
  mode,
  setMode,
}: {
  mode: { kind: "edit"; name: string } | { kind: "new" }
  setMode: (m: Mode) => void
}) {
  const isNew = mode.kind === "new"
  const navigate = useNavigate()
  const qc = useQueryClient()

  const sourceQuery = useQuery({
    queryKey: ["recipe", isNew ? "__new__" : mode.name],
    queryFn: () => (isNew ? Promise.resolve(null) : api.getRecipe(mode.name)),
  })

  const [draft, setDraft] = useState<RecipeFormValue | null>(null)
  const [name, setName] = useState<string>(isNew ? "" : mode.name)
  const [errors, setErrors] = useState<ValidationFieldError[]>([])

  // Seed the draft from the source recipe (or sane defaults for new).
  useEffect(() => {
    if (isNew) {
      setDraft(buildDefaults())
      setName("")
    } else if (sourceQuery.data?.parsed) {
      setDraft(sourceQuery.data.parsed as unknown as RecipeFormValue)
      setName(mode.name)
    }
  }, [isNew, sourceQuery.data, mode])

  const validate = useMutation({
    mutationFn: () =>
      api.validateRecipe((draft ?? {}) as unknown as Record<string, unknown>),
    onSuccess: (resp) => setErrors(resp.errors ?? []),
  })

  const save = useMutation({
    mutationFn: async (opts: { overwrite: boolean; thenLaunch: boolean }) => {
      if (!draft) throw new Error("no draft")
      const payload = draft as unknown as Record<string, unknown>
      const v = await api.validateRecipe(payload)
      if (!v.valid) {
        setErrors(v.errors ?? [])
        throw new Error("recipe has validation errors")
      }
      const cleanName = name.trim()
      if (!cleanName) throw new Error("name is required")
      const saved = await api.saveRecipe(cleanName, payload, opts.overwrite || !isNew)
      qc.invalidateQueries({ queryKey: ["recipes"] })
      qc.invalidateQueries({ queryKey: ["recipe", cleanName] })
      if (opts.thenLaunch) {
        const job = await api.createJob(payload)
        qc.invalidateQueries({ queryKey: ["jobs"] })
        navigate("/jobs")
        return { saved, job }
      }
      setMode({ kind: "preview", name: cleanName })
      return { saved }
    },
  })

  const loading = !isNew && sourceQuery.isLoading

  return (
    <div className="flex flex-col min-h-0 h-full">
      <header className="px-7 py-5 border-b border-border/60 flex items-start gap-3 shrink-0">
        <Button
          size="sm"
          variant="ghost"
          onClick={() =>
            setMode(
              isNew
                ? ({ kind: "preview", name: "" } as Mode)
                : { kind: "preview", name: mode.name },
            )
          }
        >
          <ArrowLeft className="size-3" /> Back
        </Button>
        <div className="flex-1 min-w-0">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-1">
            {isNew ? "New recipe" : "Edit recipe"}
          </div>
          {isNew ? (
            <Input
              value={name}
              placeholder="my_character"
              onChange={(e) => setName(e.target.value)}
              className="font-mono w-64 h-8"
            />
          ) : (
            <div className="text-base font-semibold tracking-tight font-mono truncate">
              {name}
            </div>
          )}
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => validate.mutate()}
          disabled={!draft || validate.isPending}
        >
          <CheckCheck className="size-3" /> Validate
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => save.mutate({ overwrite: !isNew, thenLaunch: false })}
          disabled={!draft || save.isPending}
        >
          <Save className="size-3" /> {save.isPending ? "Saving…" : "Save"}
        </Button>
        <Button
          size="sm"
          onClick={() => save.mutate({ overwrite: !isNew, thenLaunch: true })}
          disabled={!draft || save.isPending}
        >
          <Play className="size-3" /> Save & Train
        </Button>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="px-4 pb-4 pt-3 space-y-3">
          {validate.data?.valid === true && (
            <div className="rounded-[4px] border border-emerald-500/40 bg-emerald-500/5 px-4 py-2 text-xs text-emerald-700 dark:text-emerald-400">
              Recipe is valid.
            </div>
          )}
          {validate.data?.preflight && (
            <PreflightPanel preflight={validate.data.preflight} />
          )}
          {errors.length > 0 && (
            <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-destructive font-semibold flex items-center gap-1.5">
                <XCircle className="size-3" /> {errors.length} validation error(s)
              </div>
              <ul className="mt-2 text-xs font-mono text-destructive space-y-0.5">
                {errors.slice(0, 8).map((e, i) => (
                  <li key={i}>
                    <span className="text-muted-foreground">{e.loc.join(".")}</span>: {e.msg}
                  </li>
                ))}
                {errors.length > 8 && (
                  <li className="text-muted-foreground">…and {errors.length - 8} more</li>
                )}
              </ul>
            </div>
          )}
          {save.isError && (
            <ErrorBanner title="Save failed" message={(save.error as Error).message} />
          )}

          {loading || !draft ? (
            <div className="text-sm text-muted-foreground px-2 py-6">Loading…</div>
          ) : (
            <RecipeForm value={draft} onChange={setDraft} errors={errors} />
          )}
        </div>
      </div>
    </div>
  )
}

function PreflightPanel({
  preflight,
}: {
  preflight: NonNullable<Awaited<ReturnType<typeof api.validateRecipe>>["preflight"]>
}) {
  const warnings = preflight.issues.filter((issue) => issue.severity !== "info")
  const missingCaptions = preflight.paths.missing_caption_files

  return (
    <Card className="mx-4 mt-3 rounded-[6px] border-border/60 bg-card/80 shadow-[var(--panel-shadow)]">
      <CardContent className="p-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              Preflight
            </div>
            <div className="mt-1 text-sm font-medium">
              {warnings.length === 0 ? "Ready to launch" : `${warnings.length} item(s) need attention`}
            </div>
          </div>
          <div className="rounded-[4px] border border-border/70 px-3 py-2 text-right">
            <div className="flex items-center justify-end gap-1.5 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
              <Gauge className="size-3" /> VRAM estimate
            </div>
            <div className="font-mono text-lg leading-none mt-1">
              {preflight.vram.total_gib.toFixed(2)} GiB
            </div>
          </div>
        </div>

        <div className="mt-4 grid gap-2 md:grid-cols-3">
          <PreflightMetric
            label="Checkpoint"
            value={preflight.paths.checkpoint_exists ? "Found" : "Missing"}
            ok={preflight.paths.checkpoint_exists}
          />
          <PreflightMetric
            label="Dataset"
            value={`${preflight.paths.image_files} image(s)`}
            ok={preflight.paths.dataset_exists && preflight.paths.image_files > 0}
          />
          <PreflightMetric
            label="Captions"
            value={`${preflight.paths.caption_files}/${preflight.paths.image_files}`}
            ok={missingCaptions.length === 0}
          />
        </div>

        {warnings.length > 0 && (
          <ul className="mt-3 space-y-1.5 text-xs">
            {warnings.slice(0, 5).map((issue, i) => (
              <li key={i} className="flex items-start gap-2 text-muted-foreground">
                <AlertTriangle className="mt-0.5 size-3 text-amber-600 dark:text-amber-400" />
                <span>
                  <span className="font-mono text-foreground">{issue.field}</span>: {issue.message}
                </span>
              </li>
            ))}
          </ul>
        )}

        {missingCaptions.length > 0 && (
          <div className="mt-3 rounded-[4px] bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
            Missing captions:{" "}
            <span className="font-mono text-foreground">
              {missingCaptions.join(", ")}
              {preflight.paths.missing_caption_files_truncated ? ", ..." : ""}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function PreflightMetric({
  label,
  value,
  ok,
}: {
  label: string
  value: string
  ok: boolean
}) {
  return (
    <div className="rounded-[4px] border border-border/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className={cn("mt-1 text-sm font-medium", ok ? "text-emerald-600 dark:text-emerald-400" : "text-amber-700 dark:text-amber-400")}>
        {value}
      </div>
    </div>
  )
}

function ErrorBanner({ title, message }: { title: string; message: string }) {
  return (
    <div className="mx-4 mt-4 rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3">
      <div className="text-[10px] uppercase tracking-[0.18em] text-destructive font-semibold">
        {title}
      </div>
      <div className="mt-1 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
        {message}
      </div>
    </div>
  )
}

function buildDefaults(): RecipeFormValue {
  // Minimal valid skeleton — enough that the form renders with sensible
  // starting values; the user only has to fill in the two paths.
  return {
    schema_version: "1.0",
    base_model: { arch: "sdxl", checkpoint: "" },
    dataset: { source: "", resolution: [1024, 1024] },
  }
}

function shortenPath(p: string): string {
  const idx = p.toLowerCase().lastIndexOf("recipes")
  return idx >= 0 ? p.slice(idx) : p
}

function emptyOverrides(): LaunchOverrides {
  return {
    datasetSource: "",
    outputName: "",
    batchSize: "",
    epochs: "",
    maxSteps: "",
  }
}

/**
 * Pull current values out of a parsed recipe so the dialog can prefill the
 * inputs (or, when only used for placeholders, show what is currently in the
 * recipe). When `pendingDataset` is provided it wins over the recipe value
 * for `dataset.source` — that is how the Datasets page hands a freshly
 * scanned folder to the launch dialog.
 */
function extractOverrides(
  parsed: Record<string, unknown> | null,
  pendingDataset: string | null,
): LaunchOverrides {
  const dataset = (parsed?.dataset as Record<string, unknown> | undefined) ?? {}
  const output = (parsed?.output as Record<string, unknown> | undefined) ?? {}
  const schedule = (parsed?.schedule as Record<string, unknown> | undefined) ?? {}

  const datasetSource =
    pendingDataset && pendingDataset.trim().length > 0
      ? pendingDataset
      : asString(dataset.source)

  return {
    datasetSource,
    outputName: asString(output.name),
    batchSize: asString(schedule.batch_size),
    epochs: asString(schedule.epochs),
    maxSteps: asString(schedule.max_steps),
  }
}

function asString(value: unknown): string {
  if (value === null || value === undefined) return ""
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  return ""
}

/**
 * Deep clone the recipe and write back any non-empty overrides at the right
 * paths. Numeric fields are coerced; invalid input is silently dropped so the
 * backend still sees a valid value rather than NaN.
 */
function applyOverrides(
  recipe: Record<string, unknown>,
  overrides: LaunchOverrides,
): Record<string, unknown> {
  const cloned = structuredClone(recipe) as Record<string, unknown>
  const trim = (v: string) => v.trim()

  if (trim(overrides.datasetSource)) {
    setIn(cloned, ["dataset", "source"], trim(overrides.datasetSource))
  }
  if (trim(overrides.outputName)) {
    setIn(cloned, ["output", "name"], trim(overrides.outputName))
  }

  const batchSize = parsePositiveInt(overrides.batchSize)
  if (batchSize !== null) {
    setIn(cloned, ["schedule", "batch_size"], batchSize)
  }
  const epochs = parsePositiveInt(overrides.epochs)
  if (epochs !== null) {
    setIn(cloned, ["schedule", "epochs"], epochs)
  }
  const maxSteps = parsePositiveInt(overrides.maxSteps)
  if (maxSteps !== null) {
    setIn(cloned, ["schedule", "max_steps"], maxSteps)
  }

  return cloned
}

function parsePositiveInt(raw: string): number | null {
  const trimmed = raw.trim()
  if (trimmed === "") return null
  const num = Number(trimmed)
  if (!Number.isFinite(num)) return null
  const int = Math.trunc(num)
  if (int < 1) return null
  return int
}

/** Lodash-style setIn: walks/creates nested object keys and writes the leaf. */
function setIn(
  target: Record<string, unknown>,
  path: string[],
  value: unknown,
): void {
  let cursor: Record<string, unknown> = target
  for (let i = 0; i < path.length - 1; i++) {
    const key = path[i]
    const next = cursor[key]
    if (next === null || typeof next !== "object" || Array.isArray(next)) {
      const created: Record<string, unknown> = {}
      cursor[key] = created
      cursor = created
    } else {
      cursor = next as Record<string, unknown>
    }
  }
  cursor[path[path.length - 1]] = value
}

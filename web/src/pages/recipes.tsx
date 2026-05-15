import { useState, useEffect } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  SchemaForm,
  type RecipeSchema,
  type JsonValue,
} from "@/components/schema-form"
import { cn } from "@/lib/utils"

type Mode = { kind: "preview"; name: string } | { kind: "edit"; name: string } | { kind: "new" }

export function RecipesPage() {
  const list = useQuery({ queryKey: ["recipes"], queryFn: api.listRecipes })
  const recipes = list.data?.recipes ?? []

  const [mode, setMode] = useState<Mode | null>(null)

  // Default selection: first recipe in preview mode.
  useEffect(() => {
    if (mode === null && recipes.length > 0) {
      setMode({ kind: "preview", name: recipes[0].name })
    }
  }, [mode, recipes])

  return (
    <div className="grid grid-cols-[minmax(320px,380px)_1fr] h-screen">
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
}: {
  name: string
  entry: RecipeListEntry | null
  onEdit: () => void
}) {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const detail = useQuery({
    queryKey: ["recipe", name],
    queryFn: () => api.getRecipe(name),
  })

  const launch = useMutation({
    mutationFn: () => {
      if (!detail.data?.parsed) throw new Error("recipe is not valid")
      return api.createJob(detail.data.parsed)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] })
      navigate("/jobs")
    },
  })

  const data = detail.data
  const canLaunch = !!data?.parsed && !data.error
  const errorMsg = data?.error ?? entry?.error ?? null

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
        <Button size="sm" disabled={!canLaunch || launch.isPending} onClick={() => launch.mutate()}>
          <Play className="size-3" />
          {launch.isPending ? "Launching…" : "Train"}
        </Button>
      </header>

      {errorMsg && (
        <ErrorBanner title="Recipe error" message={errorMsg} />
      )}
      {launch.isError && (
        <ErrorBanner title="Launch failed" message={(launch.error as Error).message} />
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

  const schemaQuery = useQuery({
    queryKey: ["recipe-schema"],
    queryFn: api.recipeSchema,
    staleTime: Infinity,
  })

  const sourceQuery = useQuery({
    queryKey: ["recipe", isNew ? "__new__" : mode.name],
    queryFn: () => (isNew ? Promise.resolve(null) : api.getRecipe(mode.name)),
  })

  const [draft, setDraft] = useState<Record<string, JsonValue> | null>(null)
  const [name, setName] = useState<string>(isNew ? "" : mode.name)
  const [errors, setErrors] = useState<ValidationFieldError[]>([])

  // Seed the draft from the source recipe (or sane defaults for new).
  useEffect(() => {
    if (isNew) {
      setDraft(buildDefaults())
      setName("")
    } else if (sourceQuery.data?.parsed) {
      setDraft(sourceQuery.data.parsed as Record<string, JsonValue>)
      setName(mode.name)
    }
  }, [isNew, sourceQuery.data, mode])

  const validate = useMutation({
    mutationFn: () => api.validateRecipe(draft ?? {}),
    onSuccess: (resp) => setErrors(resp.errors ?? []),
  })

  const save = useMutation({
    mutationFn: async (opts: { overwrite: boolean; thenLaunch: boolean }) => {
      if (!draft) throw new Error("no draft")
      const v = await api.validateRecipe(draft)
      if (!v.valid) {
        setErrors(v.errors ?? [])
        throw new Error("recipe has validation errors")
      }
      const cleanName = name.trim()
      if (!cleanName) throw new Error("name is required")
      const saved = await api.saveRecipe(cleanName, draft, opts.overwrite || !isNew)
      qc.invalidateQueries({ queryKey: ["recipes"] })
      qc.invalidateQueries({ queryKey: ["recipe", cleanName] })
      if (opts.thenLaunch) {
        const job = await api.createJob(draft)
        qc.invalidateQueries({ queryKey: ["jobs"] })
        navigate("/jobs")
        return { saved, job }
      }
      setMode({ kind: "preview", name: cleanName })
      return { saved }
    },
  })

  const schema = schemaQuery.data as RecipeSchema | undefined
  const loading = (!isNew && sourceQuery.isLoading) || schemaQuery.isLoading

  return (
    <div className="flex flex-col min-h-0 h-full">
      <header className="px-7 py-5 border-b border-border/60 flex items-start gap-3">
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setMode(isNew ? ({ kind: "preview", name: "" } as Mode) : { kind: "preview", name: mode.name })}
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
            <div className="text-base font-semibold tracking-tight font-mono truncate">{name}</div>
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

      {validate.data?.valid === true && (
        <div className="mx-4 mt-4 rounded-[4px] border border-emerald-500/40 bg-emerald-500/5 px-4 py-2 text-xs text-emerald-700 dark:text-emerald-400">
          Recipe is valid.
        </div>
      )}
      {validate.data?.preflight && (
        <PreflightPanel preflight={validate.data.preflight} />
      )}
      {errors.length > 0 && (
        <div className="mx-4 mt-4 rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3">
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

      <div className="flex-1 min-h-0 px-4 pb-4 pt-4">
        <Tabs defaultValue="form" className="h-full flex flex-col">
          <TabsList className="self-start">
            <TabsTrigger value="form">Form</TabsTrigger>
            <TabsTrigger value="json">JSON</TabsTrigger>
          </TabsList>
          <TabsContent value="form" className="flex-1 min-h-0 mt-3">
            <Card className="h-full rounded-[6px] border-border/60 shadow-[var(--panel-shadow)] flex flex-col">
              <CardContent className="flex-1 min-h-0 p-0">
                <ScrollArea className="h-full">
                  <div className="px-5 py-5">
                    {loading || !schema || !draft ? (
                      <div className="text-sm text-muted-foreground">Loading…</div>
                    ) : (
                      <SchemaForm
                        schema={schema}
                        value={draft}
                        errors={errors}
                        onChange={setDraft}
                      />
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="json" className="flex-1 min-h-0 mt-3">
            <Card className="h-full rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
              <CardContent className="p-0 h-full">
                <textarea
                  value={draft ? JSON.stringify(draft, null, 2) : ""}
                  onChange={(e) => {
                    try {
                      setDraft(JSON.parse(e.target.value))
                    } catch {
                      // ignore parse errors mid-typing
                    }
                  }}
                  spellCheck={false}
                  className="w-full h-full font-mono text-xs p-4 bg-transparent resize-none outline-none"
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
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

function buildDefaults(): Record<string, JsonValue> {
  // Minimal valid skeleton — enough that the form renders with sensible
  // starting values; the user only has to fill in the two paths.
  return {
    schema_version: "1.0",
    base_model: { arch: "sdxl", checkpoint: "" },
    dataset: { source: "" },
  }
}

function shortenPath(p: string): string {
  const idx = p.toLowerCase().lastIndexOf("recipes")
  return idx >= 0 ? p.slice(idx) : p
}

import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Play, FileWarning, FileCheck2, Folder } from "lucide-react"
import { api, type RecipeListEntry } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export function RecipesPage() {
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const list = useQuery({ queryKey: ["recipes"], queryFn: api.listRecipes })
  const recipes = list.data?.recipes ?? []

  if (selectedName === null && recipes.length > 0) {
    setSelectedName(recipes[0].name)
  }

  const selected = recipes.find((r) => r.name === selectedName) ?? null

  return (
    <div className="grid grid-cols-[minmax(320px,380px)_1fr] h-screen">
      <aside className="border-r border-border/60 flex flex-col min-h-0">
        <header className="px-5 py-4 border-b border-border/60">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            Recipes
          </div>
          <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-1.5">
            <Folder className="size-3" />
            <span className="font-mono truncate" title={list.data?.dir ?? ""}>
              {list.data?.dir ? shortenPath(list.data.dir) : "loading…"}
            </span>
          </div>
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
                No recipes found. Drop a YAML file into{" "}
                <code className="text-foreground">recipes/</code> or run{" "}
                <code className="text-foreground">lorahub init</code>.
              </li>
            )}
            {recipes.map((r) => (
              <RecipeRow
                key={r.name}
                recipe={r}
                active={r.name === selectedName}
                onSelect={() => setSelectedName(r.name)}
              />
            ))}
          </ul>
        </ScrollArea>
      </aside>

      <section className="min-w-0 flex flex-col bg-background/60">
        {selected ? (
          <RecipeDetail name={selected.name} entry={selected} />
        ) : (
          <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
            Select a recipe to preview.
          </div>
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

function RecipeDetail({ name, entry }: { name: string; entry: RecipeListEntry }) {
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
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: ["jobs"] })
      navigate("/jobs")
      // brief log so the user sees what happened
      console.info("[lorahub] launched job", job.id)
    },
  })

  const data = detail.data
  const canLaunch = !!data?.parsed && !data.error
  const errorMsg = data?.error ?? entry.error

  return (
    <div className="flex flex-col min-h-0 h-full">
      <header className="px-7 py-5 border-b border-border/60 flex items-start gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            {entry.arch && (
              <Badge variant="outline" className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]">
                {entry.arch}
              </Badge>
            )}
            <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
              {entry.filename}
            </span>
          </div>
          <div className="text-base font-semibold tracking-tight font-mono truncate">
            {name}
          </div>
          {entry.valid && entry.summary && (
            <div className="text-xs text-muted-foreground mt-1">{entry.summary}</div>
          )}
        </div>
        <Button
          size="sm"
          disabled={!canLaunch || launch.isPending}
          onClick={() => launch.mutate()}
        >
          <Play className="size-3" />
          {launch.isPending ? "Launching…" : "Train"}
        </Button>
      </header>

      {errorMsg && (
        <div className="mx-4 mt-4 rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3">
          <div className="text-[10px] uppercase tracking-[0.18em] text-destructive font-semibold">
            Recipe error
          </div>
          <div className="mt-1 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
            {errorMsg}
          </div>
        </div>
      )}

      {launch.isError && (
        <div className="mx-4 mt-4 rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3 text-xs font-mono text-destructive">
          {(launch.error as Error).message}
        </div>
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

function shortenPath(p: string): string {
  // Trim everything before the last "recipes" segment so we just show "recipes/…"
  const idx = p.toLowerCase().lastIndexOf("recipes")
  return idx >= 0 ? p.slice(idx) : p
}

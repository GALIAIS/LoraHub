import { useEffect, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useLocation, useNavigate } from "react-router-dom"
import { Folder, Plus } from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { RecipeRow } from "./components/recipe-row"
import { RecipePreview } from "./components/recipe-preview"
import { RecipeEditor } from "./components/recipe-editor"
import type { Mode } from "./types"
import { shortenPath } from "./utils"

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
              训练配方
            </div>
            <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-1.5">
              <Folder className="size-3" />
              <span className="font-mono truncate" title={list.data?.dir ?? ""}>
                {list.data?.dir ? shortenPath(list.data.dir) : "加载中…"}
              </span>
            </div>
          </div>
          <Button size="sm" variant="outline" onClick={() => setMode({ kind: "new" })}>
            <Plus className="size-3" /> 新建
          </Button>
        </header>
        <ScrollArea className="flex-1">
          <ul className="divide-y divide-border/40">
            {list.isLoading && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                加载中…
              </li>
            )}
            {!list.isLoading && recipes.length === 0 && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                还没有配方。点击 <span className="text-foreground font-medium">+ 新建</span> 创建一个，或运行
                <code className="text-foreground"> lorahub init</code>。
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
            从左侧选择一个配方查看，或创建新的。
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

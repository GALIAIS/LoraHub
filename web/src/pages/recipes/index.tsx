import { useEffect, useMemo, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useLocation, useNavigate } from "react-router-dom"
import { api, type RecipeListEntry } from "@/lib/api"
import { ScrollArea } from "@/components/ui/scroll-area"
import { RecipeRow } from "./components/recipe-row"
import { RecipePreview } from "./components/recipe-preview"
import { RecipeEditor } from "./components/recipe-editor"
import { RecipesToolbar } from "./components/recipes-toolbar"
import {
  DeleteDialog,
  DuplicateDialog,
  RenameDialog,
} from "./components/row-action-dialogs"
import { TemplateLibraryDialog } from "./components/template-library-dialog"
import { ImportDialog } from "./components/import-dialog"
import type { ArchFilter, Mode, RowAction, SortOrder } from "./types"

type LocationState = {
  overrideDataset?: string
} | null

type RowDialogState = {
  action: RowAction
  recipe: RecipeListEntry
} | null

export function RecipesPage() {
  const list = useQuery({ queryKey: ["recipes"], queryFn: api.listRecipes })
  const recipes = list.data?.recipes ?? []

  const [mode, setMode] = useState<Mode | null>(null)
  // Pre-populated dataset path that flows in from the Datasets page via
  // router state. Once the launch dialog opens we hand it off and clear.
  const [pendingDataset, setPendingDataset] = useState<string | null>(null)
  const [autoOpenLaunch, setAutoOpenLaunch] = useState(false)

  const [query, setQuery] = useState("")
  const [archFilter, setArchFilter] = useState<ArchFilter>("all")
  const [sort, setSort] = useState<SortOrder>("name-asc")

  const [rowDialog, setRowDialog] = useState<RowDialogState>(null)
  const [templateOpen, setTemplateOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)

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

  const visibleRecipes = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = recipes.filter((r) => {
      if (archFilter !== "all" && r.arch !== archFilter) return false
      if (q && !r.name.toLowerCase().includes(q)) return false
      return true
    })
    const sorted = [...filtered].sort((a, b) => {
      switch (sort) {
        case "name-asc":
          return a.name.localeCompare(b.name)
        case "name-desc":
          return b.name.localeCompare(a.name)
        case "modified-desc":
          return (b.modified_at ?? 0) - (a.modified_at ?? 0)
      }
    })
    return sorted
  }, [recipes, query, archFilter, sort])

  // Closing a row dialog returns null; keep the previous recipe reference for
  // animation but reset action so the dialog actually closes.
  const closeRowDialog = () => setRowDialog(null)

  const handleRowAction = (recipe: RecipeListEntry, action: RowAction) => {
    setRowDialog({ recipe, action })
  }

  // After rename/delete of the currently-selected recipe, keep the page in a
  // sensible state: rename → follow the new name; delete → drop selection so
  // the next render's auto-select picks up the first remaining recipe.
  const handleRenameSuccess = (oldName: string, newName: string) => {
    if (
      mode &&
      (mode.kind === "preview" || mode.kind === "edit") &&
      mode.name === oldName
    ) {
      setMode({ kind: "preview", name: newName })
    }
  }

  const handleDeleteSuccess = (deletedName: string) => {
    if (
      mode &&
      (mode.kind === "preview" || mode.kind === "edit") &&
      mode.name === deletedName
    ) {
      setMode(null)
    }
  }

  return (
    <div className="grid grid-cols-[minmax(320px,380px)_1fr] grid-rows-[1fr] h-full min-h-0 overflow-hidden">
      <aside className="border-r border-border/60 flex flex-col min-h-0 min-w-0 overflow-hidden">
        <RecipesToolbar
          dir={list.data?.dir ?? null}
          total={recipes.length}
          visibleCount={visibleRecipes.length}
          query={query}
          onQueryChange={setQuery}
          arch={archFilter}
          onArchChange={setArchFilter}
          sort={sort}
          onSortChange={setSort}
          onCreate={() => setTemplateOpen(true)}
          onImport={() => setImportOpen(true)}
        />
        <ScrollArea className="flex-1 min-h-0">
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
            {!list.isLoading && recipes.length > 0 && visibleRecipes.length === 0 && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                没有匹配的配方。
              </li>
            )}
            {visibleRecipes.map((r) => {
              const active =
                (mode?.kind === "preview" || mode?.kind === "edit") && mode.name === r.name
              return (
                <RecipeRow
                  key={r.name}
                  recipe={r}
                  active={active}
                  onSelect={() => setMode({ kind: "preview", name: r.name })}
                  onAction={(action) => handleRowAction(r, action)}
                />
              )
            })}
          </ul>
        </ScrollArea>
      </aside>

      <section className="min-w-0 min-h-0 flex flex-col bg-background/60 overflow-hidden">
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

      <DuplicateDialog
        open={rowDialog?.action === "duplicate"}
        onOpenChange={(next) => {
          if (!next) closeRowDialog()
        }}
        recipe={rowDialog?.action === "duplicate" ? rowDialog.recipe : null}
        onSuccess={(newName) => setMode({ kind: "preview", name: newName })}
      />
      <RenameDialog
        open={rowDialog?.action === "rename"}
        onOpenChange={(next) => {
          if (!next) closeRowDialog()
        }}
        recipe={rowDialog?.action === "rename" ? rowDialog.recipe : null}
        onSuccess={(newName) => {
          if (rowDialog?.action === "rename") {
            handleRenameSuccess(rowDialog.recipe.name, newName)
          }
        }}
      />
      <DeleteDialog
        open={rowDialog?.action === "delete"}
        onOpenChange={(next) => {
          if (!next) closeRowDialog()
        }}
        recipe={rowDialog?.action === "delete" ? rowDialog.recipe : null}
        onSuccess={handleDeleteSuccess}
      />

      <TemplateLibraryDialog
        open={templateOpen}
        onOpenChange={setTemplateOpen}
        onUseBlank={() => setMode({ kind: "new" })}
        onCreated={(name) => setMode({ kind: "preview", name })}
      />

      <ImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImported={(name) => setMode({ kind: "preview", name })}
      />
    </div>
  )
}

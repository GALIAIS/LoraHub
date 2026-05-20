import { useEffect, useMemo, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useLocation, useNavigate } from "react-router-dom"
import { PanelLeftClose, PanelLeftOpen } from "lucide-react"
import { api, type ConfigListEntry } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import { ConfigRow } from "./components/config-row"
import { ConfigPreview } from "./components/config-preview"
import { ConfigEditor } from "./components/config-editor"
import { ConfigsToolbar } from "./components/configs-toolbar"
import {
  DeleteDialog,
  DuplicateDialog,
  RenameDialog,
} from "./components/row-action-dialogs"
import { TemplateLibraryDialog } from "./components/template-library-dialog"
import { ImportDialog } from "./components/import-dialog"
import type { ArchFilter, BackendFilter, Mode, RowAction, SortOrder } from "./types"

type LocationState = {
  overrideDataset?: string
} | null

type RowDialogState = {
  action: RowAction
  config: ConfigListEntry
} | null

const SIDEBAR_KEY = "lorahub.configs.sidebar"

export function ConfigsPage() {
  const list = useQuery({ queryKey: ["configs"], queryFn: api.listConfigs })
  const configs = list.data?.configs ?? []
  // Pull the workbench-level default backend so the configs list can
  // filter to "the backend the user is actually using" by default.
  // We don't gate the query on settings being loaded — until it lands
  // we just show every config (filter falls through to "all").
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
    staleTime: 60_000,
  })
  const defaultBackend = settingsQuery.data?.settings?.default_backend

  const [mode, setMode] = useState<Mode | null>(null)
  // Pre-populated dataset path that flows in from the Datasets page via
  // router state. Once the launch dialog opens we hand it off and clear.
  const [pendingDataset, setPendingDataset] = useState<string | null>(null)
  const [autoOpenLaunch, setAutoOpenLaunch] = useState(false)

  const [query, setQuery] = useState("")
  const [archFilter, setArchFilter] = useState<ArchFilter>("all")
  const [backendFilter, setBackendFilter] = useState<BackendFilter>("default")
  const [sort, setSort] = useState<SortOrder>("name-asc")

  const [rowDialog, setRowDialog] = useState<RowDialogState>(null)
  const [templateOpen, setTemplateOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => {
    if (typeof window === "undefined") return true
    return window.localStorage.getItem(SIDEBAR_KEY) !== "closed"
  })

  useEffect(() => {
    if (typeof window === "undefined") return
    window.localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? "open" : "closed")
  }, [sidebarOpen])

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

  // Default selection: first config in preview mode, picked from the
  // *visible* list (after backend / arch / search filters). The naive
  // "configs[0]" version showed the global first config — typically a
  // kohya recipe — even when the user had already filtered the
  // sidebar to ``anima_lora``. Switching the auto-select to
  // ``visibleConfigs[0]`` matches the row the user actually sees at
  // the top.
  //
  // Re-run when the visible list changes so flipping the filter
  // immediately refreshes the right pane to a config that actually
  // matches it. We additionally guard against the currently-selected
  // config no longer being in the visible set (typical after the
  // user changes the backend filter and the prior selection is
  // filtered out) — in that case we re-select the first visible
  // config rather than show a stale row.
  // (Defined below as a useMemo over `configs` + filters; this
  // effect is deferred via a dedicated hook block right after.)

  const visibleConfigs = useMemo(() => {
    const q = query.trim().toLowerCase()
    // Resolve the effective backend filter: "default" expands to the
    // workbench setting (or "all" if settings haven't loaded yet);
    // "all" disables the filter; everything else is a direct match.
    const effectiveBackendFilter: BackendFilter | "all" =
      backendFilter === "default"
        ? defaultBackend ?? "all"
        : backendFilter
    const filtered = configs.filter((r) => {
      if (archFilter !== "all" && r.arch !== archFilter) return false
      if (
        effectiveBackendFilter !== "all" &&
        r.backend !== effectiveBackendFilter
      ) {
        return false
      }
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
  }, [configs, query, archFilter, backendFilter, defaultBackend, sort])

  // Auto-select the first *visible* config so the right pane stays in
  // sync with whatever the sidebar's filters are showing. Without this
  // a user who set the backend filter to anima_lora still saw the
  // global first config (typically a kohya recipe) rendered in the
  // right pane and ConfigForm picked the wrong backend section.
  useEffect(() => {
    if (visibleConfigs.length === 0) return
    if (mode === null) {
      setMode({ kind: "preview", name: visibleConfigs[0].name })
      return
    }
    if (autoOpenLaunch && mode.kind !== "preview") {
      setMode({ kind: "preview", name: visibleConfigs[0].name })
      return
    }
    if (mode.kind === "preview" || mode.kind === "edit") {
      const stillVisible = visibleConfigs.some((c) => c.name === mode.name)
      if (!stillVisible) {
        setMode({ kind: "preview", name: visibleConfigs[0].name })
      }
    }
  }, [mode, visibleConfigs, autoOpenLaunch])

  // Closing a row dialog returns null; keep the previous config reference for
  // animation but reset action so the dialog actually closes.
  const closeRowDialog = () => setRowDialog(null)

  const handleRowAction = (config: ConfigListEntry, action: RowAction) => {
    setRowDialog({ config, action })
  }

  // After rename/delete of the currently-selected config, keep the page in a
  // sensible state: rename → follow the new name; delete → drop selection so
  // the next render's auto-select picks up the first remaining config.
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
    <div
      className={cn(
        "grid h-full min-h-0 overflow-hidden grid-rows-[1fr] transition-[grid-template-columns] duration-200",
        sidebarOpen
          ? "grid-cols-[minmax(240px,300px)_1fr]"
          : "grid-cols-[0px_1fr]",
      )}
    >
      <aside
        className={cn(
          "shiro-page-aside flex flex-col min-h-0 min-w-0 overflow-hidden",
          !sidebarOpen && "pointer-events-none opacity-0",
        )}
        aria-hidden={!sidebarOpen}
      >
        <div className="flex items-center justify-between px-4 pt-3">
          <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            训练配置
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setSidebarOpen(false)}
            title="收起侧栏"
          >
            <PanelLeftClose className="size-4" />
          </Button>
        </div>
        <ConfigsToolbar
          dir={list.data?.dir ?? null}
          total={configs.length}
          visibleCount={visibleConfigs.length}
          query={query}
          onQueryChange={setQuery}
          arch={archFilter}
          onArchChange={setArchFilter}
          backend={backendFilter}
          onBackendChange={setBackendFilter}
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
            {!list.isLoading && configs.length === 0 && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                还没有配置。点击 <span className="text-foreground font-medium">+ 新建</span> 创建一个，或运行
                <code className="text-foreground"> lorahub init</code>。
              </li>
            )}
            {!list.isLoading && configs.length > 0 && visibleConfigs.length === 0 && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                没有匹配的配置。
              </li>
            )}
            {visibleConfigs.map((r) => {
              const active =
                (mode?.kind === "preview" || mode?.kind === "edit") && mode.name === r.name
              return (
                <ConfigRow
                  key={r.name}
                  config={r}
                  active={active}
                  onSelect={() => setMode({ kind: "preview", name: r.name })}
                  onAction={(action) => handleRowAction(r, action)}
                />
              )
            })}
          </ul>
        </ScrollArea>
      </aside>

      <section className="min-w-0 min-h-0 flex flex-col bg-background/60 overflow-hidden relative">
        {!sidebarOpen && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => setSidebarOpen(true)}
            className="absolute left-3 top-3 z-10 shadow-[var(--panel-shadow)]"
            title="展开侧栏"
          >
            <PanelLeftOpen className="size-4" />
            <span className="ml-1 text-xs">{configs.length} 个配置</span>
          </Button>
        )}
        {mode === null ? (
          <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
            从左侧选择一个配置查看，或创建新的。
          </div>
        ) : mode.kind === "preview" ? (
          <ConfigPreview
            name={mode.name}
            entry={configs.find((r) => r.name === mode.name) ?? null}
            onEdit={() => setMode({ kind: "edit", name: mode.name })}
            pendingDataset={pendingDataset}
            autoOpenLaunch={autoOpenLaunch}
            onLaunchHandled={() => {
              setPendingDataset(null)
              setAutoOpenLaunch(false)
            }}
          />
        ) : (
          <ConfigEditor mode={mode} setMode={setMode} />
        )}
      </section>

      <DuplicateDialog
        open={rowDialog?.action === "duplicate"}
        onOpenChange={(next) => {
          if (!next) closeRowDialog()
        }}
        config={rowDialog?.action === "duplicate" ? rowDialog.config : null}
        onSuccess={(newName) => setMode({ kind: "preview", name: newName })}
      />
      <RenameDialog
        open={rowDialog?.action === "rename"}
        onOpenChange={(next) => {
          if (!next) closeRowDialog()
        }}
        config={rowDialog?.action === "rename" ? rowDialog.config : null}
        onSuccess={(newName) => {
          if (rowDialog?.action === "rename") {
            handleRenameSuccess(rowDialog.config.name, newName)
          }
        }}
      />
      <DeleteDialog
        open={rowDialog?.action === "delete"}
        onOpenChange={(next) => {
          if (!next) closeRowDialog()
        }}
        config={rowDialog?.action === "delete" ? rowDialog.config : null}
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

import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Save, RotateCcw } from "lucide-react"
import { api, type BackendId, type SettingsState } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface FieldProps {
  label: string
  description: string
  value: string
  placeholder?: string
  onChange: (v: string) => void
}

function Field({ label, description, value, placeholder, onChange }: FieldProps) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-x-4 items-start">
      <Label className="text-xs pt-2">{label}</Label>
      <div className="min-w-0">
        <Input
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          className="font-mono w-full max-w-2xl"
        />
        <p className="text-[11px] text-muted-foreground/80 mt-1">{description}</p>
      </div>
    </div>
  )
}

/**
 * Per-backend path editor + default-backend selector. Saving persists every
 * editable field at once (the PUT /api/settings endpoint accepts a partial
 * patch but we write all fields the form owns to keep the dirty-detection
 * model simple).
 */
export function BackendsTab() {
  const qc = useQueryClient()
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  })

  const [draft, setDraft] = useState<SettingsState | null>(null)

  // Sync draft whenever the server-side settings load or change. We mirror
  // the pattern from the old single-page form so users never lose unsaved
  // edits to a background refetch.
  useEffect(() => {
    if (settingsQuery.data) {
      setDraft(settingsQuery.data.settings)
    }
  }, [settingsQuery.data])

  const update = useMutation({
    mutationFn: (patch: Partial<SettingsState>) => api.updateSettings(patch),
    onSuccess: (data) => {
      qc.setQueryData(["settings"], data)
      qc.invalidateQueries({ queryKey: ["backends"] })
      qc.invalidateQueries({ queryKey: ["health"] })
      setDraft(data.settings)
    },
  })

  if (!draft || !settingsQuery.data) {
    return <div className="text-sm text-muted-foreground">正在加载设置…</div>
  }

  const saved = settingsQuery.data.settings
  const dirty =
    draft.sd_scripts_path !== saved.sd_scripts_path ||
    draft.python_executable !== saved.python_executable ||
    draft.diffusion_pipe_repo_path !== saved.diffusion_pipe_repo_path ||
    draft.diffusion_pipe_python !== saved.diffusion_pipe_python ||
    draft.default_backend !== saved.default_backend

  return (
    <div className="space-y-5">
      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">默认后端</CardTitle>
          <CardDescription>
            未在配方或命令行中显式指定时使用的后端。
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-[10rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">默认后端</Label>
            <Select
              value={draft.default_backend}
              onValueChange={(v) =>
                setDraft({ ...draft, default_backend: v as BackendId })
              }
            >
              <SelectTrigger className="w-64 text-xs font-mono">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="kohya">kohya-ss/sd-scripts</SelectItem>
                <SelectItem value="diffusion-pipe">
                  tdrussell/diffusion-pipe
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Kohya 后端</CardTitle>
          <CardDescription>
            指定 kohya-ss/sd-scripts 检出目录与运行它的 Python 解释器。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Field
            label="sd-scripts 路径"
            description="kohya-ss/sd-scripts 检出目录的绝对路径。留空则使用 ./sd-scripts（项目内）或 platformdirs 用户目录。"
            value={draft.sd_scripts_path ?? ""}
            placeholder="C:\\path\\to\\sd-scripts"
            onChange={(v) => setDraft({ ...draft, sd_scripts_path: v || null })}
          />
          <Field
            label="Python 解释器"
            description="可选。默认使用 <sd-scripts>/venv/Scripts/python.exe（Windows）或 .../bin/python（Unix）。"
            value={draft.python_executable ?? ""}
            placeholder="<sd-scripts>/venv/Scripts/python.exe"
            onChange={(v) => setDraft({ ...draft, python_executable: v || null })}
          />
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Diffusion-pipe 后端</CardTitle>
          <CardDescription>
            指定 tdrussell/diffusion-pipe 检出目录与其 DeepSpeed venv 的 Python。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Field
            label="repo 路径"
            description="diffusion-pipe 检出目录的绝对路径。留空则使用 ./diffusion-pipe（项目内）或 platformdirs 用户目录。"
            value={draft.diffusion_pipe_repo_path ?? ""}
            placeholder="C:\\path\\to\\diffusion-pipe"
            onChange={(v) =>
              setDraft({ ...draft, diffusion_pipe_repo_path: v || null })
            }
          />
          <Field
            label="Python 解释器"
            description="可选。默认使用 <repo>/venv/Scripts/python.exe（Windows）或 .../bin/python（Unix）。"
            value={draft.diffusion_pipe_python ?? ""}
            placeholder="<repo>/venv/Scripts/python.exe"
            onChange={(v) =>
              setDraft({ ...draft, diffusion_pipe_python: v || null })
            }
          />
        </CardContent>
      </Card>

      <div className="flex items-center gap-3 sticky bottom-4 bg-background/80 backdrop-blur rounded-[4px] border border-border/60 px-4 py-3 shadow-[var(--panel-shadow)]">
        <Button
          size="sm"
          disabled={!dirty || update.isPending}
          onClick={() =>
            update.mutate({
              sd_scripts_path: draft.sd_scripts_path,
              python_executable: draft.python_executable,
              diffusion_pipe_repo_path: draft.diffusion_pipe_repo_path,
              diffusion_pipe_python: draft.diffusion_pipe_python,
              default_backend: draft.default_backend,
            })
          }
        >
          <Save className="size-3" />
          {update.isPending ? "保存中…" : "保存"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={!dirty || update.isPending}
          onClick={() => settingsQuery.data && setDraft(settingsQuery.data.settings)}
        >
          <RotateCcw className="size-3" />
          重置
        </Button>
        {update.isError && (
          <span className="text-xs text-destructive font-mono">
            {(update.error as Error).message}
          </span>
        )}
        {update.isSuccess && !dirty && (
          <span className="text-xs text-emerald-600 dark:text-emerald-400">
            已保存。
          </span>
        )}
        <span className="ml-auto text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70 font-mono truncate">
          {settingsQuery.data.path}
        </span>
      </div>
    </div>
  )
}

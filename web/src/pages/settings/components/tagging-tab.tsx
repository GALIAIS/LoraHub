import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Save, RotateCcw } from "lucide-react"
import { api, type SettingsState } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card"

type Device = "auto" | "cpu" | "cuda"

const DEVICES: Array<{ value: Device; label: string }> = [
  { value: "auto", label: "自动" },
  { value: "cpu", label: "CPU" },
  { value: "cuda", label: "CUDA" },
]

/**
 * WD14 device picker. Owns its own dirty state so saving here doesn't
 * collide with edits to backend paths in the sibling "后端管理" tab.
 */
export function TaggingTab() {
  const qc = useQueryClient()
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  })

  const [device, setDevice] = useState<Device | null>(null)

  useEffect(() => {
    if (settingsQuery.data) {
      setDevice(settingsQuery.data.settings.tagger_device)
    }
  }, [settingsQuery.data])

  const update = useMutation({
    mutationFn: (patch: Partial<SettingsState>) => api.updateSettings(patch),
    onSuccess: (data) => {
      qc.setQueryData(["settings"], data)
      qc.invalidateQueries({ queryKey: ["health"] })
      setDevice(data.settings.tagger_device)
    },
  })

  if (!device || !settingsQuery.data) {
    return <div className="text-sm text-muted-foreground">正在加载设置…</div>
  }

  const dirty = device !== settingsQuery.data.settings.tagger_device

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">自动标注</CardTitle>
          <CardDescription>
            <code className="text-foreground">lorahub tag</code> 与 Web 标注器在未显式指定时使用的默认设备。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-[10rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">WD14 设备</Label>
            <div className="flex gap-2">
              {DEVICES.map((d) => (
                <Button
                  key={d.value}
                  type="button"
                  size="sm"
                  variant={device === d.value ? "default" : "outline"}
                  onClick={() => setDevice(d.value)}
                >
                  {d.label}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center gap-3 sticky bottom-4 bg-background/80 backdrop-blur rounded-[4px] border border-border/60 px-4 py-3 shadow-[var(--panel-shadow)]">
        <Button
          size="sm"
          disabled={!dirty || update.isPending}
          onClick={() => update.mutate({ tagger_device: device })}
        >
          <Save className="size-3" />
          {update.isPending ? "保存中…" : "保存"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={!dirty || update.isPending}
          onClick={() =>
            settingsQuery.data && setDevice(settingsQuery.data.settings.tagger_device)
          }
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

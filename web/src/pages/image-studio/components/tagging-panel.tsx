import { useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Tag } from "lucide-react"
import { api } from "@/lib/api"
import type { TaggingSession } from "@/lib/api"

interface TaggingPanelProps {
  datasetPath: string
}

export function TaggingPanel({ datasetPath }: TaggingPanelProps) {
  const [model, setModel] = useState("wd-swinv2-v3")
  const [device, setDevice] = useState<"auto" | "cpu" | "cuda">("auto")
  const [general, setGeneral] = useState(0.35)
  const [character, setCharacter] = useState(0.85)
  const [overwrite, setOverwrite] = useState(false)
  const [recursive, setRecursive] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)

  const startMutation = useMutation({
    mutationFn: () =>
      api.tagDataset({
        path: datasetPath,
        model_id: model,
        device,
        general,
        character,
        overwrite,
        recursive,
      }),
    onSuccess: (data: TaggingSession) => {
      setSessionId(data.session_id)
    },
  })

  const sessionQuery = useQuery({
    queryKey: ["tagging-session", sessionId],
    queryFn: () => api.getTaggingSession(sessionId!),
    enabled: !!sessionId,
    refetchInterval: (query) => {
      const d = query.state.data as TaggingSession | undefined
      if (d && (d.status === "succeeded" || d.status === "failed")) return false
      return 2000
    },
  })

  const session = sessionQuery.data

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center gap-2 mb-1">
        <Tag className="size-4 text-primary" />
        <span className="text-xs font-semibold">WD14/JoyTag 标注</span>
      </div>

      <label className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground w-16">模型</span>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="rounded border bg-background px-2 py-1 text-xs flex-1"
        >
          <option value="wd-swinv2-v3">WD SwinV2 v3</option>
          <option value="wd-vit-v3">WD ViT v3</option>
          <option value="wd-convnext-v3">WD ConvNext v3</option>
          <option value="joytag">JoyTag</option>
        </select>
      </label>

      <label className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground w-16">设备</span>
        <select
          value={device}
          onChange={(e) => setDevice(e.target.value as "auto" | "cpu" | "cuda")}
          className="rounded border bg-background px-2 py-1 text-xs flex-1"
        >
          <option value="auto">自动</option>
          <option value="cuda">CUDA</option>
          <option value="cpu">CPU</option>
        </select>
      </label>

      <label className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground w-16">通用阈值</span>
        <input
          type="number"
          step="0.05"
          min="0"
          max="1"
          value={general}
          onChange={(e) => setGeneral(Number(e.target.value))}
          className="rounded border bg-background px-2 py-1 text-xs w-20"
        />
      </label>

      <label className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground w-16">角色阈值</span>
        <input
          type="number"
          step="0.05"
          min="0"
          max="1"
          value={character}
          onChange={(e) => setCharacter(Number(e.target.value))}
          className="rounded border bg-background px-2 py-1 text-xs w-20"
        />
      </label>

      <div className="flex items-center gap-4">
        <label className="flex items-center gap-1.5 text-xs">
          <input
            type="checkbox"
            checked={overwrite}
            onChange={(e) => setOverwrite(e.target.checked)}
            className="size-3"
          />
          覆盖已有
        </label>
        <label className="flex items-center gap-1.5 text-xs">
          <input
            type="checkbox"
            checked={recursive}
            onChange={(e) => setRecursive(e.target.checked)}
            className="size-3"
          />
          递归子目录
        </label>
      </div>

      <button
        type="button"
        onClick={() => startMutation.mutate()}
        disabled={startMutation.isPending || (session?.status === "running")}
        className="rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
      >
        {startMutation.isPending ? "启动中..." : "开始标注"}
      </button>

      {session && (
        <div className="rounded border p-2 text-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="font-medium">
              {session.status === "running" && "标注进行中..."}
              {session.status === "succeeded" && "标注完成"}
              {session.status === "failed" && "标注失败"}
            </span>
            <span className="text-muted-foreground">{session.percent}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                session.status === "failed" ? "bg-destructive" : "bg-primary"
              }`}
              style={{ width: `${session.percent}%` }}
            />
          </div>
          {session.written != null && session.total != null && (
            <p className="mt-1 text-muted-foreground">
              已写入 {session.written} / {session.total}
            </p>
          )}
          {session.error && (
            <p className="mt-1 text-destructive">{session.error}</p>
          )}
        </div>
      )}
    </div>
  )
}

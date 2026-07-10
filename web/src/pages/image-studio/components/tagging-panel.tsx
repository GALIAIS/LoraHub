import { useEffect, useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Tag } from "lucide-react"
import { api, stopTaggingSession } from "@/lib/api"
import type { TaggingSession } from "@/lib/api"

interface TaggingPanelProps {
  datasetPath: string
}

// Server-side ground truth — see ai-bulk-modal.tsx for the rationale.
const FALLBACK_DEFAULT_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"

export function TaggingPanel({ datasetPath }: TaggingPanelProps) {
  const [model, setModel] = useState<string>(FALLBACK_DEFAULT_MODEL)
  const [device, setDevice] = useState<"auto" | "cpu" | "cuda">("auto")
  const [general, setGeneral] = useState(0.35)
  const [character, setCharacter] = useState(0.85)
  const [overwrite, setOverwrite] = useState(false)
  const [recursive, setRecursive] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [dismissedSessionId, setDismissedSessionId] = useState<string | null>(null)

  // Same dropdown source as ai-bulk-modal — short names like
  // ``wd-swinv2-v3`` are missing the ``SmilingWolf/`` owner and the
  // ``-tagger`` segment, which trips the HF resolver and 401s the
  // tagging request mid-flight.
  const wd14Models = useQuery({
    queryKey: ["wd14-models"],
    queryFn: api.listWd14Models,
    staleTime: 60 * 60 * 1000,
  })
  useEffect(() => {
    if (wd14Models.data?.default && model === FALLBACK_DEFAULT_MODEL) {
      setModel(wd14Models.data.default)
    }
  }, [wd14Models.data?.default, model])
  const modelOptions = wd14Models.data?.models ?? [
    { id: FALLBACK_DEFAULT_MODEL, label: "v3 · EvaCLIP-Large" },
  ]

  const latestTaggingTask = useQuery({
    queryKey: ["tasks", "latest", "tagging"],
    queryFn: () => api.getLatestTask("tagging"),
    retry: false,
    staleTime: 10_000,
  })

  useEffect(() => {
    if (sessionId != null) return
    const latest = latestTaggingTask.data
    if (
      latest?.metadata?.path === datasetPath &&
      latest.id !== dismissedSessionId
    ) {
      setSessionId(latest.id)
    }
  }, [datasetPath, dismissedSessionId, latestTaggingTask.data, sessionId])

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
      setDismissedSessionId(null)
      setSessionId(data.session_id)
    },
  })

  const sessionQuery = useQuery({
    queryKey: ["tagging-session", sessionId],
    queryFn: () => api.getTaggingSession(sessionId!),
    enabled: !!sessionId,
    refetchInterval: (query) => {
      const d = query.state.data as TaggingSession | undefined
      if (
        d &&
        (d.status === "succeeded" ||
          d.status === "failed" ||
          d.status === "canceled" ||
          d.status === "interrupted")
      ) {
        return false
      }
      return 2000
    },
  })

  const session = sessionQuery.data

  const stopMutation = useMutation({
    mutationFn: () => stopTaggingSession(session!.session_id),
    onSuccess: () => {
      void sessionQuery.refetch()
    },
  })

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
          {modelOptions.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
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
        disabled={
          startMutation.isPending ||
          session?.status === "running" ||
          session?.status === "stop_requested"
        }
        className="rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
      >
        {startMutation.isPending ? "启动中…" : "开始标注"}
      </button>

      {session && (
        <div className="rounded border p-2 text-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="font-medium">
              {session.status === "running" && "标注进行中…"}
              {session.status === "stop_requested" && "正在停止标注…"}
              {session.status === "succeeded" && "标注完成"}
              {session.status === "failed" && "标注失败"}
              {session.status === "canceled" && "标注已停止"}
              {session.status === "interrupted" && "标注已中断"}
            </span>
            <span className="text-muted-foreground">{session.percent}%</span>
          </div>
          <div className="shiro-progress-track h-1.5 border-0 bg-muted">
            <div
              className={`shiro-progress-fill ${
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
          {session.status === "running" || session.status === "stop_requested" ? (
            <button
              type="button"
              className="mt-2 rounded border border-destructive/40 px-2 py-1 text-[11px] text-destructive hover:bg-destructive/10 disabled:opacity-50"
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending || session.status === "stop_requested"}
            >
              {stopMutation.isPending || session.status === "stop_requested"
                ? "停止中..."
                : "停止"}
            </button>
          ) : (
            <button
              type="button"
              className="mt-2 rounded border px-2 py-1 text-[11px] text-muted-foreground hover:text-foreground"
              onClick={() => {
                setDismissedSessionId(session.session_id)
                setSessionId(null)
              }}
            >
              关闭结果
            </button>
          )}
        </div>
      )}
    </div>
  )
}

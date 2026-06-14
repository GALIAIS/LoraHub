import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { ApiError, api } from "@/lib/api"

export type Source = "huggingface" | "modelscope"

export const SOURCE_LABEL: Record<Source, string> = {
  huggingface: "HuggingFace",
  modelscope: "ModelScope",
}

export const SOURCE_OPTIONS: { value: Source; label: string }[] = [
  { value: "modelscope", label: SOURCE_LABEL.modelscope },
  { value: "huggingface", label: SOURCE_LABEL.huggingface },
]

export const DEFAULT_ALLOW_PATTERNS = [
  "*.safetensors",
  "*.ckpt",
  "*.pt",
  "*.pth",
  "*.bin",
  "*.gguf",
  "*.onnx",
  "*.json",
  "*.txt",
  "*.model",
  "*.vocab",
  "*.merges",
].join(", ")

export const DEFAULT_IGNORE_PATTERNS = [
  ".gitattributes",
  "README*",
  "LICENSE*",
  "*.md",
  "*.png",
  "*.jpg",
  "*.jpeg",
  "*.webp",
  "*.gif",
  "*.mp4",
  "*.zip",
  "*.tar",
  "*.tar.gz",
].join(", ")

const MODEL_DOWNLOAD_SESSION_KEY = "lorahub:model-download-session-id"

export function useModelDownloadPanel() {
  const [source, setSource] = useState<Source>("modelscope")
  const [repoId, setRepoId] = useState("")
  const [revision, setRevision] = useState("")
  const [targetDir, setTargetDir] = useState("")
  const [threads, setThreads] = useState(4)
  const [allowPatterns, setAllowPatterns] = useState(DEFAULT_ALLOW_PATTERNS)
  const [ignorePatterns, setIgnorePatterns] = useState(DEFAULT_IGNORE_PATTERNS)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())
  const [lastListedKey, setLastListedKey] = useState("")

  const parsedAllowPatterns = useMemo(
    () => parsePatterns(allowPatterns),
    [allowPatterns],
  )
  const parsedIgnorePatterns = useMemo(
    () => parsePatterns(ignorePatterns),
    [ignorePatterns],
  )

  const latestDownload = useQuery({
    queryKey: ["model-download-latest"],
    queryFn: api.getLatestModelDownload,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 800 : false,
    staleTime: 400,
  })

  useEffect(() => {
    const stored = window.localStorage.getItem(MODEL_DOWNLOAD_SESSION_KEY)
    if (stored) setSessionId(stored)
  }, [])

  useEffect(() => {
    const latest = latestDownload.data
    if (!latest?.session_id) return
    if (latest.status === "running" || !sessionId) {
      setSessionId(latest.session_id)
      window.localStorage.setItem(MODEL_DOWNLOAD_SESSION_KEY, latest.session_id)
    }
  }, [latestDownload.data, sessionId])

  const startDownload = useMutation({
    mutationFn: () =>
      api.downloadModel({
        source,
        repo_id: repoId.trim(),
        revision: revision.trim() || (source === "modelscope" ? "master" : "main"),
        target_dir: targetDir.trim() || undefined,
        threads,
        paths: Array.from(selectedPaths),
        allow_patterns: parsedAllowPatterns,
        ignore_patterns: parsedIgnorePatterns,
      }),
    onSuccess: (session) => {
      setSessionId(session.session_id)
      window.localStorage.setItem(MODEL_DOWNLOAD_SESSION_KEY, session.session_id)
    },
  })

  const fileList = useMutation({
    mutationFn: () =>
      api.listModelFiles({
        source,
        repo_id: repoId.trim(),
        revision: revision.trim() || (source === "modelscope" ? "master" : "main"),
        allow_patterns: parsedAllowPatterns,
        ignore_patterns: parsedIgnorePatterns,
      }),
    onSuccess: (res) => {
      setSelectedPaths(
        new Set(res.files.filter((file) => file.selected).map((file) => file.path)),
      )
      setLastListedKey(
        listKey(source, repoId, revision, parsedAllowPatterns, parsedIgnorePatterns),
      )
    },
  })

  useEffect(() => {
    setSelectedPaths(new Set())
    setLastListedKey("")
    fileList.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, repoId, revision, allowPatterns, ignorePatterns])

  const session = useQuery({
    queryKey: ["model-download", sessionId],
    queryFn: () => api.getModelDownload(sessionId!),
    enabled: !!sessionId,
    refetchInterval: (query) =>
      query.state.data?.status === "running" || !query.state.data ? 800 : false,
    staleTime: 400,
  })

  useEffect(() => {
    if (!(session.error instanceof ApiError) || session.error.status !== 404) {
      return
    }
    setSessionId(null)
    window.localStorage.removeItem(MODEL_DOWNLOAD_SESSION_KEY)
    void latestDownload.refetch()
  }, [latestDownload, session.error])

  const latestCurrent =
    latestDownload.data?.session_id &&
    latestDownload.data.status !== "idle" &&
    (!sessionId || latestDownload.data.session_id === sessionId)
      ? latestDownload.data
      : null
  const current = session.data ?? startDownload.data ?? latestCurrent ?? null
  const error = (startDownload.error as Error | undefined) ?? (session.error as Error | undefined)
  const ready = repoId.includes("/") && repoId.trim().length > 2
  const running = current?.status === "running" || startDownload.isPending
  const listing = fileList.isPending
  const latest = current?.events.at(-1)
  const percent = Math.max(0, Math.min(100, current?.percent ?? 0))
  const currentSource = current?.source
  const listed = fileList.data?.files ?? []
  const selectedFiles = listed.filter((file) => selectedPaths.has(file.path))
  const selectedBytes = selectedFiles.reduce((sum, file) => sum + file.size, 0)
  const staleList =
    lastListedKey !==
    listKey(source, repoId, revision, parsedAllowPatterns, parsedIgnorePatterns)
  const canDownload = ready && selectedPaths.size > 0 && !running && !staleList
  const result = current?.result
  const summary = useMemo(() => {
    if (!current) return "尚未开始下载"
    if (current.status === "running") return latest?.message ?? "下载进行中"
    if (current.status === "failed") return current.error ?? "下载失败"
    if (current.status === "canceled") return current.error ?? "下载已取消"
    if (current.status === "interrupted") return current.error ?? "下载已中断"
    return "下载完成"
  }, [current, latest])

  return {
    source,
    setSource,
    repoId,
    setRepoId,
    revision,
    setRevision,
    targetDir,
    setTargetDir,
    threads,
    setThreads,
    allowPatterns,
    setAllowPatterns,
    ignorePatterns,
    setIgnorePatterns,
    selectedPaths,
    setSelectedPaths,
    startDownload,
    fileList,
    current,
    error,
    ready,
    running,
    listing,
    latest,
    percent,
    currentSource,
    listed,
    selectedBytes,
    staleList,
    canDownload,
    result,
    summary,
  }
}

export function togglePath(
  path: string,
  setSelectedPaths: Dispatch<SetStateAction<Set<string>>>,
) {
  setSelectedPaths((current) => {
    const next = new Set(current)
    if (next.has(path)) {
      next.delete(path)
    } else {
      next.add(path)
    }
    return next
  })
}

export function clampThreadCount(value: number): number {
  if (!Number.isFinite(value)) return 1
  return Math.max(1, Math.min(16, Math.round(value)))
}

function parsePatterns(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((part) => part.trim())
    .filter(Boolean)
}

function listKey(
  source: Source,
  repoId: string,
  revision: string,
  allowPatterns: string[],
  ignorePatterns: string[],
): string {
  return [
    source,
    repoId.trim(),
    revision.trim(),
    allowPatterns.join("\0"),
    ignorePatterns.join("\0"),
  ].join(":")
}

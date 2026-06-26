import { useCallback, useEffect, useRef, useState } from "react"
import { Upload } from "lucide-react"

interface UploadTask {
  id: string
  file: File
  status: "pending" | "uploading" | "extracting" | "done" | "error"
  percent: number
  error?: string
  extracted?: number
}

interface UploadDropZoneProps {
  datasetName: string
  onComplete: () => void
}

interface UploadSummary {
  doneTasks: number
  importedFiles: number
}

export function UploadDropZone({ datasetName, onComplete }: UploadDropZoneProps) {
  const [dragging, setDragging] = useState(false)
  const [queue, setQueue] = useState<UploadTask[]>([])
  const [summary, setSummary] = useState<UploadSummary>({
    doneTasks: 0,
    importedFiles: 0,
  })
  const processingRef = useRef(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const completeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Debounced onComplete: batch rapid completions into a single refresh
  const debouncedComplete = useCallback(() => {
    if (completeTimerRef.current) clearTimeout(completeTimerRef.current)
    completeTimerRef.current = setTimeout(() => {
      onComplete()
      completeTimerRef.current = null
    }, 300)
  }, [onComplete])

  const isArchive = (name: string) =>
    /\.(zip|tar|tar\.gz|tgz|7z|rar|gz)$/i.test(name)

  const enqueueFiles = (fileList: FileList | File[]) => {
    const files = Array.from(fileList)
    if (files.length === 0) return
    const newTasks: UploadTask[] = files.map((f, i) => ({
      id: `${Date.now()}-${i}`,
      file: f,
      status: "pending",
      percent: 0,
    }))
    setQueue((prev) => [...prev, ...newTasks])
  }

  useEffect(() => {
    if (processingRef.current) return
    const pending = queue.find((t) => t.status === "pending")
    if (!pending) {
      // Auto-trim: keep at most 10 completed/errored tasks visible
      const terminal = queue.filter((t) => t.status === "done" || t.status === "error")
      if (terminal.length > 10) {
        setQueue((prev) => {
          const keep = prev.filter((t) => t.status === "pending" || t.status === "uploading" || t.status === "extracting")
          const done = prev.filter((t) => t.status === "done" || t.status === "error")
          return [...keep, ...done.slice(-10)]
        })
      }
      return
    }
    processingRef.current = true

    const updateTask = (id: string, patch: Partial<UploadTask>) => {
      setQueue((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)))
    }

    const processTask = (task: UploadTask) => {
      updateTask(task.id, { status: "uploading", percent: 0 })

      const formData = new FormData()
      formData.append("files", task.file)
      formData.append("keepCaptions", "true")
      formData.append("onConflict", "rename")

      const xhr = new XMLHttpRequest()
      xhr.open(
        "POST",
        `/api/image-studio/datasets/${encodeURIComponent(datasetName)}/upload`,
      )

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          updateTask(task.id, { percent: Math.round((e.loaded / e.total) * 100) })
        }
      }

      xhr.upload.onload = () => {
        if (isArchive(task.file.name)) {
          updateTask(task.id, { status: "extracting", percent: 100 })
        }
      }

      xhr.onreadystatechange = () => {
        if (xhr.readyState === XMLHttpRequest.DONE) {
          if (xhr.status >= 200 && xhr.status < 300) {
            let extracted = 0
            const text = xhr.responseText
            for (const line of text.split("\n")) {
              if (line.startsWith("data: ")) {
                try {
                  const d = JSON.parse(line.slice(6))
                  if (d.totalExtracted !== undefined) extracted = d.totalExtracted
                  else if (d.count !== undefined) extracted += d.count
                } catch {
                  /* skip */
                }
              }
            }
            updateTask(task.id, { status: "done", percent: 100, extracted })
            setSummary((prev) => ({
              doneTasks: prev.doneTasks + 1,
              importedFiles: prev.importedFiles + extracted,
            }))
          } else {
            updateTask(task.id, { status: "error", error: `HTTP ${xhr.status}` })
          }
          processingRef.current = false
          debouncedComplete()
        }
      }

      xhr.onerror = () => {
        updateTask(task.id, { status: "error", error: "网络错误" })
        processingRef.current = false
      }

      xhr.send(formData)
    }

    processTask(pending)
  }, [queue, datasetName, debouncedComplete])

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files.length > 0) enqueueFiles(e.dataTransfer.files)
  }

  const clearDone = () =>
    setQueue((prev) => {
      const active = prev.filter((t) => t.status !== "done" && t.status !== "error")
      if (active.length === 0) {
        setSummary({ doneTasks: 0, importedFiles: 0 })
      }
      return active
    })

  const activeTasks = queue.filter((t) => t.status !== "done" && t.status !== "error")
  const doneTasks = queue.filter((t) => t.status === "done")
  const hasActive = activeTasks.length > 0

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`mx-4 mt-2 shrink-0 rounded-lg border-2 border-dashed transition-colors ${
        dragging
          ? "border-primary bg-primary/5"
          : "border-muted-foreground/20 hover:border-muted-foreground/40"
      }`}
    >
      <div className="px-4 py-2">
        <div className="flex items-center gap-3">
          <Upload className="size-4 text-muted-foreground shrink-0" />
          {!hasActive && queue.length === 0 && (
            <span className="text-xs text-muted-foreground flex-1">
              拖入图片或压缩包 (.zip/.tar.gz/.7z) 上传
            </span>
          )}
          {!hasActive && summary.doneTasks > 0 && (
            <span className="text-xs text-muted-foreground flex-1">
              已完成 {summary.doneTasks} 个任务，共导入 {summary.importedFiles} 个文件
            </span>
          )}
          {hasActive && (
            <span className="text-xs text-muted-foreground flex-1">
              队列：{activeTasks.length} 个任务等待处理
            </span>
          )}
          <div className="flex items-center gap-2">
            {doneTasks.length > 0 && (
              <button
                type="button"
                onClick={clearDone}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                清除已完成
              </button>
            )}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="rounded bg-muted px-2 py-1 text-xs hover:bg-muted/80"
            >
              选择文件
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*,.zip,.tar,.tar.gz,.tgz,.7z"
              className="hidden"
              onChange={(e) => {
                if (e.target.files) enqueueFiles(e.target.files)
                e.target.value = ""
              }}
            />
          </div>
        </div>

        {queue.length > 0 && (
          <div className="mt-2 flex flex-col gap-1 max-h-32 overflow-y-auto">
            {queue.map((task) => (
              <div key={task.id} className="flex items-center gap-2 text-xs">
                <span
                  className={`size-2 rounded-full shrink-0 ${
                    task.status === "done"
                      ? "bg-green-500"
                      : task.status === "error"
                        ? "bg-red-500"
                        : task.status === "uploading" || task.status === "extracting"
                          ? "bg-primary animate-pulse"
                          : "bg-muted-foreground/30"
                  }`}
                />
                <span className="truncate flex-1 max-w-[200px]">{task.file.name}</span>
                <span className="text-muted-foreground shrink-0 w-16 text-right">
                  {task.status === "pending" && "等待中"}
                  {task.status === "uploading" && `${task.percent}%`}
                  {task.status === "extracting" && "解压中"}
                  {task.status === "done" && "完成"}
                  {task.status === "error" && (task.error || "失败")}
                </span>
                {(task.status === "uploading" || task.status === "extracting") && (
                  <div className="shiro-progress-track h-1.5 w-20 border-0 bg-muted">
                    <div
                      className={`shiro-progress-fill ${
                        task.status === "extracting" ? "bg-amber-500 animate-pulse" : "bg-primary"
                      }`}
                      style={{ width: `${task.percent}%` }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

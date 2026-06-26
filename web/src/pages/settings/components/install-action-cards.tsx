import { AlertTriangle, Check, Download, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type {
  AnimaLoraBackendStatus,
  AnimaModelDownloadStatus,
  MsvcInstallStatus,
} from "@/lib/api"

export function AnimaModelDownloadCard({
  missing,
  status,
  isPending,
  onDownload,
  error,
}: {
  missing: string[]
  status: AnimaModelDownloadStatus | undefined
  isPending: boolean
  onDownload: () => void
  error: string | null
}) {
  const isRunning = status?.status === "running"
  const failed = status?.status === "failed"
  const succeeded = status?.status === "succeeded" && missing.length === 0
  const percent = status?.percent ?? 0
  const filesDone = status?.files_done ?? 0
  const filesTotal = status?.files_total ?? missing.length
  const lastEvent = status?.events?.[status.events.length - 1]

  return (
    <div className="rounded-[4px] border border-amber-500/40 bg-amber-500/5 px-3 py-3 space-y-2.5">
      <div className="flex items-start gap-3">
        <AlertTriangle className="size-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <div className="flex-1 text-xs text-amber-700 dark:text-amber-400">
          <div className="font-semibold text-foreground">
            anima 模型未就绪
          </div>
          <div className="mt-0.5 leading-relaxed">
            训练 / 推理需要 3 个 safetensors 检查点（DiT 基模型、Qwen3 文本编码器、
            Qwen Image VAE）。从 ModelScope
            <code className="mx-1 text-foreground">circlestone-labs/Anima</code>
            下载到项目根
            <code className="mx-1 text-foreground">models/</code>
            目录。
          </div>
          {missing.length > 0 && !succeeded && (
            <ul className="mt-1.5 ml-2 font-mono text-[11px] space-y-0.5 text-muted-foreground">
              {missing.map((f) => (
                <li key={f}>· {f}</li>
              ))}
            </ul>
          )}
        </div>
        <Button
          size="sm"
          variant={succeeded ? "outline" : "default"}
          disabled={isRunning || isPending || succeeded}
          onClick={onDownload}
        >
          {isRunning || isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : succeeded ? (
            <Check className="size-3" />
          ) : (
            <Download className="size-3" />
          )}
          {isRunning
            ? "下载中…"
            : succeeded
              ? "已完成"
              : failed
                ? "重试"
                : "下载模型"}
        </Button>
      </div>

      {(isRunning || filesDone > 0) && filesTotal > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] text-muted-foreground">
            <span>
              已完成 {filesDone} / {filesTotal} 文件
            </span>
            <span className="font-mono tabular-nums">{percent.toFixed(0)}%</span>
          </div>
          <div className="shiro-progress-track h-1.5">
            <div
              className={cn(
                "shiro-progress-fill",
                failed ? "bg-destructive" : "bg-emerald-500",
              )}
              style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
            />
          </div>
          {lastEvent && (
            <div className="font-mono text-[10px] text-muted-foreground/80 break-all">
              {status?.source ? `${status.source}: ` : ""}
              {lastEvent.message}
            </div>
          )}
        </div>
      )}

      {(failed || error) && (
        <div className="rounded-[3px] border border-destructive/40 bg-destructive/5 px-2.5 py-1.5 text-[11px] font-mono text-destructive break-all">
          {status?.error || error}
        </div>
      )}
    </div>
  )
}

export function MsvcInstallCard({
  detection,
  status,
  isPending,
  onInstall,
  error,
}: {
  detection: AnimaLoraBackendStatus["msvc"]
  status: MsvcInstallStatus | undefined
  isPending: boolean
  onInstall: () => void
  error: string | null
}) {
  const isRunning = status?.status === "running"
  const failed = status?.status === "failed"
  const succeeded = status?.msvc?.ok || status?.status === "succeeded"
  const log = status?.log ?? []
  const lastLine = log.length > 0 ? log[log.length - 1] : null

  return (
    <div className="rounded-[4px] border border-amber-500/40 bg-amber-500/5 px-3 py-3 space-y-2.5">
      <div className="flex items-start gap-3">
        <AlertTriangle className="size-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <div className="flex-1 text-xs text-amber-700 dark:text-amber-400">
          <div className="font-semibold text-foreground">
            缺少 Visual Studio Build Tools
          </div>
          <div className="mt-0.5 leading-relaxed">
            anima_lora 使用
            <code className="mx-1 text-foreground">torch.compile</code>
            ，PyTorch Inductor 需要通过 triton-windows 调用
            <code className="mx-1 text-foreground">cl.exe</code>
            。未检测到 cl.exe 时无法完成编译。
          </div>
          <div className="mt-1 leading-relaxed">
            点击下方按钮调用
            <code className="mx-1 text-foreground">winget</code>
            自动安装
            <strong className="mx-0.5 text-foreground">
              Build Tools for Visual Studio 2022
            </strong>
            （含 C++ 工作负载与 Windows 11 SDK）。不会安装完整的 Visual Studio IDE。
          </div>
          {detection.reason && (
            <div className="mt-1 font-mono text-[10px] text-muted-foreground/80 break-all">
              {detection.reason}
            </div>
          )}
          {!detection.winget_available && (
            <div className="mt-1.5 text-[11px]">
              <strong className="text-foreground">winget 不可用</strong>
              ，需要手动下载安装：
              <a
                href="https://aka.ms/vs/17/release/vs_BuildTools.exe"
                target="_blank"
                rel="noreferrer"
                className="ml-1 underline"
              >
                vs_BuildTools.exe
              </a>
            </div>
          )}
        </div>
        <Button
          size="sm"
          variant={succeeded ? "outline" : "default"}
          disabled={
            isRunning || isPending || succeeded || !detection.winget_available
          }
          onClick={onInstall}
        >
          {isRunning || isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : succeeded ? (
            <Check className="size-3" />
          ) : (
            <Download className="size-3" />
          )}
          {isRunning
            ? "安装中…"
            : succeeded
              ? "已安装"
              : failed
                ? "重试"
                : "一键安装"}
        </Button>
      </div>

      {(isRunning || lastLine) && (
        <div className="space-y-1">
          {lastLine && (
            <div className="font-mono text-[10px] text-muted-foreground/80 break-all">
              {lastLine}
            </div>
          )}
          {log.length > 1 && (
            <details className="text-[10px] text-muted-foreground/70">
              <summary className="cursor-pointer select-none">
                查看完整日志（{log.length} 行）
              </summary>
              <pre className="mt-1 max-h-48 overflow-auto rounded-[3px] border border-border/60 bg-muted/20 px-2 py-1.5 font-mono text-[10px] text-foreground/70">
                {log.join("\n")}
              </pre>
            </details>
          )}
        </div>
      )}

      {(failed || error) && (
        <div className="rounded-[3px] border border-destructive/40 bg-destructive/5 px-2.5 py-1.5 text-[11px] font-mono text-destructive break-all">
          {status?.error || error}
        </div>
      )}
    </div>
  )
}

import { useMemo, useState } from "react"
import { Sparkles, AlertTriangle, CheckCircle2, Loader2 } from "lucide-react"
import { toast } from "sonner"

import { api } from "@/lib/api"
import { fieldDisplay } from "@/lib/field-labels"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"

/**
 * 「智能推荐」按钮 + dialog。
 *
 * 把当前 form 草稿 + 用户填的意图 + 自动探到的硬件信息 POST 到
 * `/api/configs/llm-advise`,LLM 返回 patches + fullConfig + rationale。
 * Dialog 把对比展示给用户:左侧是字段级 patch 列表(每条都能点 accept),
 * 右侧是整份 yaml 的 JSON 对比;底部「全部接受」一键替换 form 草稿。
 *
 * 不在 form 里把字段值改掉 — 接受后通过 ``onApply`` 把 fullConfig
 * 回交给 form 的状态(整份替换 vs 局部 patch 由用户在 dialog 里决定)。
 */
export interface LlmAdvisorButtonProps {
  /** Current form value (camelCase, pre-validation). */
  currentCfg: Record<string, unknown>
  /** Apply a fully-replaced config (用户点「全部接受」时调用). */
  onApply: (next: Record<string, unknown>) => void
  /** Apply a single field patch (用户点单条 patch 的 accept 时调用). */
  onPatch: (field: string, value: unknown) => void
  className?: string
}

interface AdvisorResult {
  rationale: string
  patches: Array<{ field: string; value: unknown; reason: string }>
  fullConfig: Record<string, unknown>
  validationIssues: Array<{
    severity: "info" | "warning" | "error"
    field: string
    message: string
  }>
  providerId: string
  modelId: string
  elapsedMs: number
}

export function LlmAdvisorButton({
  currentCfg,
  onApply,
  onPatch,
  className,
}: LlmAdvisorButtonProps) {
  const [open, setOpen] = useState(false)
  const [intent, setIntent] = useState("")
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<AdvisorResult | null>(null)
  const [acceptedFields, setAcceptedFields] = useState<Set<string>>(new Set())

  const handleRequest = async () => {
    setBusy(true)
    setResult(null)
    setAcceptedFields(new Set())
    try {
      const out = await api.llmAdviseConfig({
        currentCfg,
        intent: intent.trim(),
      })
      setResult(out)
    } catch (e) {
      toast.error("智能推荐失败", {
        description: e instanceof Error ? e.message : String(e),
        duration: 12_000,
      })
    } finally {
      setBusy(false)
    }
  }

  const accept = (field: string, value: unknown) => {
    onPatch(field, value)
    setAcceptedFields((s) => new Set(s).add(field))
  }

  const acceptAll = () => {
    if (!result) return
    onApply(result.fullConfig)
    toast.success("已应用 LLM 推荐", {
      description: `${result.patches.length} 项更新已应用到表单。`,
    })
    setOpen(false)
  }

  return (
    <>
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => setOpen(true)}
        className={`gap-1.5 ${className ?? ""}`}
      >
        <Sparkles className="size-3.5" />
        智能推荐
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-[1100px] w-[95vw] max-h-[88vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="size-4" />
              智能推荐配置
            </DialogTitle>
            <DialogDescription>
              告诉 LLM 你想训练什么 / 硬件预算是多少;LLM 会根据 schema、显存、
              数据集统计给出一份建议。建议会先经过 schema 校验和字段冲突检测后才返回。
            </DialogDescription>
          </DialogHeader>

          <div className="flex-1 min-h-0 overflow-y-auto space-y-3">
            <div>
              <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground mb-1.5">
                训练意图
              </div>
              <textarea
                value={intent}
                onChange={(e) => setIntent(e.target.value)}
                placeholder="例:风格 LoRA,32GB 4090,大概 100 张图,想 4 小时内收敛,愿意牺牲一点细节换稳定。"
                className="w-full rounded-[4px] border border-input bg-background/82 px-3 py-2 text-sm font-mono resize-none min-h-[80px] focus:border-ring focus:outline-none focus:ring-3 focus:ring-ring/35"
                rows={3}
                disabled={busy}
              />
            </div>

            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setOpen(false)}
                disabled={busy}
              >
                关闭
              </Button>
              <Button
                type="button"
                size="sm"
                onClick={handleRequest}
                disabled={busy}
                className="gap-1.5"
              >
                {busy ? (
                  <>
                    <Loader2 className="size-3.5 animate-spin" />
                    LLM 思考中...
                  </>
                ) : (
                  <>
                    <Sparkles className="size-3.5" />
                    {result ? "重新生成" : "开始推荐"}
                  </>
                )}
              </Button>
            </div>

            {result && (
              <AdvisorResultPane
                result={result}
                accepted={acceptedFields}
                onAccept={accept}
                currentCfg={currentCfg}
              />
            )}
          </div>

          {result && (
            <div className="shrink-0 border-t border-border/60 pt-3 flex justify-between items-center">
              <div className="text-[11px] text-muted-foreground">
                {result.providerId}/{result.modelId} · {result.elapsedMs}ms · 共 {result.patches.length} 项更新
              </div>
              <Button
                type="button"
                onClick={acceptAll}
                disabled={busy}
                className="gap-1.5"
              >
                <CheckCircle2 className="size-3.5" />
                全部接受
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}

function AdvisorResultPane({
  result,
  accepted,
  onAccept,
  currentCfg,
}: {
  result: AdvisorResult
  accepted: Set<string>
  onAccept: (field: string, value: unknown) => void
  currentCfg: Record<string, unknown>
}) {
  const currentByField = useMemo(
    () => flattenForLookup(currentCfg),
    [currentCfg],
  )

  return (
    <div className="space-y-3">
      {result.rationale && (
        <div className="rounded-[4px] bg-muted/40 px-3 py-2 text-[12px] leading-relaxed">
          {result.rationale}
        </div>
      )}

      {result.validationIssues.length > 0 && (
        <div className="rounded-[4px] border border-amber-700/40 bg-amber-700/5 px-3 py-2 space-y-1">
          <div className="flex items-center gap-2 text-[11px] text-amber-700 dark:text-amber-400 font-medium">
            <AlertTriangle className="size-3.5" />
            LLM 建议触发了以下校验提示
          </div>
          <ul className="space-y-0.5">
            {result.validationIssues.map((iss, i) => {
              const fd = fieldDisplay(iss.field)
              return (
                <li key={i} className="text-[11px]">
                  <span className="opacity-60">[{iss.severity}]</span>{" "}
                  <span className="font-medium">{fd.label}</span>
                  {fd.hasLabel && (
                    <span className="ml-1 font-mono text-[10px] opacity-60">{fd.raw}</span>
                  )}
                  <span className="text-foreground/80">: {iss.message}</span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-3">
        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
            字段级补丁({result.patches.length})
          </div>
          <ul className="space-y-1.5">
            {result.patches.map((p) => {
              const isAccepted = accepted.has(p.field)
              const cur = currentByField.get(p.field)
              return (
                <li
                  key={p.field}
                  className={`rounded-[4px] border px-3 py-2 ${
                    isAccepted
                      ? "border-emerald-600/40 bg-emerald-600/5"
                      : "border-border/60 bg-background/50"
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <div className="flex-1 min-w-0">
                      <PatchFieldHeader field={p.field} />
                      <div className="mt-0.5 text-[11px] flex items-center gap-1.5">
                        <span className="font-mono text-muted-foreground line-through">
                          {fmtValue(cur)}
                        </span>
                        <span className="text-muted-foreground/60">→</span>
                        <span className="font-mono text-foreground">
                          {fmtValue(p.value)}
                        </span>
                      </div>
                      <div className="mt-1 text-[11px] text-muted-foreground leading-relaxed">
                        {p.reason}
                      </div>
                    </div>
                    {isAccepted ? (
                      <Badge variant="outline" className="gap-1 rounded-[2px] text-emerald-600 dark:text-emerald-400">
                        <CheckCircle2 className="size-2.5" />
                        已接受
                      </Badge>
                    ) : (
                      <Button
                        type="button"
                        size="xs"
                        variant="outline"
                        onClick={() => onAccept(p.field, p.value)}
                      >
                        接受
                      </Button>
                    )}
                  </div>
                </li>
              )
            })}
          </ul>
        </div>

        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
            完整配置预览
          </div>
          <pre className="rounded-[4px] border border-border/60 bg-muted/30 px-3 py-2 text-[11px] font-mono whitespace-pre-wrap break-words max-h-[320px] overflow-y-auto">
            {JSON.stringify(result.fullConfig, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  )
}

/** Flatten an object so dotted paths like ``backend.animaLora.networkDim``
 * can be read directly. We intentionally don't index into arrays — none of
 * the config-recommend paths target array elements. */
function flattenForLookup(
  obj: Record<string, unknown>,
  prefix = "",
  out: Map<string, unknown> = new Map(),
): Map<string, unknown> {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      flattenForLookup(v as Record<string, unknown>, key, out)
    } else {
      out.set(key, v)
    }
  }
  return out
}

function fmtValue(v: unknown): string {
  if (v === undefined) return "(unset)"
  if (v === null) return "null"
  if (typeof v === "string") return v.length > 40 ? `${v.slice(0, 37)}…` : v
  if (typeof v === "number" || typeof v === "boolean") return String(v)
  return JSON.stringify(v)
}

function PatchFieldHeader({ field }: { field: string }) {
  const fd = fieldDisplay(field)
  return (
    <div className="text-[11px] truncate">
      <span className="text-foreground/85 font-medium">{fd.label}</span>
      {fd.hasLabel && (
        <span className="ml-1 font-mono text-[10px] text-muted-foreground/70">
          {fd.raw}
        </span>
      )}
    </div>
  )
}

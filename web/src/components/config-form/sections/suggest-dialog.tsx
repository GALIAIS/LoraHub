/**
 * Suggest dialog — one-click hyperparameter recommendations.
 *
 * Open from the anima_lora section. User provides ``dataset_size``
 * + target type (character/style/concept); VRAM is sniffed from the
 * live ``/api/system/stats`` so the recommendation reflects the
 * machine actually running training. Apply commits the suggestion
 * via the Setter the parent section already uses, so the existing
 * dirty-state machinery picks it up.
 */
import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Loader2, Sparkles } from "lucide-react"
import { api, type HyperparamSuggestion } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { Setter } from "@/components/config-form/types"

type Target = "character" | "style" | "concept"

interface Props {
  /** Bound to the parent section's Setter so we can commit a batch
   *  of field updates in one click without each Row re-mounting. */
  set: Setter
  /** Backend label for the API call. Defaults to anima_lora since
   *  that's where this dialog lives, but the recommend endpoint
   *  accepts every backend the project supports. */
  backend?: "kohya" | "diffusion-pipe" | "anima_lora"
}

const TARGET_LABEL: Record<Target, string> = {
  character: "人物 (character)",
  style: "风格 (style)",
  concept: "概念 (concept)",
}

export function SuggestDialog({ set, backend = "anima_lora" }: Props) {
  const [open, setOpen] = useState(false)
  const [datasetSize, setDatasetSize] = useState<string>("100")
  const [target, setTarget] = useState<Target>("character")
  const [latestSuggestion, setLatestSuggestion] =
    useState<HyperparamSuggestion | null>(null)

  // Sniff VRAM from the live system snapshot. Shares the cache with
  // the dashboard / status-bar so opening the dialog typically gets an
  // already-fresh answer instead of triggering its own poll.
  const stats = useQuery({
    queryKey: ["system", "stats"],
    queryFn: api.getSystemStats,
    enabled: open,
    staleTime: 30_000,
  })

  const vramMb = useMemo(() => {
    const total = stats.data?.gpus?.[0]?.memory_total_bytes
    if (typeof total === "number" && total > 0) {
      return Math.round(total / (1024 * 1024))
    }
    return 16 * 1024 // sane default for the no-GPU dev case
  }, [stats.data])

  const recommend = useMutation({
    mutationFn: () =>
      api.recommendHyperparams({
        dataset_size: parseInt(datasetSize, 10) || 0,
        gpu_vram_mb: vramMb,
        backend,
        target,
      }),
    onSuccess: (resp) => {
      setLatestSuggestion(resp.suggestion)
    },
  })

  // Reset transient state when the dialog closes — a fresh open
  // shouldn't show last time's preview.
  useEffect(() => {
    if (!open) {
      setLatestSuggestion(null)
      recommend.reset()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const apply = () => {
    if (!latestSuggestion) return
    const s = latestSuggestion
    // Map flat suggestion → nested config paths the form uses. We
    // mirror the field names the AnimaLora section binds to (see
    // backend-anima-lora.tsx); fields that don't have a UI binding
    // (``extra_flags``) get committed under animaLora's overrides
    // bag so the trainer still picks them up via cli args.
    const path = ["backend", "animaLora"] as const
    set([...path, "trainBatchSize"], s.batch_size)
    set([...path, "gradientAccumulationSteps"], s.gradient_accumulation_steps)
    set([...path, "unetLr"], s.learning_rate)
    set([...path, "networkDim"], s.network_dim)
    set([...path, "networkAlpha"], s.network_alpha)
    set([...path, "maxTrainEpochs"], s.max_train_epochs)
    set([...path, "optimizerType"], s.optimizer_type)
    if (s.extra_flags?.gradient_checkpointing) {
      set([...path, "gradientCheckpointing"], true)
    }
    setOpen(false)
  }

  return (
    <>
      <Button
        size="sm"
        variant="outline"
        className="gap-1.5"
        onClick={() => setOpen(true)}
      >
        <Sparkles className="size-3" />
        生成参数方案
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-[520px]">
          <DialogHeader>
            <DialogTitle>生成超参方案</DialogTitle>
            <DialogDescription>
              根据数据集大小、目标类型和当前 GPU 显存生成一组初始参数。
              会覆盖已配置的 batch / 学习率 / rank / epochs / 优化器。
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div className="grid grid-cols-[8rem_1fr] gap-x-4 gap-y-3 items-center">
              <Label htmlFor="ds-size" className="text-xs">
                数据集图片数
              </Label>
              <Input
                id="ds-size"
                type="number"
                min={1}
                value={datasetSize}
                onChange={(e) => setDatasetSize(e.target.value)}
                className="font-mono"
              />

              <Label className="text-xs">训练目标</Label>
              <Select
                items={[
                  { value: "character", label: TARGET_LABEL.character },
                  { value: "style", label: TARGET_LABEL.style },
                  { value: "concept", label: TARGET_LABEL.concept },
                ]}
                value={target}
                onValueChange={(v) => v && setTarget(v as Target)}
              >
                <SelectTrigger className="text-xs h-9">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="character">{TARGET_LABEL.character}</SelectItem>
                  <SelectItem value="style">{TARGET_LABEL.style}</SelectItem>
                  <SelectItem value="concept">{TARGET_LABEL.concept}</SelectItem>
                </SelectContent>
              </Select>

              <Label className="text-xs">GPU 显存</Label>
              <div className="text-xs font-mono text-muted-foreground">
                {(vramMb / 1024).toFixed(1)} GB ({vramMb.toLocaleString()} MB)
              </div>
            </div>

            {recommend.isError && (
              <div className="text-xs text-destructive font-mono">
                {(recommend.error as Error).message}
              </div>
            )}

            {latestSuggestion && (
              <div className="rounded-[4px] border border-border/60 bg-muted/30 p-3 space-y-2">
                <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                  参数方案
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                  <KV k="batch_size" v={latestSuggestion.batch_size} />
                  <KV k="grad_accum" v={latestSuggestion.gradient_accumulation_steps} />
                  <KV k="learning_rate" v={latestSuggestion.learning_rate.toExponential(2)} />
                  <KV k="network_dim" v={latestSuggestion.network_dim} />
                  <KV k="network_alpha" v={latestSuggestion.network_alpha} />
                  <KV k="max_epochs" v={latestSuggestion.max_train_epochs} />
                  <KV k="optimizer" v={latestSuggestion.optimizer_type} />
                </div>
                <details className="text-[11px]">
                  <summary className="cursor-pointer text-muted-foreground hover:text-foreground select-none">
                    推理依据 ({latestSuggestion.rationale.length})
                  </summary>
                  <ul className="mt-1.5 space-y-1 text-muted-foreground/85 leading-relaxed">
                    {latestSuggestion.rationale.map((r, i) => (
                      <li key={i} className="before:content-['—'] before:mr-2">
                        {r}
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            )}
          </div>

          <DialogFooter>
            {!latestSuggestion ? (
              <Button
                size="sm"
                onClick={() => recommend.mutate()}
                disabled={recommend.isPending}
              >
                {recommend.isPending && <Loader2 className="size-3 animate-spin" />}
                生成方案
              </Button>
            ) : (
              <>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => recommend.mutate()}
                  disabled={recommend.isPending}
                >
                  重新生成
                </Button>
                <Button size="sm" onClick={apply}>
                  应用到表单
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

function KV({ k, v }: { k: string; v: string | number }) {
  return (
    <div className="flex items-baseline gap-2 min-w-0">
      <span className="text-muted-foreground/80">{k}</span>
      <code className="font-mono text-foreground truncate">{v}</code>
    </div>
  )
}

/**
 * Lock / risk badges for anima_lora upstream-default fields.
 *
 * The anima_lora backend exposes every base.toml field for editor
 * completeness, but several of them can't actually be changed at the
 * train.py argparse level — base.toml hard-codes them ``= true`` and
 * upstream offers no ``--no_<x>`` reverse flag, or the value is baked
 * into the static-shape compile path. We render a colored badge next
 * to the field's label so the user immediately sees:
 *
 *   🔒 LOCKED — silently ignored upstream. Editing has no effect.
 *   ⚠️ RISKY  — works, but breaks something obvious if changed.
 *
 * The accompanying tooltip text comes from
 * ``lorahub.core.backends.anima_lora.compiler.LOCKED_FIELDS`` (kept
 * mirrored here so the badge text stays in sync with the compiler's
 * warning log without an extra round-trip).
 */
import type { ComponentProps } from "react"
import { Lock, AlertTriangle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export type LockKind = "locked_true" | "locked_value" | "risky"

export interface LockMeta {
  kind: LockKind
  /** Chinese-language reason surfaced on hover. */
  reason: string
}

/**
 * Field-name → lock metadata map. Mirrored from
 * `lorahub/core/backends/anima_lora/compiler.py::LOCKED_FIELDS`.
 * Keep in sync — tests on the python side assert every key present
 * here also has an entry over there (TODO: cross-reference test).
 */
export const ANIMA_LORA_LOCKS: Record<string, LockMeta> = {
  maskedLoss: {
    kind: "locked_true",
    reason: "Anima 训练流程要求 masked loss。",
  },
  torchCompile: {
    kind: "locked_true",
    reason: "Anima static-shape 训练路径要求 torch.compile。",
  },
  skipCacheCheck: {
    kind: "locked_true",
    reason: "缓存校验保持跳过。",
  },
  dataloaderPinMemory: {
    kind: "locked_true",
    reason: "DataLoader pin_memory 固定开启。",
  },
  enableBucket: {
    kind: "locked_true",
    reason: "Anima static-shape compile 要求 bucketing。",
  },
  cacheLatents: {
    kind: "locked_true",
    reason: "训练流程要求 latent 缓存。",
  },
  cacheLatentsToDisk: {
    kind: "locked_true",
    reason: "latent 缓存写入磁盘。",
  },
  cacheTextEncoderOutputs: {
    kind: "locked_true",
    reason: "训练流程要求 text encoder 缓存。",
  },
  cacheTextEncoderOutputsToDisk: {
    kind: "locked_true",
    reason: "text encoder 缓存写入磁盘。",
  },
  cacheLlmAdapterOutputs: {
    kind: "locked_true",
    reason: "训练流程要求 LLM adapter 缓存。",
  },
  staticTokenCount: {
    kind: "risky",
    reason: "修改 static_token_count 后需要重建 dataset 缓存。",
  },
  vaeChunkSize: {
    kind: "locked_value",
    reason: "VAE chunk size 固定为 64。",
  },
  captionExtension: {
    kind: "locked_value",
    reason: "caption 后缀固定为 .txt。",
  },
  saveModelAs: {
    kind: "locked_value",
    reason: "保存格式固定为 safetensors。",
  },
  vaeDisableCache: {
    kind: "risky",
    reason: "影响 VAE encode 路径。",
  },
  noHalfVae: {
    kind: "risky",
    reason: "影响 VAE 精度与显存。",
  },
  trimCrossattnKv: {
    kind: "risky",
    reason: "影响 cross-attention KV 长度。",
  },
  savePrecision: {
    kind: "risky",
    reason: "影响 checkpoint 体积与 dtype。",
  },
  persistentDataLoaderWorkers: {
    kind: "risky",
    reason: "影响 DataLoader worker 生命周期。",
  },
  keepTokens: {
    kind: "risky",
    reason: "影响 caption shuffle 保留 token。",
  },
  validationSplitNum: {
    kind: "risky",
    reason: "影响 CMMD / val_loss 验证。",
  },
}

const KIND_TONE: Record<LockKind, string> = {
  locked_true:
    "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-300",
  locked_value:
    "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-300",
  risky:
    "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
}

const KIND_LABEL: Record<LockKind, string> = {
  locked_true: "锁定",
  locked_value: "锁定值",
  risky: "谨慎",
}

interface LockBadgeProps extends ComponentProps<"span"> {
  meta: LockMeta
}

export function LockBadge({ meta, className, ...rest }: LockBadgeProps) {
  const Icon = meta.kind === "risky" ? AlertTriangle : Lock
  return (
    <Badge
      variant="outline"
      className={cn(
        "rounded-[2px] uppercase text-[9px] tracking-[0.1em] font-mono",
        KIND_TONE[meta.kind],
        className,
      )}
      title={meta.reason}
      {...rest}
    >
      <Icon className="size-2.5" />
      {KIND_LABEL[meta.kind]}
    </Badge>
  )
}

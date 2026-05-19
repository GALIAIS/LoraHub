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
    reason: "Anima 训练管线硬依赖 masked loss;关掉是无效操作。",
  },
  torchCompile: {
    kind: "locked_true",
    reason:
      "torch.compile 是 static_token_count 性能收益的前提;upstream 训练循环假定开启。",
  },
  skipCacheCheck: {
    kind: "locked_true",
    reason: "缓存校验跳过对训练正确性无影响,只影响启动速度;关掉无意义。",
  },
  dataloaderPinMemory: {
    kind: "locked_true",
    reason: "DataLoader pin_memory 一直开;upstream 没提供反向 flag。",
  },
  enableBucket: {
    kind: "locked_true",
    reason: "constant-token bucketing 是 Anima static-shape compile 的硬约束。",
  },
  cacheLatents: {
    kind: "locked_true",
    reason: "anima_lora 训练流程依赖预计算 latent 缓存,关掉训练会失败。",
  },
  cacheLatentsToDisk: {
    kind: "locked_true",
    reason: "缓存必须落盘以避免每次 epoch 重算。",
  },
  cacheTextEncoderOutputs: {
    kind: "locked_true",
    reason: "TE 输出必缓存,否则训练时拖累 Qwen3 的 forward。",
  },
  cacheTextEncoderOutputsToDisk: {
    kind: "locked_true",
    reason: "TE 缓存必须落盘。",
  },
  cacheLlmAdapterOutputs: {
    kind: "locked_true",
    reason: "LLM adapter 输出必缓存。",
  },
  staticTokenCount: {
    kind: "locked_value",
    reason:
      "Anima DiT torch.compile 路径锁死 4096(constant-token bucket map);其它值会引发每个分辨率重新编译。",
  },
  vaeChunkSize: {
    kind: "locked_value",
    reason: "QwenImage VAE memory layout 锁死 64;改了多半 OOM 或无收益。",
  },
  captionExtension: {
    kind: "locked_value",
    reason: "数据 pipeline 写死 .txt 后缀;改了所有图片会被跳过且无警告。",
  },
  saveModelAs: {
    kind: "locked_value",
    reason: "Anima 只能加载 safetensors;其它格式无法 round-trip。",
  },
  vaeDisableCache: {
    kind: "risky",
    reason: "改成 false 会拖慢 VAE encode ~30%,但与官方 VAE 行为一致。",
  },
  noHalfVae: {
    kind: "risky",
    reason: "true 半精度 VAE 省显存,但偶尔在边缘数据集上产生 NaN。",
  },
  trimCrossattnKv: {
    kind: "risky",
    reason: "true 启用 KV trimming(短 caption 加速 ~10-15%),但需匹配 caption 长度分布。",
  },
  savePrecision: {
    kind: "risky",
    reason:
      "fp32 是双倍体积无质量收益;fp16 略小但偶有量化损失;bf16 是 upstream 默认。",
  },
  persistentDataLoaderWorkers: {
    kind: "risky",
    reason: "true 减少 epoch 边界 stall,但长跑可能泄漏 file handle。",
  },
  keepTokens: {
    kind: "risky",
    reason:
      "anima 训练模板把 trigger / character / character-feature 三件放前 3 位;改了 trigger word 不再可靠。",
  },
  validationSplitNum: {
    kind: "risky",
    reason: "0 = 关 CMMD 验证(val_loss 不再更新)。",
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

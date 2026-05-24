/**
 * Attention backend section.
 *
 * `training` selects the kernel used by the trainer's forward + backward
 * pass. We dynamically grey out kernels the local GPU can't run by querying
 * `/api/system/attention-backends` (server-side gating mirrors
 * `attention_backends_for_gpu` in `system_stats.py`).
 */
import { memo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { api } from "@/lib/api"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { Row, ToggleSwitch, useReadOnly } from "../widgets"

// Static labels for every backend in the canonical set. Kept here (not in
// `options.ts`) so the requirement note ("需要 sm_X+") lives next to the GPU
// gating context. New backends added to `ALL_ATTENTION_BACKENDS` server-side
// just need a new entry here.
const BACKEND_LABELS: Record<
  string,
  { label: string; hint: string; requirement: string }
> = {
  auto: {
    label: "Auto · 自动选择",
    hint: "由 kohya / dp 自行决定（多数情况下为 SDPA）",
    requirement: "",
  },
  torch: {
    label: "Torch · 朴素 PyTorch",
    hint: "诊断用，速度最慢",
    requirement: "",
  },
  sdpa: {
    label: "SDPA · F.scaled_dot_product_attention",
    hint: "PyTorch 原生算子，兼容性最好",
    requirement: "",
  },
  flex: {
    label: "FlexAttention · torch.nn.attention.flex_attention",
    hint: "PyTorch 2.5+ 实验性算子",
    requirement: "",
  },
  xformers: {
    label: "xFormers",
    hint: "Meta 的内核库，需要单独 wheel",
    requirement: "需要 sm_70+",
  },
  flash: {
    label: "FlashAttention 2",
    hint: "Ampere / Ada / Hopper 通用",
    requirement: "需要 sm_80+（Ampere / Ada / Hopper）",
  },
  flash3: {
    label: "FlashAttention 3",
    hint: "Hopper 专属新内核，速度最快",
    requirement: "需要 sm_90（Hopper · H100 / H200）",
  },
  flash4: {
    label: "FlashAttention 4 · Beta",
    hint: "Hopper / Blackwell Beta 内核",
    requirement: "需要 sm_90 或 sm_100+（Hopper / Blackwell）",
  },
}

export const AttentionFields = memo(function AttentionFields({
  value = {},
  set,
}: {
  value: ConfigFormValue["attention"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const training = v.training ?? "auto"
  const readOnly = useReadOnly()

  const backends = useQuery({
    queryKey: ["attention-backends"],
    queryFn: api.getAttentionBackends,
    staleTime: 60_000,
  })

  // Use server-driven `all` when available so the dropdown stays in sync if
  // the schema enum grows on the backend; fall back to the labels we know.
  const allBackends = backends.data?.all ?? Object.keys(BACKEND_LABELS)
  const supported = new Set(backends.data?.supported ?? allBackends)

  const capLabel = (() => {
    if (backends.isLoading) return "正在检测 GPU…"
    const cap = backends.data?.compute_capability
    if (cap) return `检测到 NVIDIA GPU · 计算能力 sm_${cap.replace(".", "")}`
    return "未检测到 NVIDIA GPU（仅显示 PyTorch 原生内核）"
  })()

  return (
    <>
      <div className="-mt-1 mb-1 text-[11px] text-muted-foreground/80">
        {capLabel}
      </div>
      <Row label="训练注意力内核" description="影响训练 forward/backward 的速度与显存。">
        <Select
          items={allBackends.map((b) => ({ value: b, label: b }))}
          value={training}
          onValueChange={(next) =>
            next && set(["attention", "training"], next)
          }
          disabled={readOnly}
        >
          <SelectTrigger className="w-72">
            <SelectValue placeholder="选择内核" />
          </SelectTrigger>
          <SelectContent>
            {allBackends.map((b) => {
              const meta = BACKEND_LABELS[b] ?? {
                label: b,
                hint: "",
                requirement: "",
              }
              const isSupported = supported.has(b)
              const subtitle = isSupported
                ? meta.hint
                : meta.requirement || "本机 GPU 不支持"
              return (
                <SelectItem
                  key={b}
                  value={b}
                  disabled={!isSupported}
                  title={isSupported ? meta.hint : meta.requirement}
                >
                  <span className="flex flex-col">
                    <span className="text-xs">{meta.label}</span>
                    {subtitle && (
                      <span className="text-[10px] text-muted-foreground">
                        {subtitle}
                      </span>
                    )}
                  </span>
                </SelectItem>
              )
            })}
          </SelectContent>
        </Select>
      </Row>
      {training === "xformers" && (
        <Row
          label="拆分注意力"
          description="kohya 的 --split_attn；与 xformers 配合使用以降低峰值显存。"
        >
          <ToggleSwitch
            checked={v.split ?? false}
            onCheckedChange={(b) => set(["attention", "split"], b)}
          />
        </Row>
      )}
    </>
  )
})

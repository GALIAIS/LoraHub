/**
 * ArchPathsConfig editor — per-component checkpoint paths and arch-specific
 * memory / dropout / token-length knobs.
 *
 * Most arches don't use every field; the form intentionally exposes the full
 * union so any user can fill in the bits their arch needs. The wrapping
 * <Section> in `index.tsx` decides default-open semantics (collapsed by
 * default, but auto-expanded when the chosen arch typically wants these).
 */
import { memo } from "react"
import { T5_DTYPE_OPTIONS } from "../options"
import type { ArchPathsValue, ErrorMap, Setter } from "../types"
import {
  EnumSelect,
  FloatInput,
  IntInput,
  PathInput,
  Row,
  TextInput,
  ToggleSwitch,
} from "../widgets"

export const ArchPathsFields = memo(function ArchPathsFields({
  value,
  set,
  errorMap,
  arch,
}: {
  value: ArchPathsValue | undefined
  set: Setter
  errorMap: ErrorMap
  arch: string
}) {
  const v = value ?? {}
  const p = (k: keyof ArchPathsValue) => `base_model.arch_paths.${k}` as const
  // Per-arch hint — purely informational; the schema doesn't actually
  // enforce a subset, so we just nudge users towards the relevant inputs.
  const hint = (() => {
    switch (arch) {
      case "flux":
      case "flux2":
        return "FLUX 通常需要 clip_l / t5xxl / ae 三个组件路径。"
      case "sd3":
        return "SD3 通常需要 clip_l / clip_g / t5xxl 三个组件路径。"
      case "anima":
        return "Anima 通常需要 transformer / qwen3 / t5_tokenizer / llm_adapter。"
      case "hunyuan_image":
        return "HunyuanImage 通常需要 byt5 / text_encoder / transformer。"
      default:
        return "仅在使用多文件 bundle 的 arch 上需要填写；其它 arch 留空即可。"
    }
  })()

  return (
    <>
      <div className="-mt-1 mb-1 text-[11px] text-muted-foreground/80">
        {hint}
      </div>

      <SubGroup label="FLUX / SD3 组件">
        <Row label="clip_l" errors={errorMap.get(p("clip_l"))}>
          <PathInput
            value={v.clip_l ?? ""}
            onChange={(s) => set(["base_model", "arch_paths", "clip_l"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="clip_g" errors={errorMap.get(p("clip_g"))}>
          <PathInput
            value={v.clip_g ?? ""}
            onChange={(s) => set(["base_model", "arch_paths", "clip_g"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="t5xxl" errors={errorMap.get(p("t5xxl"))}>
          <PathInput
            value={v.t5xxl ?? ""}
            onChange={(s) => set(["base_model", "arch_paths", "t5xxl"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="ae" description="FLUX autoencoder。" errors={errorMap.get(p("ae"))}>
          <PathInput
            value={v.ae ?? ""}
            onChange={(s) => set(["base_model", "arch_paths", "ae"], s || null)}
            placeholder="（可选）"
          />
        </Row>
      </SubGroup>

      <SubGroup label="Anima 组件">
        <Row label="qwen3" errors={errorMap.get(p("qwen3"))}>
          <PathInput
            value={v.qwen3 ?? ""}
            onChange={(s) => set(["base_model", "arch_paths", "qwen3"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="llm_adapter" errors={errorMap.get(p("llm_adapter"))}>
          <PathInput
            value={v.llm_adapter ?? ""}
            onChange={(s) =>
              set(["base_model", "arch_paths", "llm_adapter"], s || null)
            }
            placeholder="（可选）"
          />
        </Row>
        <Row label="t5_tokenizer" errors={errorMap.get(p("t5_tokenizer"))}>
          <PathInput
            value={v.t5_tokenizer ?? ""}
            onChange={(s) =>
              set(["base_model", "arch_paths", "t5_tokenizer"], s || null)
            }
            placeholder="（可选）"
          />
        </Row>
      </SubGroup>

      <SubGroup label="通用组件（Anima / Wan / HunyuanImage / chroma）">
        <Row label="transformer" errors={errorMap.get(p("transformer"))}>
          <PathInput
            value={v.transformer ?? ""}
            onChange={(s) =>
              set(["base_model", "arch_paths", "transformer"], s || null)
            }
            placeholder="（可选）"
          />
        </Row>
        <Row label="text_encoder" errors={errorMap.get(p("text_encoder"))}>
          <PathInput
            value={v.text_encoder ?? ""}
            onChange={(s) =>
              set(["base_model", "arch_paths", "text_encoder"], s || null)
            }
            placeholder="（可选）"
          />
        </Row>
        <Row label="llm" description="Anima Qwen3 / HunyuanVideo LLM。" errors={errorMap.get(p("llm"))}>
          <PathInput
            value={v.llm ?? ""}
            onChange={(s) => set(["base_model", "arch_paths", "llm"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="byt5" description="HunyuanImage byT5。" errors={errorMap.get(p("byt5"))}>
          <PathInput
            value={v.byt5 ?? ""}
            onChange={(s) => set(["base_model", "arch_paths", "byt5"], s || null)}
            placeholder="（可选）"
          />
        </Row>
      </SubGroup>

      <SubGroup label="Token 长度上限">
        <Row label="t5xxl_max_token_length" errors={errorMap.get(p("t5xxl_max_token_length"))}>
          <IntInput
            min={1}
            value={v.t5xxl_max_token_length ?? null}
            onChange={(n) =>
              set(["base_model", "arch_paths", "t5xxl_max_token_length"], n)
            }
            placeholder="（默认）"
          />
        </Row>
        <Row label="qwen3_max_token_length" errors={errorMap.get(p("qwen3_max_token_length"))}>
          <IntInput
            min={1}
            value={v.qwen3_max_token_length ?? null}
            onChange={(n) =>
              set(["base_model", "arch_paths", "qwen3_max_token_length"], n)
            }
            placeholder="（默认）"
          />
        </Row>
        <Row label="t5_max_token_length" errors={errorMap.get(p("t5_max_token_length"))}>
          <IntInput
            min={1}
            value={v.t5_max_token_length ?? null}
            onChange={(n) =>
              set(["base_model", "arch_paths", "t5_max_token_length"], n)
            }
            placeholder="（默认）"
          />
        </Row>
      </SubGroup>

      <SubGroup label="Attention 掩码与 dropout（FLUX / SD3）">
        <Row label="apply_t5_attn_mask" description="对 T5 输出施加 attention mask。">
          <ToggleSwitch
            checked={v.apply_t5_attn_mask ?? false}
            onCheckedChange={(b) =>
              set(["base_model", "arch_paths", "apply_t5_attn_mask"], b)
            }
          />
        </Row>
        <Row label="apply_lg_attn_mask" description="对 CLIP-L/G 输出施加 attention mask。">
          <ToggleSwitch
            checked={v.apply_lg_attn_mask ?? false}
            onCheckedChange={(b) =>
              set(["base_model", "arch_paths", "apply_lg_attn_mask"], b)
            }
          />
        </Row>
        <Row label="t5_dropout_rate" errors={errorMap.get(p("t5_dropout_rate"))}>
          <FloatInput
            step={0.05}
            value={v.t5_dropout_rate ?? 0}
            onChange={(n) =>
              set(["base_model", "arch_paths", "t5_dropout_rate"], n ?? 0)
            }
          />
        </Row>
        <Row label="clip_l_dropout_rate" errors={errorMap.get(p("clip_l_dropout_rate"))}>
          <FloatInput
            step={0.05}
            value={v.clip_l_dropout_rate ?? 0}
            onChange={(n) =>
              set(["base_model", "arch_paths", "clip_l_dropout_rate"], n ?? 0)
            }
          />
        </Row>
        <Row label="clip_g_dropout_rate" errors={errorMap.get(p("clip_g_dropout_rate"))}>
          <FloatInput
            step={0.05}
            value={v.clip_g_dropout_rate ?? 0}
            onChange={(n) =>
              set(["base_model", "arch_paths", "clip_g_dropout_rate"], n ?? 0)
            }
          />
        </Row>
        <Row
          label="pos_emb_random_crop_rate"
          description="SD3 位置编码随机 crop 概率。"
          errors={errorMap.get(p("pos_emb_random_crop_rate"))}
        >
          <FloatInput
            step={0.05}
            value={v.pos_emb_random_crop_rate ?? 0}
            onChange={(n) =>
              set(["base_model", "arch_paths", "pos_emb_random_crop_rate"], n ?? 0)
            }
          />
        </Row>
        <Row label="enable_scaled_pos_embed" description="SD3 启用缩放位置编码。">
          <ToggleSwitch
            checked={v.enable_scaled_pos_embed ?? false}
            onCheckedChange={(b) =>
              set(["base_model", "arch_paths", "enable_scaled_pos_embed"], b)
            }
          />
        </Row>
      </SubGroup>

      <SubGroup label="FLUX guidance / TE 设备 / VAE 内存">
        <Row
          label="guidance_scale"
          description="FLUX dev 蒸馏版需要烘焙到 LoRA 的 guidance。留空跳过。"
          errors={errorMap.get(p("guidance_scale"))}
        >
          <FloatInput
            step={0.1}
            value={v.guidance_scale ?? null}
            onChange={(n) => set(["base_model", "arch_paths", "guidance_scale"], n)}
            placeholder="（默认）"
          />
        </Row>
        <Row label="t5xxl_device" description="例如 cuda / cuda:1 / cpu。">
          <TextInput
            className="w-48"
            value={v.t5xxl_device ?? ""}
            onChange={(s) =>
              set(["base_model", "arch_paths", "t5xxl_device"], s || null)
            }
            placeholder="（默认）"
          />
        </Row>
        <Row label="t5xxl_dtype">
          <EnumSelect
            value={v.t5xxl_dtype ?? ""}
            onChange={(s) =>
              set(["base_model", "arch_paths", "t5xxl_dtype"], s || null)
            }
            options={T5_DTYPE_OPTIONS}
          />
        </Row>
        <Row
          label="vae_chunk_size"
          description="Anima / HunyuanImage / Wan VAE 分块大小（节省显存）。"
          errors={errorMap.get(p("vae_chunk_size"))}
        >
          <IntInput
            min={1}
            value={v.vae_chunk_size ?? null}
            onChange={(n) =>
              set(["base_model", "arch_paths", "vae_chunk_size"], n)
            }
            placeholder="（默认）"
          />
        </Row>
        <Row label="vae_disable_cache" description="禁用 VAE 输出缓存。">
          <ToggleSwitch
            checked={v.vae_disable_cache ?? false}
            onCheckedChange={(b) =>
              set(["base_model", "arch_paths", "vae_disable_cache"], b)
            }
          />
        </Row>
        <Row label="text_encoder_cpu" description="把文本编码器固定在 CPU。">
          <ToggleSwitch
            checked={v.text_encoder_cpu ?? false}
            onCheckedChange={(b) =>
              set(["base_model", "arch_paths", "text_encoder_cpu"], b)
            }
          />
        </Row>
      </SubGroup>
    </>
  )
})

const SubGroup = memo(function SubGroup({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="rounded-[4px] border border-border/40 bg-muted/10 p-3 space-y-3.5">
      <div className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.18em]">
        {label}
      </div>
      {children}
    </div>
  )
})

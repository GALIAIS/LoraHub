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
  Row,
  TextInput,
  ToggleSwitch,
} from "../widgets"
import { ModelPathPicker } from "../widgets-model-picker"

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
  const p = (k: keyof ArchPathsValue) => `baseModel.archPaths.${k}` as const
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
        <Row label="clipL" errors={errorMap.get(p("clipL"))}>
          <ModelPathPicker
            value={v.clipL ?? ""}
            onChange={(s) => set(["baseModel", "archPaths", "clipL"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="clipG" errors={errorMap.get(p("clipG"))}>
          <ModelPathPicker
            value={v.clipG ?? ""}
            onChange={(s) => set(["baseModel", "archPaths", "clipG"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="t5xxl" errors={errorMap.get(p("t5xxl"))}>
          <ModelPathPicker
            value={v.t5xxl ?? ""}
            onChange={(s) => set(["baseModel", "archPaths", "t5xxl"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="ae" description="FLUX autoencoder。" errors={errorMap.get(p("ae"))}>
          <ModelPathPicker
            value={v.ae ?? ""}
            onChange={(s) => set(["baseModel", "archPaths", "ae"], s || null)}
            placeholder="（可选）"
          />
        </Row>
      </SubGroup>

      <SubGroup label="Anima 组件">
        <Row label="qwen3" errors={errorMap.get(p("qwen3"))}>
          <ModelPathPicker
            value={v.qwen3 ?? ""}
            onChange={(s) => set(["baseModel", "archPaths", "qwen3"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="llmAdapter" errors={errorMap.get(p("llmAdapter"))}>
          <ModelPathPicker
            value={v.llmAdapter ?? ""}
            onChange={(s) =>
              set(["baseModel", "archPaths", "llmAdapter"], s || null)
            }
            placeholder="（可选）"
          />
        </Row>
        <Row label="t5Tokenizer" errors={errorMap.get(p("t5Tokenizer"))}>
          <ModelPathPicker
            value={v.t5Tokenizer ?? ""}
            onChange={(s) =>
              set(["baseModel", "archPaths", "t5Tokenizer"], s || null)
            }
            placeholder="（可选）"
          />
        </Row>
      </SubGroup>

      <SubGroup label="通用组件（Anima / Wan / HunyuanImage / chroma）">
        <Row label="transformer" errors={errorMap.get(p("transformer"))}>
          <ModelPathPicker
            value={v.transformer ?? ""}
            onChange={(s) =>
              set(["baseModel", "archPaths", "transformer"], s || null)
            }
            placeholder="（可选）"
          />
        </Row>
        <Row label="textEncoder" errors={errorMap.get(p("textEncoder"))}>
          <ModelPathPicker
            value={v.textEncoder ?? ""}
            onChange={(s) =>
              set(["baseModel", "archPaths", "textEncoder"], s || null)
            }
            placeholder="（可选）"
          />
        </Row>
        <Row label="llm" description="Anima Qwen3 / HunyuanVideo LLM。" errors={errorMap.get(p("llm"))}>
          <ModelPathPicker
            value={v.llm ?? ""}
            onChange={(s) => set(["baseModel", "archPaths", "llm"], s || null)}
            placeholder="（可选）"
          />
        </Row>
        <Row label="byt5" description="HunyuanImage byT5。" errors={errorMap.get(p("byt5"))}>
          <ModelPathPicker
            value={v.byt5 ?? ""}
            onChange={(s) => set(["baseModel", "archPaths", "byt5"], s || null)}
            placeholder="（可选）"
          />
        </Row>
      </SubGroup>

      <SubGroup label="Token 长度上限">
        <Row label="t5xxlMaxTokenLength" errors={errorMap.get(p("t5xxlMaxTokenLength"))}>
          <IntInput
            min={1}
            value={v.t5xxlMaxTokenLength ?? null}
            onChange={(n) =>
              set(["baseModel", "archPaths", "t5xxlMaxTokenLength"], n)
            }
            placeholder="（默认）"
          />
        </Row>
        <Row label="qwen3MaxTokenLength" errors={errorMap.get(p("qwen3MaxTokenLength"))}>
          <IntInput
            min={1}
            value={v.qwen3MaxTokenLength ?? null}
            onChange={(n) =>
              set(["baseModel", "archPaths", "qwen3MaxTokenLength"], n)
            }
            placeholder="（默认）"
          />
        </Row>
        <Row label="t5MaxTokenLength" errors={errorMap.get(p("t5MaxTokenLength"))}>
          <IntInput
            min={1}
            value={v.t5MaxTokenLength ?? null}
            onChange={(n) =>
              set(["baseModel", "archPaths", "t5MaxTokenLength"], n)
            }
            placeholder="（默认）"
          />
        </Row>
      </SubGroup>

      <SubGroup label="Attention 掩码与 dropout（FLUX / SD3）">
        <Row label="applyT5AttnMask" description="对 T5 输出施加 attention mask。">
          <ToggleSwitch
            checked={v.applyT5AttnMask ?? false}
            onCheckedChange={(b) =>
              set(["baseModel", "archPaths", "applyT5AttnMask"], b)
            }
          />
        </Row>
        <Row label="applyLgAttnMask" description="对 CLIP-L/G 输出施加 attention mask。">
          <ToggleSwitch
            checked={v.applyLgAttnMask ?? false}
            onCheckedChange={(b) =>
              set(["baseModel", "archPaths", "applyLgAttnMask"], b)
            }
          />
        </Row>
        <Row label="t5DropoutRate" errors={errorMap.get(p("t5DropoutRate"))}>
          <FloatInput
            step={0.05}
            value={v.t5DropoutRate ?? 0}
            onChange={(n) =>
              set(["baseModel", "archPaths", "t5DropoutRate"], n ?? 0)
            }
          />
        </Row>
        <Row label="clipLDropoutRate" errors={errorMap.get(p("clipLDropoutRate"))}>
          <FloatInput
            step={0.05}
            value={v.clipLDropoutRate ?? 0}
            onChange={(n) =>
              set(["baseModel", "archPaths", "clipLDropoutRate"], n ?? 0)
            }
          />
        </Row>
        <Row label="clipGDropoutRate" errors={errorMap.get(p("clipGDropoutRate"))}>
          <FloatInput
            step={0.05}
            value={v.clipGDropoutRate ?? 0}
            onChange={(n) =>
              set(["baseModel", "archPaths", "clipGDropoutRate"], n ?? 0)
            }
          />
        </Row>
        <Row
          label="posEmbRandomCropRate"
          description="SD3 位置编码随机 crop 概率。"
          errors={errorMap.get(p("posEmbRandomCropRate"))}
        >
          <FloatInput
            step={0.05}
            value={v.posEmbRandomCropRate ?? 0}
            onChange={(n) =>
              set(["baseModel", "archPaths", "posEmbRandomCropRate"], n ?? 0)
            }
          />
        </Row>
        <Row label="enableScaledPosEmbed" description="SD3 启用缩放位置编码。">
          <ToggleSwitch
            checked={v.enableScaledPosEmbed ?? false}
            onCheckedChange={(b) =>
              set(["baseModel", "archPaths", "enableScaledPosEmbed"], b)
            }
          />
        </Row>
      </SubGroup>

      <SubGroup label="FLUX guidance / TE 设备 / VAE 内存">
        <Row
          label="guidanceScale"
          description="FLUX dev 蒸馏版需要烘焙到 LoRA 的 guidance。留空跳过。"
          errors={errorMap.get(p("guidanceScale"))}
        >
          <FloatInput
            step={0.1}
            value={v.guidanceScale ?? null}
            onChange={(n) => set(["baseModel", "archPaths", "guidanceScale"], n)}
            placeholder="（默认）"
          />
        </Row>
        <Row label="t5xxlDevice" description="例如 cuda / cuda:1 / cpu。">
          <TextInput
            className="w-48"
            value={v.t5xxlDevice ?? ""}
            onChange={(s) =>
              set(["baseModel", "archPaths", "t5xxlDevice"], s || null)
            }
            placeholder="（默认）"
          />
        </Row>
        <Row label="t5xxlDtype">
          <EnumSelect
            value={v.t5xxlDtype ?? ""}
            onChange={(s) =>
              set(["baseModel", "archPaths", "t5xxlDtype"], s || null)
            }
            options={T5_DTYPE_OPTIONS}
          />
        </Row>
        <Row
          label="vaeChunkSize"
          description="Anima / HunyuanImage / Wan VAE 分块大小（节省显存）。"
          errors={errorMap.get(p("vaeChunkSize"))}
        >
          <IntInput
            min={1}
            value={v.vaeChunkSize ?? null}
            onChange={(n) =>
              set(["baseModel", "archPaths", "vaeChunkSize"], n)
            }
            placeholder="（默认）"
          />
        </Row>
        <Row label="vaeDisableCache" description="禁用 VAE 输出缓存。">
          <ToggleSwitch
            checked={v.vaeDisableCache ?? false}
            onCheckedChange={(b) =>
              set(["baseModel", "archPaths", "vaeDisableCache"], b)
            }
          />
        </Row>
        <Row label="textEncoderCpu" description="把文本编码器固定在 CPU。">
          <ToggleSwitch
            checked={v.textEncoderCpu ?? false}
            onCheckedChange={(b) =>
              set(["baseModel", "archPaths", "textEncoderCpu"], b)
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

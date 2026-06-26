/**
 * ArchPathsConfig editor — per-component checkpoint paths and arch-specific
 * memory / dropout / token-length knobs.
 *
 * Most arches don't use every field. The form renders only the subset the
 * selected backend + architecture can consume; hidden values stay in state so
 * switching architectures does not discard existing YAML.
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
  backendType,
}: {
  value: ArchPathsValue | undefined
  set: Setter
  errorMap: ErrorMap
  arch: string
  backendType?: string
}) {
  const v = value ?? {}
  const p = (k: keyof ArchPathsValue) => `baseModel.archPaths.${k}` as const
  const isFlux = arch === "flux" || arch === "flux2"
  const isSd3 = arch === "sd3"
  const isAnima = arch === "anima"
  const isHunyuanImage = arch === "hunyuan_image"
  const isVideoBundle =
    arch === "wan" ||
    arch === "hunyuan_video" ||
    arch === "hunyuan_video_15" ||
    arch === "ltx_video" ||
    arch === "ltx2" ||
    arch === "cosmos" ||
    arch === "cosmos_predict2"
  const isDp = backendType === "diffusion-pipe"
  const isAnimaBackend = backendType === "anima_lora"
  const showFlux = isFlux || (isDp && !arch)
  const showSd3 = isSd3 || (isDp && !arch)
  const showAnima = isAnima
  const showHunyuan = isHunyuanImage
  const showGenericDp = isDp && (isVideoBundle || arch === "chroma" || arch === "omnigen2")
  const showNothing =
    !showFlux && !showSd3 && !showAnima && !showHunyuan && !showGenericDp

  return (
    <>
      <div className="-mt-1 mb-1 text-[11px] text-muted-foreground/80">
        {hintFor(arch, backendType)}
      </div>

      {showNothing && (
        <div className="rounded-[4px] border border-dashed border-border/50 bg-muted/10 px-3 py-2 text-xs text-muted-foreground">
          当前架构不读取组件路径。
        </div>
      )}

      {showFlux && (
        <SubGroup label="FLUX 组件">
          <Row label="clipL" errors={errorMap.get(p("clipL"))}>
            <ModelPathPicker
              value={v.clipL ?? ""}
              onChange={(s) => set(["baseModel", "archPaths", "clipL"], s || null)}
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
          <Row label="ae" description="FLUX VAE。" errors={errorMap.get(p("ae"))}>
            <ModelPathPicker
              value={v.ae ?? ""}
              onChange={(s) => set(["baseModel", "archPaths", "ae"], s || null)}
              placeholder="（可选）"
            />
          </Row>
        </SubGroup>
      )}

      {showSd3 && (
        <SubGroup label="SD3 组件">
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
        </SubGroup>
      )}

      {showAnima && (
        <SubGroup label="Anima 组件">
          {!isAnimaBackend && (
            <>
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
            </>
          )}
        <Row label="qwen3" errors={errorMap.get(p("qwen3"))}>
          <ModelPathPicker
            value={v.qwen3 ?? ""}
            onChange={(s) => set(["baseModel", "archPaths", "qwen3"], s || null)}
            placeholder="（可选）"
          />
        </Row>
          <Row
            label="ae"
            description="Anima VAE。"
            errors={errorMap.get(p("ae"))}
          >
            <ModelPathPicker
              value={v.ae ?? ""}
              onChange={(s) => set(["baseModel", "archPaths", "ae"], s || null)}
              placeholder="（可选）"
            />
          </Row>
        </SubGroup>
      )}

      {(showHunyuan || showGenericDp) && (
        <SubGroup label={showHunyuan ? "HunyuanImage 组件" : "diffusion-pipe 组件"}>
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
          <Row label="llm" description="LLM 权重路径。" errors={errorMap.get(p("llm"))}>
            <ModelPathPicker
              value={v.llm ?? ""}
              onChange={(s) => set(["baseModel", "archPaths", "llm"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          {(showHunyuan || isDp) && (
            <Row label="byt5" description="byT5 权重路径。" errors={errorMap.get(p("byt5"))}>
              <ModelPathPicker
                value={v.byt5 ?? ""}
                onChange={(s) => set(["baseModel", "archPaths", "byt5"], s || null)}
                placeholder="（可选）"
              />
            </Row>
          )}
        </SubGroup>
      )}

      {(showFlux || showSd3 || showAnima) && (
        <SubGroup label="Token 长度上限">
          {(showFlux || showSd3) && (
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
          )}
          {showAnima && (
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
          )}
          {showAnima && !isAnimaBackend && (
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
          )}
        </SubGroup>
      )}

      {(showFlux || showSd3) && (
        <SubGroup label="Attention 掩码与 dropout">
        <Row label="applyT5AttnMask" description="启用 T5 attention mask。">
          <ToggleSwitch
            checked={v.applyT5AttnMask ?? false}
            onCheckedChange={(b) =>
              set(["baseModel", "archPaths", "applyT5AttnMask"], b)
            }
          />
        </Row>
          {showSd3 && (
            <Row label="applyLgAttnMask" description="启用 CLIP-L/G attention mask。">
          <ToggleSwitch
            checked={v.applyLgAttnMask ?? false}
            onCheckedChange={(b) =>
              set(["baseModel", "archPaths", "applyLgAttnMask"], b)
            }
          />
        </Row>
          )}
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
          {showSd3 && (
            <Row label="clipGDropoutRate" errors={errorMap.get(p("clipGDropoutRate"))}>
          <FloatInput
            step={0.05}
            value={v.clipGDropoutRate ?? 0}
            onChange={(n) =>
              set(["baseModel", "archPaths", "clipGDropoutRate"], n ?? 0)
            }
          />
        </Row>
          )}
          {showSd3 && (
            <>
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
            </>
          )}
      </SubGroup>
      )}

      {(showFlux || showSd3 || showAnima || showHunyuan) && (
        <SubGroup label="附加参数">
          {showFlux && (
        <Row
          label="guidanceScale"
          description="FLUX guidance scale。留空不写入。"
          errors={errorMap.get(p("guidanceScale"))}
        >
          <FloatInput
            step={0.1}
            value={v.guidanceScale ?? null}
            onChange={(n) => set(["baseModel", "archPaths", "guidanceScale"], n)}
            placeholder="（默认）"
          />
        </Row>
          )}
          {showSd3 && (
            <>
        <Row label="t5xxlDevice" description="T5XXL 设备，例如 cuda / cuda:1 / cpu。">
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
            </>
          )}
          {(showAnima || showHunyuan) && (
            <>
        <Row
          label="vaeChunkSize"
          description="VAE 分块大小。"
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
              {showAnima && (
        <Row label="vaeDisableCache" description="关闭 VAE 输出缓存。">
          <ToggleSwitch
            checked={v.vaeDisableCache ?? false}
            onCheckedChange={(b) =>
              set(["baseModel", "archPaths", "vaeDisableCache"], b)
            }
          />
        </Row>
              )}
            </>
          )}
          {showHunyuan && (
        <Row label="textEncoderCpu" description="文本编码器使用 CPU。">
          <ToggleSwitch
            checked={v.textEncoderCpu ?? false}
            onCheckedChange={(b) =>
              set(["baseModel", "archPaths", "textEncoderCpu"], b)
            }
          />
        </Row>
          )}
      </SubGroup>
      )}
    </>
  )
})

function hintFor(arch: string, backendType?: string) {
  if (backendType === "anima_lora" && arch === "anima") {
    return "当前仅显示 qwen3 与 ae 字段。"
  }
  switch (arch) {
    case "flux":
    case "flux2":
      return "FLUX 读取 clip_l / t5xxl / ae。"
    case "sd3":
      return "SD3 读取 clip_l / clip_g / t5xxl。"
    case "anima":
      return "Anima 读取 qwen3 / ae；kohya 可读取 llm_adapter / t5_tokenizer。"
    case "hunyuan_image":
      return "HunyuanImage 读取 transformer / text_encoder / byt5。"
    default:
      return "仅显示当前架构会读取的字段。"
  }
}

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

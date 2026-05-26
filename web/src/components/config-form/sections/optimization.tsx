import { memo } from "react"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { IntInput, Row, ToggleSwitch } from "../widgets"

/**
 * Editor for `optimization` (OptimizationConfig in schema.py).
 *
 * Speed / VRAM toggles compiled into kohya argv or diffusion-pipe TOML.
 * Each description tags which backend actually consumes the field; the
 * compilers ignore mismatched fields rather than erroring, so flipping
 * a kohya-only toggle on a dp recipe is harmless (and vice versa).
 */
export const OptimizationFields = memo(function OptimizationFields({
  value,
  set,
  errorMap,
}: {
  value: ConfigFormValue["optimization"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row
        label="torchCompile"
        description="kohya 训练加速；dp 已默认启用，保留此开关只为配置互通。"
      >
        <ToggleSwitch
          checked={v.torchCompile ?? false}
          onCheckedChange={(b) => set(["optimization", "torchCompile"], b)}
        />
      </Row>
      <Row
        label="fusedBackwardPass"
        description="kohya 专用，融合反向 + 优化器步进以节省显存；dp 暂无对应开关。"
      >
        <ToggleSwitch
          checked={v.fusedBackwardPass ?? false}
          onCheckedChange={(b) =>
            set(["optimization", "fusedBackwardPass"], b)
          }
        />
      </Row>
      <Row
        label="fullBf16"
        description="双后端通用，把模型 / 梯度 / 优化器状态全部放到 bf16，优化器内存约减半。"
      >
        <ToggleSwitch
          checked={v.fullBf16 ?? false}
          onCheckedChange={(b) => set(["optimization", "fullBf16"], b)}
        />
      </Row>
      <Row
        label="fullFp16"
        description="老一代 GPU 用：全 fp16 训练（包含优化器状态）。"
      >
        <ToggleSwitch
          checked={v.fullFp16 ?? false}
          onCheckedChange={(b) => set(["optimization", "fullFp16"], b)}
        />
      </Row>
      <Row
        label="blocksToSwap"
        description="FLUX/SD3/dp 适用，把 N 个 transformer 块临时换出到 CPU 以省显存；SDXL 不支持。"
        errors={errorMap.get("optimization.blocksToSwap")}
      >
        <IntInput
          min={0}
          value={v.blocksToSwap ?? 0}
          onChange={(n) => set(["optimization", "blocksToSwap"], n ?? 0)}
        />
      </Row>

      <details className="rounded-[4px] border border-border/40 bg-muted/10 px-3 py-2 group">
        <summary className="cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.18em]">
          FP8 / 内存策略 / 缓存
        </summary>
        <div className="mt-3 space-y-3.5">
          <Row label="fp8Base" description="FLUX/SD3/HunyuanImage 把 base 模型权重以 fp8 加载，显存约 -40%。">
            <ToggleSwitch
              checked={v.fp8Base ?? false}
              onCheckedChange={(b) => set(["optimization", "fp8Base"], b)}
            />
          </Row>
          <Row label="fp8BaseUnet" description="仅对 UNet 应用 fp8。">
            <ToggleSwitch
              checked={v.fp8BaseUnet ?? false}
              onCheckedChange={(b) => set(["optimization", "fp8BaseUnet"], b)}
            />
          </Row>
          <Row label="fp8Scaled" description="HunyuanImage 缩放 FP8（与 fp8_base 不同的算法）。">
            <ToggleSwitch
              checked={v.fp8Scaled ?? false}
              onCheckedChange={(b) => set(["optimization", "fp8Scaled"], b)}
            />
          </Row>
          <Row label="fp8VlTextEncoder" description="HunyuanImage VL 文本编码器以 fp8 加载。">
            <ToggleSwitch
              checked={v.fp8VlTextEncoder ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "fp8VlTextEncoder"], b)
              }
            />
          </Row>
          <Row label="lowram" description="kohya 内存吃紧策略。">
            <ToggleSwitch
              checked={v.lowram ?? false}
              onCheckedChange={(b) => set(["optimization", "lowram"], b)}
            />
          </Row>
          <Row label="highvram" description="kohya 显存充裕策略（少缓存）。">
            <ToggleSwitch
              checked={v.highvram ?? false}
              onCheckedChange={(b) => set(["optimization", "highvram"], b)}
            />
          </Row>
          <Row
            label="noHalfVae"
            description="SDXL VAE 强制 fp32，避免半精度色彩异常。"
          >
            <ToggleSwitch
              checked={v.noHalfVae ?? false}
              onCheckedChange={(b) => set(["optimization", "noHalfVae"], b)}
            />
          </Row>
          <Row
            label="disableMmapLoadSafetensors"
            description="禁用 safetensors mmap 加载（NFS / 网络盘必需）。"
          >
            <ToggleSwitch
              checked={v.disableMmapLoadSafetensors ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "disableMmapLoadSafetensors"], b)
              }
            />
          </Row>
          <Row
            label="cpuOffloadCheckpointing"
            description="梯度检查点 offload 到 CPU。"
          >
            <ToggleSwitch
              checked={v.cpuOffloadCheckpointing ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "cpuOffloadCheckpointing"], b)
              }
            />
          </Row>
          <Row
            label="unslothOffloadCheckpointing"
            description="Anima 专用 unsloth offload。"
          >
            <ToggleSwitch
              checked={v.unslothOffloadCheckpointing ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "unslothOffloadCheckpointing"], b)
              }
            />
          </Row>
          <Row
            label="cacheTextEncoderOutputs"
            description="把文本编码器输出缓存到 RAM。"
          >
            <ToggleSwitch
              checked={v.cacheTextEncoderOutputs ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "cacheTextEncoderOutputs"], b)
              }
            />
          </Row>
          <Row
            label="cacheTextEncoderOutputsToDisk"
            description="把文本编码器输出缓存到磁盘（释放显存）。"
          >
            <ToggleSwitch
              checked={v.cacheTextEncoderOutputsToDisk ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "cacheTextEncoderOutputsToDisk"], b)
              }
            />
          </Row>
        </div>
      </details>
    </>
  )
})

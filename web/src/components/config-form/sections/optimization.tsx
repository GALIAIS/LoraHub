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
        label="torch_compile"
        description="kohya 训练加速；dp 已默认启用，保留此开关只为配方互通。"
      >
        <ToggleSwitch
          checked={v.torch_compile ?? false}
          onCheckedChange={(b) => set(["optimization", "torch_compile"], b)}
        />
      </Row>
      <Row
        label="fused_backward_pass"
        description="kohya 专用，融合反向 + 优化器步进以节省显存；dp 暂无对应开关。"
      >
        <ToggleSwitch
          checked={v.fused_backward_pass ?? false}
          onCheckedChange={(b) =>
            set(["optimization", "fused_backward_pass"], b)
          }
        />
      </Row>
      <Row
        label="full_bf16"
        description="双后端通用，把模型 / 梯度 / 优化器状态全部放到 bf16，优化器内存约减半。"
      >
        <ToggleSwitch
          checked={v.full_bf16 ?? false}
          onCheckedChange={(b) => set(["optimization", "full_bf16"], b)}
        />
      </Row>
      <Row
        label="full_fp16"
        description="老一代 GPU 用：全 fp16 训练（包含优化器状态）。"
      >
        <ToggleSwitch
          checked={v.full_fp16 ?? false}
          onCheckedChange={(b) => set(["optimization", "full_fp16"], b)}
        />
      </Row>
      <Row
        label="blocks_to_swap"
        description="FLUX/SD3/dp 适用，把 N 个 transformer 块临时换出到 CPU 以省显存；SDXL 不支持。"
        errors={errorMap.get("optimization.blocks_to_swap")}
      >
        <IntInput
          min={0}
          value={v.blocks_to_swap ?? 0}
          onChange={(n) => set(["optimization", "blocks_to_swap"], n ?? 0)}
        />
      </Row>

      <details className="rounded-[4px] border border-border/40 bg-muted/10 px-3 py-2 group">
        <summary className="cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.18em]">
          FP8 / 内存策略 / 缓存
        </summary>
        <div className="mt-3 space-y-3.5">
          <Row label="fp8_base" description="FLUX/SD3/HunyuanImage 把 base 模型权重以 fp8 加载，显存约 -40%。">
            <ToggleSwitch
              checked={v.fp8_base ?? false}
              onCheckedChange={(b) => set(["optimization", "fp8_base"], b)}
            />
          </Row>
          <Row label="fp8_base_unet" description="仅对 UNet 应用 fp8。">
            <ToggleSwitch
              checked={v.fp8_base_unet ?? false}
              onCheckedChange={(b) => set(["optimization", "fp8_base_unet"], b)}
            />
          </Row>
          <Row label="fp8_scaled" description="HunyuanImage 缩放 FP8（与 fp8_base 不同的算法）。">
            <ToggleSwitch
              checked={v.fp8_scaled ?? false}
              onCheckedChange={(b) => set(["optimization", "fp8_scaled"], b)}
            />
          </Row>
          <Row label="fp8_vl_text_encoder" description="HunyuanImage VL 文本编码器以 fp8 加载。">
            <ToggleSwitch
              checked={v.fp8_vl_text_encoder ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "fp8_vl_text_encoder"], b)
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
            label="no_half_vae"
            description="SDXL VAE 强制 fp32，避免半精度色彩异常。"
          >
            <ToggleSwitch
              checked={v.no_half_vae ?? false}
              onCheckedChange={(b) => set(["optimization", "no_half_vae"], b)}
            />
          </Row>
          <Row
            label="disable_mmap_load_safetensors"
            description="禁用 safetensors mmap 加载（NFS / 网络盘必需）。"
          >
            <ToggleSwitch
              checked={v.disable_mmap_load_safetensors ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "disable_mmap_load_safetensors"], b)
              }
            />
          </Row>
          <Row
            label="cpu_offload_checkpointing"
            description="梯度检查点 offload 到 CPU。"
          >
            <ToggleSwitch
              checked={v.cpu_offload_checkpointing ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "cpu_offload_checkpointing"], b)
              }
            />
          </Row>
          <Row
            label="unsloth_offload_checkpointing"
            description="Anima 专用 unsloth offload。"
          >
            <ToggleSwitch
              checked={v.unsloth_offload_checkpointing ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "unsloth_offload_checkpointing"], b)
              }
            />
          </Row>
          <Row
            label="cache_text_encoder_outputs"
            description="把文本编码器输出缓存到 RAM。"
          >
            <ToggleSwitch
              checked={v.cache_text_encoder_outputs ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "cache_text_encoder_outputs"], b)
              }
            />
          </Row>
          <Row
            label="cache_text_encoder_outputs_to_disk"
            description="把文本编码器输出缓存到磁盘（释放显存）。"
          >
            <ToggleSwitch
              checked={v.cache_text_encoder_outputs_to_disk ?? false}
              onCheckedChange={(b) =>
                set(["optimization", "cache_text_encoder_outputs_to_disk"], b)
              }
            />
          </Row>
        </div>
      </details>
    </>
  )
})

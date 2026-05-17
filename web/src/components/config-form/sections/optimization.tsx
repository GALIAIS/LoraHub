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
    </>
  )
})

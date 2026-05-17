import { memo } from "react"
import { PRECISION_OPTIONS } from "../options"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import { EnumSelect, Row, ToggleSwitch } from "../widgets"

export const PrecisionFields = memo(function PrecisionFields({
  value,
  set,
}: {
  value: ConfigFormValue
  set: Setter
  errorMap: ErrorMap
}) {
  return (
    <>
      <Row label="精度" description="bf16 需要 Ampere 及以上（RTX 30/40、A100、H100）。">
        <EnumSelect
          value={value.precision ?? "bf16"}
          onChange={(p) => set(["precision"], p)}
          options={PRECISION_OPTIONS}
        />
      </Row>
      <Row
        label="梯度检查点"
        description="以约 20% 吞吐为代价节省显存。8GB 显卡几乎必开。"
      >
        <ToggleSwitch
          checked={value.gradient_checkpointing ?? true}
          onCheckedChange={(b) => set(["gradient_checkpointing"], b)}
        />
      </Row>
      <Row
        label="缓存潜变量"
        description="提前用 VAE 编码图片并存盘，大幅提速但占额外硬盘。"
      >
        <ToggleSwitch
          checked={value.cache_latents ?? true}
          onCheckedChange={(b) => set(["cache_latents"], b)}
        />
      </Row>
      <Row
        label="cache_latents_to_disk"
        description="把潜变量缓存写到磁盘（释放内存）。"
      >
        <ToggleSwitch
          checked={value.cache_latents_to_disk ?? false}
          onCheckedChange={(b) => set(["cache_latents_to_disk"], b)}
        />
      </Row>
      <Row
        label="skip_cache_check"
        description="跳过缓存一致性检查，加速冷启动。"
      >
        <ToggleSwitch
          checked={value.skip_cache_check ?? false}
          onCheckedChange={(b) => set(["skip_cache_check"], b)}
        />
      </Row>
      <Row
        label="cache_info"
        description="把缓存元信息单独写出（kohya 调试用）。"
      >
        <ToggleSwitch
          checked={value.cache_info ?? false}
          onCheckedChange={(b) => set(["cache_info"], b)}
        />
      </Row>
      <Row
        label="train_inpainting"
        description="启用 inpainting 训练目标。"
      >
        <ToggleSwitch
          checked={value.train_inpainting ?? false}
          onCheckedChange={(b) => set(["train_inpainting"], b)}
        />
      </Row>
    </>
  )
})

import { memo } from "react"
import { Switch } from "@/components/ui/switch"
import { PRECISION_OPTIONS } from "../options"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { EnumSelect, Row } from "../widgets"

export const PrecisionFields = memo(function PrecisionFields({
  value,
  set,
}: {
  value: RecipeFormValue
  set: Setter
  errorMap: ErrorMap
}) {
  return (
    <>
      <Row label="Precision" description="bf16 needs Ampere+ (RTX 30/40, A100, H100).">
        <EnumSelect
          value={value.precision ?? "bf16"}
          onChange={(p) => set(["precision"], p)}
          options={PRECISION_OPTIONS}
        />
      </Row>
      <Row
        label="Gradient checkpointing"
        description="Cuts VRAM at the cost of ~20% throughput. Almost always on for 8GB cards."
      >
        <Switch
          checked={value.gradient_checkpointing ?? true}
          onCheckedChange={(b) => set(["gradient_checkpointing"], b)}
        />
      </Row>
      <Row
        label="Cache latents"
        description="Pre-encode images via VAE, store on disk. Big speedup; needs disk space."
      >
        <Switch
          checked={value.cache_latents ?? true}
          onCheckedChange={(b) => set(["cache_latents"], b)}
        />
      </Row>
    </>
  )
})

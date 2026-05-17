/**
 * FlowMatchConfig editor — flow-matching loss knobs for FLUX/SD3/Lumina/Anima/
 * HunyuanImage/chroma/qwen_image. Hidden for ε-prediction arches that don't
 * consume any of these (SDXL, SD1.5, etc.).
 *
 * Every field is nullable on the schema side ("None means use trainer
 * default for the chosen arch"); the form mirrors that with empty Number /
 * Select inputs that send `null` upstream when blanked.
 */
import { memo } from "react"
import {
  FLOW_MATCH_ARCHES,
  FLOW_MATCH_PRED_TYPE_OPTIONS,
  FLOW_MATCH_TIMESTEP_OPTIONS,
  FLOW_MATCH_WEIGHTING_OPTIONS,
} from "../options"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import { EnumSelect, FloatInput, Row } from "../widgets"

export const FlowMatchFields = memo(function FlowMatchFields({
  value = {},
  set,
  errorMap,
  arch,
}: {
  value: ConfigFormValue["flow_match"]
  set: Setter
  errorMap: ErrorMap
  arch: string
}) {
  if (!FLOW_MATCH_ARCHES.has(arch)) return null
  const v = value ?? {}
  const k = (n: string) => `flow_match.${n}` as const
  return (
    <>
      <Row
        label="timestep_sampling"
        description="时间步采样方式。"
        errors={errorMap.get(k("timestep_sampling"))}
      >
        <EnumSelect
          value={v.timestep_sampling ?? ""}
          onChange={(s) => set(["flow_match", "timestep_sampling"], s || null)}
          options={FLOW_MATCH_TIMESTEP_OPTIONS}
        />
      </Row>
      <Row label="sigmoid_scale" errors={errorMap.get(k("sigmoid_scale"))}>
        <FloatInput
          step={0.01}
          value={v.sigmoid_scale ?? null}
          onChange={(n) => set(["flow_match", "sigmoid_scale"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="model_prediction_type" errors={errorMap.get(k("model_prediction_type"))}>
        <EnumSelect
          value={v.model_prediction_type ?? ""}
          onChange={(s) =>
            set(["flow_match", "model_prediction_type"], s || null)
          }
          options={FLOW_MATCH_PRED_TYPE_OPTIONS}
        />
      </Row>
      <Row
        label="discrete_flow_shift"
        description="FLUX / Anima 离散 flow 时间步偏移。"
        errors={errorMap.get(k("discrete_flow_shift"))}
      >
        <FloatInput
          step={0.1}
          value={v.discrete_flow_shift ?? null}
          onChange={(n) => set(["flow_match", "discrete_flow_shift"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="training_shift"
        description="SD3 训练时 shift。"
        errors={errorMap.get(k("training_shift"))}
      >
        <FloatInput
          step={0.1}
          value={v.training_shift ?? null}
          onChange={(n) => set(["flow_match", "training_shift"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="weighting_scheme"
        description="FLUX/SD3 时间步加权方案。"
        errors={errorMap.get(k("weighting_scheme"))}
      >
        <EnumSelect
          value={v.weighting_scheme ?? ""}
          onChange={(s) => set(["flow_match", "weighting_scheme"], s || null)}
          options={FLOW_MATCH_WEIGHTING_OPTIONS}
        />
      </Row>
      <Row label="logit_mean" errors={errorMap.get(k("logit_mean"))}>
        <FloatInput
          step={0.01}
          value={v.logit_mean ?? null}
          onChange={(n) => set(["flow_match", "logit_mean"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="logit_std" errors={errorMap.get(k("logit_std"))}>
        <FloatInput
          step={0.01}
          value={v.logit_std ?? null}
          onChange={(n) => set(["flow_match", "logit_std"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="mode_scale" errors={errorMap.get(k("mode_scale"))}>
        <FloatInput
          step={0.01}
          value={v.mode_scale ?? null}
          onChange={(n) => set(["flow_match", "mode_scale"], n)}
          placeholder="（默认）"
        />
      </Row>
    </>
  )
})

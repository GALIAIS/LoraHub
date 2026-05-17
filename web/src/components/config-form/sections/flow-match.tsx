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
  value: ConfigFormValue["flowMatch"]
  set: Setter
  errorMap: ErrorMap
  arch: string
}) {
  if (!FLOW_MATCH_ARCHES.has(arch)) return null
  const v = value ?? {}
  const k = (n: string) => `flowMatch.${n}` as const
  return (
    <>
      <Row
        label="timestepSampling"
        description="时间步采样方式。"
        errors={errorMap.get(k("timestepSampling"))}
      >
        <EnumSelect
          value={v.timestepSampling ?? ""}
          onChange={(s) => set(["flowMatch", "timestepSampling"], s || null)}
          options={FLOW_MATCH_TIMESTEP_OPTIONS}
        />
      </Row>
      <Row label="sigmoidScale" errors={errorMap.get(k("sigmoidScale"))}>
        <FloatInput
          step={0.01}
          value={v.sigmoidScale ?? null}
          onChange={(n) => set(["flowMatch", "sigmoidScale"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="modelPredictionType" errors={errorMap.get(k("modelPredictionType"))}>
        <EnumSelect
          value={v.modelPredictionType ?? ""}
          onChange={(s) =>
            set(["flowMatch", "modelPredictionType"], s || null)
          }
          options={FLOW_MATCH_PRED_TYPE_OPTIONS}
        />
      </Row>
      <Row
        label="discreteFlowShift"
        description="FLUX / Anima 离散 flow 时间步偏移。"
        errors={errorMap.get(k("discreteFlowShift"))}
      >
        <FloatInput
          step={0.1}
          value={v.discreteFlowShift ?? null}
          onChange={(n) => set(["flowMatch", "discreteFlowShift"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="trainingShift"
        description="SD3 训练时 shift。"
        errors={errorMap.get(k("trainingShift"))}
      >
        <FloatInput
          step={0.1}
          value={v.trainingShift ?? null}
          onChange={(n) => set(["flowMatch", "trainingShift"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="weightingScheme"
        description="FLUX/SD3 时间步加权方案。"
        errors={errorMap.get(k("weightingScheme"))}
      >
        <EnumSelect
          value={v.weightingScheme ?? ""}
          onChange={(s) => set(["flowMatch", "weightingScheme"], s || null)}
          options={FLOW_MATCH_WEIGHTING_OPTIONS}
        />
      </Row>
      <Row label="logitMean" errors={errorMap.get(k("logitMean"))}>
        <FloatInput
          step={0.01}
          value={v.logitMean ?? null}
          onChange={(n) => set(["flowMatch", "logitMean"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="logitStd" errors={errorMap.get(k("logitStd"))}>
        <FloatInput
          step={0.01}
          value={v.logitStd ?? null}
          onChange={(n) => set(["flowMatch", "logitStd"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="modeScale" errors={errorMap.get(k("modeScale"))}>
        <FloatInput
          step={0.01}
          value={v.modeScale ?? null}
          onChange={(n) => set(["flowMatch", "modeScale"], n)}
          placeholder="（默认）"
        />
      </Row>
    </>
  )
})

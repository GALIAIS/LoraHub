import { memo } from "react"
import type { BackendKey, ErrorMap, ConfigFormValue, Setter } from "../types"
import { IntInput, Row } from "../widgets"

export const ScheduleFields = memo(function ScheduleFields({
  value = {},
  set,
  errorMap,
  backendType,
}: {
  value: ConfigFormValue["schedule"]
  set: Setter
  errorMap: ErrorMap
  backendType?: BackendKey
}) {
  const v = value ?? {}
  const isAnima = backendType === "anima_lora"
  const isDiffusionPipe = backendType === "diffusion-pipe"
  return (
    <>
      {!isAnima && (
        <Row label="训练回合 (Epochs)" errors={errorMap.get("schedule.epochs")}>
          <IntInput min={1} value={v.epochs ?? 10} onChange={(n) => set(["schedule", "epochs"], n ?? 1)} />
        </Row>
      )}
      <Row label="批大小 (Batch)" errors={errorMap.get("schedule.batchSize")}>
        <IntInput min={1} value={v.batchSize ?? 1} onChange={(n) => set(["schedule", "batchSize"], n ?? 1)} />
      </Row>
      <Row label="梯度累积" description="有效批量 = batch × gradAccum。">
        <IntInput min={1} value={v.gradAccum ?? 2} onChange={(n) => set(["schedule", "gradAccum"], n ?? 1)} />
      </Row>
      <Row label="最大步数" description="可选硬性上限；留空则跑完所有回合。">
        <IntInput
          min={1}
          value={v.maxSteps ?? null}
          onChange={(n) => set(["schedule", "maxSteps"], n)}
          placeholder="（不限）"
        />
      </Row>
      {!isDiffusionPipe && (
        <>
          <Row
            label="随机种子"
            description="留空则使用随机种子。"
            errors={errorMap.get("schedule.seed")}
          >
            <IntInput
              value={v.seed ?? null}
              onChange={(n) => set(["schedule", "seed"], n)}
              placeholder="（随机）"
            />
          </Row>
          <Row
            label="lrDecaySteps"
            description="cosine / linear 衰减的步数窗口；留空使用全程长度。"
            errors={errorMap.get("schedule.lrDecaySteps")}
          >
            <IntInput
              min={1}
              value={v.lrDecaySteps ?? null}
              onChange={(n) => set(["schedule", "lrDecaySteps"], n)}
              placeholder="（默认）"
            />
          </Row>
        </>
      )}
    </>
  )
})

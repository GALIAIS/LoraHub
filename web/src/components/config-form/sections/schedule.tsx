import { memo } from "react"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { IntInput, Row } from "../widgets"

export const ScheduleFields = memo(function ScheduleFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["schedule"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row label="训练回合 (Epochs)" errors={errorMap.get("schedule.epochs")}>
        <IntInput min={1} value={v.epochs ?? 10} onChange={(n) => set(["schedule", "epochs"], n ?? 1)} />
      </Row>
      <Row label="批大小 (Batch)" errors={errorMap.get("schedule.batch_size")}>
        <IntInput min={1} value={v.batch_size ?? 1} onChange={(n) => set(["schedule", "batch_size"], n ?? 1)} />
      </Row>
      <Row label="梯度累积" description="有效批量 = batch × grad_accum。">
        <IntInput min={1} value={v.grad_accum ?? 2} onChange={(n) => set(["schedule", "grad_accum"], n ?? 1)} />
      </Row>
      <Row label="最大步数" description="可选硬性上限；留空则跑完所有回合。">
        <IntInput
          min={1}
          value={v.max_steps ?? null}
          onChange={(n) => set(["schedule", "max_steps"], n)}
          placeholder="（不限）"
        />
      </Row>
    </>
  )
})

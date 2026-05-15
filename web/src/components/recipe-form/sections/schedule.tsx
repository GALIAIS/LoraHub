import { memo } from "react"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { IntInput, Row } from "../widgets"

export const ScheduleFields = memo(function ScheduleFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["schedule"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row label="Epochs" errors={errorMap.get("schedule.epochs")}>
        <IntInput min={1} value={v.epochs ?? 10} onChange={(n) => set(["schedule", "epochs"], n ?? 1)} />
      </Row>
      <Row label="Batch size" errors={errorMap.get("schedule.batch_size")}>
        <IntInput min={1} value={v.batch_size ?? 1} onChange={(n) => set(["schedule", "batch_size"], n ?? 1)} />
      </Row>
      <Row label="Grad accumulation" description="Effective batch = batch × grad_accum.">
        <IntInput min={1} value={v.grad_accum ?? 2} onChange={(n) => set(["schedule", "grad_accum"], n ?? 1)} />
      </Row>
      <Row label="Max steps" description="Optional hard cap; leave empty to run all epochs.">
        <IntInput
          min={1}
          value={v.max_steps ?? null}
          onChange={(n) => set(["schedule", "max_steps"], n)}
          placeholder="(unbounded)"
        />
      </Row>
    </>
  )
})

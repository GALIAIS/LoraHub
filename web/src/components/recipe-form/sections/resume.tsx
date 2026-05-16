import { memo } from "react"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { IntInput, Row, ToggleSwitch } from "../widgets"

/**
 * Checkpoint state writing for resume support (ResumeConfig in schema.py).
 *
 * State directories are large; use `save_state_every_n_epochs` to throttle
 * writes if disk is tight.
 */
export const ResumeFields = memo(function ResumeFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["resume"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row
        label="保存训练状态"
        description="同时落盘 optimizer / scheduler 状态以便后续 resume。"
      >
        <ToggleSwitch
          checked={v.save_state ?? true}
          onCheckedChange={(b) => set(["resume", "save_state"], b)}
        />
      </Row>
      <Row
        label="结束时保存"
        description="训练正常结束时再写一份 state（用于断点续训）。"
      >
        <ToggleSwitch
          checked={v.save_state_at_end ?? true}
          onCheckedChange={(b) => set(["resume", "save_state_at_end"], b)}
        />
      </Row>
      <Row
        label="每 N 回合保存状态"
        description="可选。State 目录较大，磁盘紧张时用此节流。留空则关闭周期性保存。"
        errors={errorMap.get("resume.save_state_every_n_epochs")}
      >
        <IntInput
          min={1}
          value={v.save_state_every_n_epochs ?? null}
          onChange={(n) => set(["resume", "save_state_every_n_epochs"], n)}
          placeholder="（不周期保存）"
        />
      </Row>
    </>
  )
})

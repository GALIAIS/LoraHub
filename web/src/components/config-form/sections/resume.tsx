import { memo } from "react"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { IntInput, PathInput, Row, ToggleSwitch } from "../widgets"

/**
 * Checkpoint state writing for resume support (ResumeConfig in schema.py).
 *
 * State directories are large; use `saveStateEveryNEpochs` to throttle
 * writes if disk is tight.
 */
export const ResumeFields = memo(function ResumeFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["resume"]
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
          checked={v.saveState ?? true}
          onCheckedChange={(b) => set(["resume", "saveState"], b)}
        />
      </Row>
      <Row
        label="结束时保存"
        description="训练正常结束时再写一份 state（用于断点续训）。"
      >
        <ToggleSwitch
          checked={v.saveStateAtEnd ?? true}
          onCheckedChange={(b) => set(["resume", "saveStateAtEnd"], b)}
        />
      </Row>
      <Row
        label="每 N 回合保存状态"
        description="可选。State 目录较大，磁盘紧张时用此节流。留空则关闭周期性保存。"
        errors={errorMap.get("resume.saveStateEveryNEpochs")}
      >
        <IntInput
          min={1}
          value={v.saveStateEveryNEpochs ?? null}
          onChange={(n) => set(["resume", "saveStateEveryNEpochs"], n)}
          placeholder="（不周期保存）"
        />
      </Row>
      <Row
        label="resumeFrom"
        description="本地恢复路径（kohya `--resume`）。"
        errors={errorMap.get("resume.resumeFrom")}
      >
        <PathInput
          value={v.resumeFrom ?? ""}
          onChange={(s) => set(["resume", "resumeFrom"], s || null)}
          placeholder="（可选）"
        />
      </Row>
      <Row
        label="保留最近 N 回合 state"
        errors={errorMap.get("resume.saveLastNEpochsState")}
      >
        <IntInput
          min={1}
          value={v.saveLastNEpochsState ?? null}
          onChange={(n) => set(["resume", "saveLastNEpochsState"], n)}
          placeholder="（不限）"
        />
      </Row>
      <Row
        label="保留最近 N 步 state"
        errors={errorMap.get("resume.saveLastNStepsState")}
      >
        <IntInput
          min={1}
          value={v.saveLastNStepsState ?? null}
          onChange={(n) => set(["resume", "saveLastNStepsState"], n)}
          placeholder="（不限）"
        />
      </Row>
      <Row
        label="skipUntilInitialStep"
        description="resume 时跳到指定步（kohya）。"
      >
        <ToggleSwitch
          checked={v.skipUntilInitialStep ?? false}
          onCheckedChange={(b) => set(["resume", "skipUntilInitialStep"], b)}
        />
      </Row>
      <Row label="initialEpoch" errors={errorMap.get("resume.initialEpoch")}>
        <IntInput
          min={1}
          value={v.initialEpoch ?? null}
          onChange={(n) => set(["resume", "initialEpoch"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="initialStep" errors={errorMap.get("resume.initialStep")}>
        <IntInput
          min={0}
          value={v.initialStep ?? null}
          onChange={(n) => set(["resume", "initialStep"], n)}
          placeholder="（默认）"
        />
      </Row>
    </>
  )
})

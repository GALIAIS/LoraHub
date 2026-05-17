import { memo } from "react"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { IntInput, PathInput, Row, ToggleSwitch } from "../widgets"

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
      <Row
        label="resume_from"
        description="本地恢复路径（kohya `--resume`）。"
        errors={errorMap.get("resume.resume_from")}
      >
        <PathInput
          value={v.resume_from ?? ""}
          onChange={(s) => set(["resume", "resume_from"], s || null)}
          placeholder="（可选）"
        />
      </Row>
      <Row
        label="保留最近 N 回合 state"
        errors={errorMap.get("resume.save_last_n_epochs_state")}
      >
        <IntInput
          min={1}
          value={v.save_last_n_epochs_state ?? null}
          onChange={(n) => set(["resume", "save_last_n_epochs_state"], n)}
          placeholder="（不限）"
        />
      </Row>
      <Row
        label="保留最近 N 步 state"
        errors={errorMap.get("resume.save_last_n_steps_state")}
      >
        <IntInput
          min={1}
          value={v.save_last_n_steps_state ?? null}
          onChange={(n) => set(["resume", "save_last_n_steps_state"], n)}
          placeholder="（不限）"
        />
      </Row>
      <Row
        label="skip_until_initial_step"
        description="resume 时跳到指定步（kohya）。"
      >
        <ToggleSwitch
          checked={v.skip_until_initial_step ?? false}
          onCheckedChange={(b) => set(["resume", "skip_until_initial_step"], b)}
        />
      </Row>
      <Row label="initial_epoch" errors={errorMap.get("resume.initial_epoch")}>
        <IntInput
          min={1}
          value={v.initial_epoch ?? null}
          onChange={(n) => set(["resume", "initial_epoch"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="initial_step" errors={errorMap.get("resume.initial_step")}>
        <IntInput
          min={0}
          value={v.initial_step ?? null}
          onChange={(n) => set(["resume", "initial_step"], n)}
          placeholder="（默认）"
        />
      </Row>
    </>
  )
})

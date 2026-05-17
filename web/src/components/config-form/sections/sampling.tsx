import { memo } from "react"
import { SAMPLING_ATTENTION_OPTIONS } from "../options"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { EnumSelect, IntInput, PathInput, ResolutionInput, Row, ToggleSwitch } from "../widgets"

export const SamplingFields = memo(function SamplingFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["sampling"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const enabled = v.enabled ?? true
  return (
    <>
      <Row label="启用采样" description="训练过程中生成预览图。">
        <ToggleSwitch
          checked={enabled}
          onCheckedChange={(b) => set(["sampling", "enabled"], b)}
        />
      </Row>
      {enabled && (
        <>
          <Row label="每 N 回合一次">
            <IntInput
              min={1}
              value={v.everyNEpochs ?? 1}
              onChange={(n) => set(["sampling", "everyNEpochs"], n ?? 1)}
            />
          </Row>
          <Row
            label="每 N 步一次"
            description="kohya `--sample_every_n_steps`；与 everyNEpochs 互不冲突。留空仅按回合采样。"
            errors={errorMap.get("sampling.everyNSteps")}
          >
            <IntInput
              min={1}
              value={v.everyNSteps ?? null}
              onChange={(n) => set(["sampling", "everyNSteps"], n)}
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="训练前先采样"
            description="kohya `--sample_at_first`：第 0 步生成一组基线样图。"
          >
            <ToggleSwitch
              checked={v.atFirst ?? false}
              onCheckedChange={(b) => set(["sampling", "atFirst"], b)}
            />
          </Row>
          <Row label="提示词文件" description="纯文本，每行一条提示词。" errors={errorMap.get("sampling.promptsFile")}>
            <PathInput
              value={v.promptsFile ?? ""}
              onChange={(s) => set(["sampling", "promptsFile"], s || null)}
              placeholder="./prompts.txt"
            />
          </Row>
          <Row label="分辨率">
            <ResolutionInput
              value={v.resolution}
              onChange={(r) => set(["sampling", "resolution"], r)}
            />
          </Row>
          <Row label="随机种子">
            <IntInput value={v.seed ?? 42} onChange={(n) => set(["sampling", "seed"], n ?? 42)} />
          </Row>
          <Row
            label="采样 Attention"
            description="仅采样/验证前向使用，不影响训练梯度。当前为占位字段：选择非默认值会被记录但暂不生效（待运行时 wrapper），训练仍走 attention.training 指定的内核。"
            errors={errorMap.get("sampling.attention")}
          >
            <EnumSelect
              value={v.attention ?? "default"}
              onChange={(s) => set(["sampling", "attention"], s || "default")}
              options={SAMPLING_ATTENTION_OPTIONS}
            />
          </Row>
        </>
      )}
    </>
  )
})

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
              value={v.every_n_epochs ?? 1}
              onChange={(n) => set(["sampling", "every_n_epochs"], n ?? 1)}
            />
          </Row>
          <Row label="提示词文件" description="纯文本，每行一条提示词。" errors={errorMap.get("sampling.prompts_file")}>
            <PathInput
              value={v.prompts_file ?? ""}
              onChange={(s) => set(["sampling", "prompts_file"], s || null)}
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

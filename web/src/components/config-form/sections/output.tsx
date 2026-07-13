import { memo } from "react"
import { Input } from "@/components/ui/input"
import { SAVE_DTYPE_OPTIONS } from "../options"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import {
  EnumSelect,
  IntInput,
  KeyValueTextArea,
  PathInput,
  Row,
  TextInput,
  ToggleSwitch,
} from "../widgets"

export const OutputFields = memo(function OutputFields({
  value = {},
  set,
  errorMap,
  backendType,
}: {
  value: ConfigFormValue["output"]
  set: Setter
  errorMap: ErrorMap
  backendType?: "kohya" | "diffusion-pipe" | "anima_lora" | "ai_toolkit"
}) {
  const v = value ?? {}
  const isDiffusionPipe = backendType === "diffusion-pipe"
  const isAiToolkit = backendType === "ai_toolkit"
  return (
    <>
      {!isDiffusionPipe && (
        <Row label="名称" description="作为 LoRA 文件名和任务标识。">
          <Input
            value={v.name ?? ""}
            className="font-mono w-64"
            onChange={(e) => set(["output", "name"], e.target.value)}
            placeholder="my_character"
          />
        </Row>
      )}
      <Row label="每 N 回合保存一次">
        <IntInput
          min={1}
          value={v.saveEveryNEpochs ?? 1}
          onChange={(n) => set(["output", "saveEveryNEpochs"], n ?? 1)}
        />
      </Row>
      <Row
        label="每 N 步保存"
        description="可选。kohya / dp 步级保存频率。"
        errors={errorMap.get("output.saveEveryNSteps")}
      >
        <IntInput
          min={1}
          value={v.saveEveryNSteps ?? null}
          onChange={(n) => set(["output", "saveEveryNSteps"], n)}
          placeholder="（默认）"
        />
      </Row>
      {isDiffusionPipe && (
        <Row
          label="每 N 样本保存"
          description="dp examples 级保存频率。"
          errors={errorMap.get("output.saveEveryNExamples")}
        >
          <IntInput
            min={1}
            value={v.saveEveryNExamples ?? null}
            onChange={(n) => set(["output", "saveEveryNExamples"], n)}
            placeholder="（默认）"
          />
        </Row>
      )}
      {!isDiffusionPipe && (
        <>
          <Row
            label="保留最近 N 回合"
            description="只保留最近 N 个回合检查点。"
            errors={errorMap.get("output.saveLastNEpochs")}
          >
            <IntInput
              min={1}
              value={v.saveLastNEpochs ?? null}
              onChange={(n) => set(["output", "saveLastNEpochs"], n)}
              placeholder="（不限）"
            />
          </Row>
          <Row
            label="保留最近 N 步"
            description="只保留最近 N 个步级检查点。"
            errors={errorMap.get("output.saveLastNSteps")}
          >
            <IntInput
              min={1}
              value={v.saveLastNSteps ?? null}
              onChange={(n) => set(["output", "saveLastNSteps"], n)}
              placeholder="（不限）"
            />
          </Row>
        </>
      )}
      <Row label="保存精度" description="fp16 文件更小；bf16 需要 Ampere 及以上。">
        <EnumSelect
          value={v.saveDtype ?? "fp16"}
          onChange={(d) => set(["output", "saveDtype"], d)}
          options={SAVE_DTYPE_OPTIONS}
        />
      </Row>
      {!isAiToolkit && (
        <Row label="输出目录" description="默认使用任务工作目录下的 output。" errors={errorMap.get("output.outputDir")}>
          <PathInput
            value={v.outputDir ?? ""}
            onChange={(s) => set(["output", "outputDir"], s || null)}
            placeholder="（默认）"
          />
        </Row>
      )}
      {!isDiffusionPipe && !isAiToolkit && (
        <>
          <Row
            label="训练注释"
            description="写入 LoRA 元数据。"
          >
            <TextInput
              className="w-full max-w-xl"
              value={v.trainingComment ?? ""}
              onChange={(s) => set(["output", "trainingComment"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          <Row
            label="不写元数据"
            description="不写入 metadata。"
          >
            <ToggleSwitch
              checked={v.noMetadata ?? false}
              onCheckedChange={(b) => set(["output", "noMetadata"], b)}
            />
          </Row>
          <Row
            label="元数据"
            description="自定义 key=value，每行一对。"
            errors={errorMap.get("output.metadata")}
          >
            <KeyValueTextArea
              rows={4}
              value={v.metadata}
              onChange={(next) => set(["output", "metadata"], next)}
              placeholder={"author = me\nlicense = CC-BY"}
            />
          </Row>
        </>
      )}
    </>
  )
})

import { memo } from "react"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import {
  FloatInput,
  IntInput,
  KeyValueTextArea,
  Row,
  ToggleSwitch,
} from "../widgets"
import {
  DiffusionPipeDtypeSection,
  DiffusionPipeMediaSection,
  DiffusionPipePerformanceSection,
} from "./backend-diffusion-pipe-basics"
import { SubGroup } from "./backend-diffusion-pipe-shared"

/**
 * Editor for `backend.diffusionPipe` (DiffusionPipeOptions in schema.py).
 *
 * Only mounted when the user picks the diffusion-pipe backend; the kohya
 * backend ignores this entire branch. Fields are grouped to keep the long
 * list scannable: 性能 / 评估 / 监控 / AR bucket / 模型路径 / 其他.
 */
export const BackendDiffusionPipeFields = memo(
  function BackendDiffusionPipeFields({
    value,
    set,
    errorMap,
  }: {
    value: NonNullable<ConfigFormValue["backend"]>["diffusionPipe"]
    set: Setter
    errorMap: ErrorMap
  }) {
    const v = value ?? {}
    const evalEnabled = v.evalEveryNEpochs !== null && v.evalEveryNEpochs !== undefined
    return (
      <>
        <DiffusionPipePerformanceSection
          value={v}
          set={set}
          errorMap={errorMap}
        />
        <DiffusionPipeMediaSection
          value={v}
          set={set}
          errorMap={errorMap}
        />
        <DiffusionPipeDtypeSection value={v} set={set} />
        <SubGroup label="评估">
          <Row
            label="每 N 回合验证"
            description="可选；勾选后才写入 [eval] 段。"
            errors={errorMap.get("backend.diffusionPipe.evalEveryNEpochs")}
          >
            <div className="flex items-center gap-3">
              <ToggleSwitch
                checked={evalEnabled}
                onCheckedChange={(b) =>
                  set(
                    ["backend", "diffusionPipe", "evalEveryNEpochs"],
                    b ? 1 : null,
                  )
                }
              />
              {evalEnabled && (
                <IntInput
                  min={1}
                  value={v.evalEveryNEpochs ?? 1}
                  onChange={(n) =>
                    set(
                      ["backend", "diffusionPipe", "evalEveryNEpochs"],
                      n ?? 1,
                    )
                  }
                />
              )}
            </div>
          </Row>
          <Row
            label="每 N 步验证"
            errors={errorMap.get("backend.diffusionPipe.evalEveryNSteps")}
          >
            <IntInput
              min={1}
              value={v.evalEveryNSteps ?? null}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "evalEveryNSteps"], n)
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="每 N 样本验证"
            errors={errorMap.get("backend.diffusionPipe.evalEveryNExamples")}
          >
            <IntInput
              min={1}
              value={v.evalEveryNExamples ?? null}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "evalEveryNExamples"], n)
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="首步前先评估"
            description="第一步训练之前先跑一次评估。"
          >
            <ToggleSwitch
              checked={v.evalBeforeFirstStep ?? false}
              onCheckedChange={(b) =>
                set(["backend", "diffusionPipe", "evalBeforeFirstStep"], b)
              }
            />
          </Row>
          <Row
            label="评估批大小"
            description="单 GPU 的 micro-batch 大小。"
            errors={errorMap.get(
              "backend.diffusionPipe.evalMicroBatchSizePerGpu",
            )}
          >
            <IntInput
              min={1}
              value={v.evalMicroBatchSizePerGpu ?? 1}
              onChange={(n) =>
                set(
                  ["backend", "diffusionPipe", "evalMicroBatchSizePerGpu"],
                  n ?? 1,
                )
              }
            />
          </Row>
          <Row
            label="评估梯度累积"
            description="评估期梯度累积步数（影响 perplexity / loss 平均口径）。"
            errors={errorMap.get(
              "backend.diffusionPipe.evalGradientAccumulationSteps",
            )}
          >
            <IntInput
              min={1}
              value={v.evalGradientAccumulationSteps ?? 1}
              onChange={(n) =>
                set(
                  [
                    "backend",
                    "diffusionPipe",
                    "evalGradientAccumulationSteps",
                  ],
                  n ?? 1,
                )
              }
            />
          </Row>
          <Row
            label="Disable Block Swap For Eval"
            description="评估时跳过 block_swap（评估占用更小）。"
          >
            <ToggleSwitch
              checked={v.disableBlockSwapForEval ?? false}
              onCheckedChange={(b) =>
                set(
                  ["backend", "diffusionPipe", "disableBlockSwapForEval"],
                  b,
                )
              }
            />
          </Row>
          <Row
            label="Eval Datasets"
            description="独立评估数据集列表（JSON 数组，每项 {name, config_path}）。"
            errors={errorMap.get("backend.diffusionPipe.evalDatasets")}
          >
            <textarea
              value={JSON.stringify(v.evalDatasets ?? [], null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value || "[]")
                  if (Array.isArray(parsed)) {
                    set(
                      ["backend", "diffusionPipe", "evalDatasets"],
                      parsed,
                    )
                  }
                } catch {
                  // 用户编辑中途允许 JSON 不合法。
                }
              }}
              rows={4}
              className="font-mono w-full max-w-2xl rounded-[4px] border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none placeholder:text-muted-foreground/60 focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
              placeholder={'[{"name": "holdout", "config_path": "./eval.toml"}]'}
            />
          </Row>
        </SubGroup>

        <SubGroup label="DeepSpeed Checkpoint">
          <Row
            label="checkpointEveryNEpochs"
            errors={errorMap.get(
              "backend.diffusionPipe.checkpointEveryNEpochs",
            )}
          >
            <IntInput
              min={1}
              value={v.checkpointEveryNEpochs ?? null}
              onChange={(n) =>
                set(
                  ["backend", "diffusionPipe", "checkpointEveryNEpochs"],
                  n,
                )
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="checkpointEveryNMinutes"
            errors={errorMap.get(
              "backend.diffusionPipe.checkpointEveryNMinutes",
            )}
          >
            <IntInput
              min={1}
              value={v.checkpointEveryNMinutes ?? null}
              onChange={(n) =>
                set(
                  ["backend", "diffusionPipe", "checkpointEveryNMinutes"],
                  n,
                )
              }
              placeholder="（默认）"
            />
          </Row>
        </SubGroup>

        <SubGroup label="AR Bucket">
          <Row
            label="最小宽高比"
            errors={errorMap.get("backend.diffusionPipe.minAr")}
          >
            <FloatInput
              step={0.1}
              value={v.minAr ?? 0.5}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "minAr"], n ?? 0.5)
              }
            />
          </Row>
          <Row
            label="最大宽高比"
            errors={errorMap.get("backend.diffusionPipe.maxAr")}
          >
            <FloatInput
              step={0.1}
              value={v.maxAr ?? 2.0}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "maxAr"], n ?? 2.0)
              }
            />
          </Row>
          <Row
            label="AR Bucket 数"
            errors={errorMap.get("backend.diffusionPipe.numArBuckets")}
          >
            <IntInput
              min={1}
              value={v.numArBuckets ?? 7}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "numArBuckets"], n ?? 7)
              }
            />
          </Row>
        </SubGroup>

        <SubGroup label="模型路径">
          <Row
            label="modelPaths"
            description="dp [model] 段的额外路径键值。每行一对，例如 transformer_path = /path/to.safetensors。"
            errors={errorMap.get("backend.diffusionPipe.modelPaths")}
          >
            <KeyValueTextArea
              rows={5}
              value={v.modelPaths}
              onChange={(next) =>
                set(["backend", "diffusionPipe", "modelPaths"], next)
              }
              placeholder={
                "transformer_path = /path/to/transformer.safetensors\nvae_path = /path/to/vae.safetensors\nllm_path = /path/to/llm"
              }
            />
          </Row>
        </SubGroup>

        <SubGroup label="其他">
          <Row
            label="Cache Shuffle Num"
            description="预缓存阶段打乱样本数；0 保持原顺序。"
            errors={errorMap.get("backend.diffusionPipe.cacheShuffleNum")}
          >
            <IntInput
              min={0}
              value={v.cacheShuffleNum ?? 0}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "cacheShuffleNum"], n ?? 0)
              }
            />
          </Row>
          <Row
            label="跳过空 Caption"
            description="忽略 caption 为空的样本。"
          >
            <ToggleSwitch
              checked={v.skipEmptyCaption ?? true}
              onCheckedChange={(b) =>
                set(["backend", "diffusionPipe", "skipEmptyCaption"], b)
              }
            />
          </Row>
        </SubGroup>
      </>
    )
  },
)

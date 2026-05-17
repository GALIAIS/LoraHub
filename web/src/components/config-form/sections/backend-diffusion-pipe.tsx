import { memo } from "react"
import {
  DP_DIFFUSION_DTYPE_OPTIONS,
  DP_TIMESTEP_SAMPLE_OPTIONS,
  DP_TRANSFORMER_DTYPE_OPTIONS,
  DP_VIDEO_CLIP_MODE_OPTIONS,
  PARTITION_METHOD_OPTIONS,
} from "../options"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import {
  EnumSelect,
  FloatInput,
  IntInput,
  KeyValueTextArea,
  Row,
  TextInput,
  ToggleSwitch,
} from "../widgets"

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
        <SubGroup label="性能与切分">
          <Row
            label="Pipeline Stages"
            description="将模型沿层切分到多 GPU 的份数（1 表示不切）。"
            errors={errorMap.get("backend.diffusionPipe.pipelineStages")}
          >
            <IntInput
              min={1}
              value={v.pipelineStages ?? 1}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "pipelineStages"], n ?? 1)
              }
            />
          </Row>
          <Row
            label="Blocks To Swap"
            description="CPU offload 的 transformer block 数；显存吃紧时调高。"
            errors={errorMap.get("backend.diffusionPipe.blocksToSwap")}
          >
            <IntInput
              min={0}
              value={v.blocksToSwap ?? 0}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "blocksToSwap"], n ?? 0)
              }
            />
          </Row>
          <Row
            label="Caching Batch Size"
            description="VAE / 文本嵌入预缓存阶段的批大小。"
            errors={errorMap.get("backend.diffusionPipe.cachingBatchSize")}
          >
            <IntInput
              min={1}
              value={v.cachingBatchSize ?? 1}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "cachingBatchSize"], n ?? 1)
              }
            />
          </Row>
          <Row label="Compile" description="启用 torch.compile 加速（PyTorch 2.x）。">
            <ToggleSwitch
              checked={v.compile ?? false}
              onCheckedChange={(b) =>
                set(["backend", "diffusionPipe", "compile"], b)
              }
            />
          </Row>
          <Row
            label="Gradient Clipping"
            description="梯度范数裁剪上限。"
            errors={errorMap.get("backend.diffusionPipe.gradientClipping")}
          >
            <FloatInput
              step={0.1}
              value={v.gradientClipping ?? 1.0}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "gradientClipping"], n ?? 1.0)
              }
            />
          </Row>
          <Row label="Partition Method" description="pipeline 分片策略。">
            <EnumSelect
              value={v.partitionMethod ?? "parameters"}
              onChange={(t) =>
                set(["backend", "diffusionPipe", "partitionMethod"], t)
              }
              options={PARTITION_METHOD_OPTIONS}
            />
          </Row>
          <Row
            label="Steps Per Print"
            description="每多少步 flush 一次训练日志。"
            errors={errorMap.get("backend.diffusionPipe.stepsPerPrint")}
          >
            <IntInput
              min={1}
              value={v.stepsPerPrint ?? 1}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "stepsPerPrint"], n ?? 1)
              }
            />
          </Row>
          <Row
            label="Partition Split"
            description="manual 切分时各 stage 的层数列表，逗号分隔（长度 = pipelineStages - 1）。"
            errors={errorMap.get("backend.diffusionPipe.partitionSplit")}
          >
            <TextInput
              className="w-64"
              value={(v.partitionSplit ?? []).join(",")}
              onChange={(s) => {
                const list = s
                  .split(",")
                  .map((x) => x.trim())
                  .filter((x) => x.length > 0)
                  .map((x) => parseInt(x, 10))
                  .filter((n) => !Number.isNaN(n))
                set(
                  ["backend", "diffusionPipe", "partitionSplit"],
                  list.length ? list : null,
                )
              }}
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="Reentrant Activation Checkpointing"
            description="管线并行 + 重入式激活检查点（dp 限定场景）。"
          >
            <ToggleSwitch
              checked={v.reentrantActivationCheckpointing ?? false}
              onCheckedChange={(b) =>
                set(
                  ["backend", "diffusionPipe", "reentrantActivationCheckpointing"],
                  b,
                )
              }
            />
          </Row>
          <Row
            label="Force Constant LR"
            description="忽略 scheduler，强制使用恒定 LR（resume 调试用）。"
            errors={errorMap.get("backend.diffusionPipe.forceConstantLr")}
          >
            <FloatInput
              step={1e-5}
              value={v.forceConstantLr ?? null}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "forceConstantLr"], n)
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="Uncond Fraction"
            description="CFG 风格训练：丢弃 caption 的步数比例（0..1）。"
            errors={errorMap.get("backend.diffusionPipe.uncondFraction")}
          >
            <FloatInput
              step={0.05}
              value={v.uncondFraction ?? 0}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "uncondFraction"], n ?? 0)
              }
            />
          </Row>
          <Row
            label="X-axis Examples"
            description="Tensorboard X 轴用 examples 而不是 steps。"
          >
            <ToggleSwitch
              checked={v.xAxisExamples ?? false}
              onCheckedChange={(b) =>
                set(["backend", "diffusionPipe", "xAxisExamples"], b)
              }
            />
          </Row>
          <Row
            label="Logging Steps"
            description="每多少步写一次 wandb / tensorboard。"
            errors={errorMap.get("backend.diffusionPipe.loggingSteps")}
          >
            <IntInput
              min={1}
              value={v.loggingSteps ?? 1}
              onChange={(n) =>
                set(["backend", "diffusionPipe", "loggingSteps"], n ?? 1)
              }
            />
          </Row>
        </SubGroup>

        <SubGroup label="混合 image / video">
          <Row
            label="Image Micro Batch Size"
            description="混合训练时单 GPU 图像 micro batch。"
            errors={errorMap.get(
              "backend.diffusionPipe.imageMicroBatchSizePerGpu",
            )}
          >
            <IntInput
              min={1}
              value={v.imageMicroBatchSizePerGpu ?? null}
              onChange={(n) =>
                set(
                  ["backend", "diffusionPipe", "imageMicroBatchSizePerGpu"],
                  n,
                )
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="Image Eval Micro Batch Size"
            description="混合训练评估时单 GPU 图像 micro batch。"
            errors={errorMap.get(
              "backend.diffusionPipe.imageEvalMicroBatchSizePerGpu",
            )}
          >
            <IntInput
              min={1}
              value={v.imageEvalMicroBatchSizePerGpu ?? null}
              onChange={(n) =>
                set(
                  [
                    "backend",
                    "diffusionPipe",
                    "imageEvalMicroBatchSizePerGpu",
                  ],
                  n,
                )
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="Video Clip Mode"
            description="视频片段抽取策略。"
          >
            <EnumSelect
              value={v.videoClipMode ?? "single_beginning"}
              onChange={(s) =>
                set(["backend", "diffusionPipe", "videoClipMode"], s)
              }
              options={DP_VIDEO_CLIP_MODE_OPTIONS}
            />
          </Row>
        </SubGroup>

        <SubGroup label="dtype / 时间步采样">
          <Row label="Transformer dtype">
            <EnumSelect
              value={v.transformerDtype ?? ""}
              onChange={(s) =>
                set(
                  ["backend", "diffusionPipe", "transformerDtype"],
                  s || null,
                )
              }
              options={DP_TRANSFORMER_DTYPE_OPTIONS}
            />
          </Row>
          <Row label="Diffusion model dtype">
            <EnumSelect
              value={v.diffusionModelDtype ?? ""}
              onChange={(s) =>
                set(
                  ["backend", "diffusionPipe", "diffusionModelDtype"],
                  s || null,
                )
              }
              options={DP_DIFFUSION_DTYPE_OPTIONS}
            />
          </Row>
          <Row label="Timestep sample method">
            <EnumSelect
              value={v.timestepSampleMethod ?? ""}
              onChange={(s) =>
                set(
                  ["backend", "diffusionPipe", "timestepSampleMethod"],
                  s || null,
                )
              }
              options={DP_TIMESTEP_SAMPLE_OPTIONS}
            />
          </Row>
        </SubGroup>

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

        <SubGroup label="监控">
          <Row label="启用 W&amp;B" description="把 loss / metrics 推送到 wandb。">
            <ToggleSwitch
              checked={v.enableWandb ?? false}
              onCheckedChange={(b) =>
                set(["backend", "diffusionPipe", "enableWandb"], b)
              }
            />
          </Row>
          <Row label="Tracker 名称" description="W&amp;B project / tracker 名。">
            <TextInput
              className="w-64"
              value={v.trackerName ?? ""}
              onChange={(s) =>
                set(["backend", "diffusionPipe", "trackerName"], s || null)
              }
              placeholder="（可选）"
            />
          </Row>
          <Row label="Run 名称" description="W&amp;B run 名；留空自动派生。">
            <TextInput
              className="w-64"
              value={v.runName ?? ""}
              onChange={(s) =>
                set(["backend", "diffusionPipe", "runName"], s || null)
              }
              placeholder="（可选）"
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

const SubGroup = memo(function SubGroup({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="rounded-[4px] border border-border/40 bg-muted/10 p-3 space-y-3.5">
      <div className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.18em]">
        {label}
      </div>
      {children}
    </div>
  )
})

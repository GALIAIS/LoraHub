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
  Row,
  TextInput,
  ToggleSwitch,
} from "../widgets"
import { SubGroup } from "./backend-diffusion-pipe-shared"

type DiffusionPipeValue = NonNullable<
  NonNullable<ConfigFormValue["backend"]>["diffusionPipe"]
>

export function DiffusionPipePerformanceSection({
  value,
  set,
  errorMap,
}: {
  value: DiffusionPipeValue
  set: Setter
  errorMap: ErrorMap
}) {
  return (
    <SubGroup label="性能与切分">
      <Row
        label="Pipeline Stages"
        description="将模型沿层切分到多 GPU 的份数（1 表示不切）。"
        errors={errorMap.get("backend.diffusionPipe.pipelineStages")}
      >
        <IntInput
          min={1}
          value={value.pipelineStages ?? 1}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "pipelineStages"], next ?? 1)
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
          value={value.blocksToSwap ?? 0}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "blocksToSwap"], next ?? 0)
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
          value={value.cachingBatchSize ?? 1}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "cachingBatchSize"], next ?? 1)
          }
        />
      </Row>
      <Row label="Compile" description="启用 torch.compile 加速（PyTorch 2.x）。">
        <ToggleSwitch
          checked={value.compile ?? false}
          onCheckedChange={(checked) =>
            set(["backend", "diffusionPipe", "compile"], checked)
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
          value={value.gradientClipping ?? 1.0}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "gradientClipping"], next ?? 1.0)
          }
        />
      </Row>
      <Row label="Partition Method" description="pipeline 分片策略。">
        <EnumSelect
          value={value.partitionMethod ?? "parameters"}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "partitionMethod"], next)
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
          value={value.stepsPerPrint ?? 1}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "stepsPerPrint"], next ?? 1)
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
          value={(value.partitionSplit ?? []).join(",")}
          onChange={(raw) => {
            const list = raw
              .split(",")
              .map((part) => part.trim())
              .filter((part) => part.length > 0)
              .map((part) => parseInt(part, 10))
              .filter((next) => !Number.isNaN(next))
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
          checked={value.reentrantActivationCheckpointing ?? false}
          onCheckedChange={(checked) =>
            set(
              ["backend", "diffusionPipe", "reentrantActivationCheckpointing"],
              checked,
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
          value={value.forceConstantLr ?? null}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "forceConstantLr"], next)
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
          value={value.uncondFraction ?? 0}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "uncondFraction"], next ?? 0)
          }
        />
      </Row>
      <Row
        label="X-axis Examples"
        description="Tensorboard X 轴用 examples 而不是 steps。"
      >
        <ToggleSwitch
          checked={value.xAxisExamples ?? false}
          onCheckedChange={(checked) =>
            set(["backend", "diffusionPipe", "xAxisExamples"], checked)
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
          value={value.loggingSteps ?? 1}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "loggingSteps"], next ?? 1)
          }
        />
      </Row>
    </SubGroup>
  )
}

export function DiffusionPipeMediaSection({
  value,
  set,
  errorMap,
}: {
  value: DiffusionPipeValue
  set: Setter
  errorMap: ErrorMap
}) {
  return (
    <SubGroup label="混合 image / video">
      <Row
        label="Image Micro Batch Size"
        description="混合训练时单 GPU 图像 micro batch。"
        errors={errorMap.get("backend.diffusionPipe.imageMicroBatchSizePerGpu")}
      >
        <IntInput
          min={1}
          value={value.imageMicroBatchSizePerGpu ?? null}
          onChange={(next) =>
            set(
              ["backend", "diffusionPipe", "imageMicroBatchSizePerGpu"],
              next,
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
          value={value.imageEvalMicroBatchSizePerGpu ?? null}
          onChange={(next) =>
            set(
              ["backend", "diffusionPipe", "imageEvalMicroBatchSizePerGpu"],
              next,
            )
          }
          placeholder="（默认）"
        />
      </Row>
      <Row label="Video Clip Mode" description="视频片段抽取策略。">
        <EnumSelect
          value={value.videoClipMode ?? "single_beginning"}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "videoClipMode"], next)
          }
          options={DP_VIDEO_CLIP_MODE_OPTIONS}
        />
      </Row>
    </SubGroup>
  )
}

export function DiffusionPipeDtypeSection({
  value,
  set,
}: {
  value: DiffusionPipeValue
  set: Setter
}) {
  return (
    <SubGroup label="dtype / 时间步采样">
      <Row label="Transformer dtype">
        <EnumSelect
          value={value.transformerDtype ?? ""}
          onChange={(next) =>
            set(["backend", "diffusionPipe", "transformerDtype"], next || null)
          }
          options={DP_TRANSFORMER_DTYPE_OPTIONS}
        />
      </Row>
      <Row label="Diffusion model dtype">
        <EnumSelect
          value={value.diffusionModelDtype ?? ""}
          onChange={(next) =>
            set(
              ["backend", "diffusionPipe", "diffusionModelDtype"],
              next || null,
            )
          }
          options={DP_DIFFUSION_DTYPE_OPTIONS}
        />
      </Row>
      <Row label="Timestep sample method">
        <EnumSelect
          value={value.timestepSampleMethod ?? ""}
          onChange={(next) =>
            set(
              ["backend", "diffusionPipe", "timestepSampleMethod"],
              next || null,
            )
          }
          options={DP_TIMESTEP_SAMPLE_OPTIONS}
        />
      </Row>
    </SubGroup>
  )
}

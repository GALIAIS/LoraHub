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
 * Editor for `backend.diffusion_pipe` (DiffusionPipeOptions in schema.py).
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
    value: NonNullable<ConfigFormValue["backend"]>["diffusion_pipe"]
    set: Setter
    errorMap: ErrorMap
  }) {
    const v = value ?? {}
    const evalEnabled = v.eval_every_n_epochs !== null && v.eval_every_n_epochs !== undefined
    return (
      <>
        <SubGroup label="性能与切分">
          <Row
            label="Pipeline Stages"
            description="将模型沿层切分到多 GPU 的份数（1 表示不切）。"
            errors={errorMap.get("backend.diffusion_pipe.pipeline_stages")}
          >
            <IntInput
              min={1}
              value={v.pipeline_stages ?? 1}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "pipeline_stages"], n ?? 1)
              }
            />
          </Row>
          <Row
            label="Blocks To Swap"
            description="CPU offload 的 transformer block 数；显存吃紧时调高。"
            errors={errorMap.get("backend.diffusion_pipe.blocks_to_swap")}
          >
            <IntInput
              min={0}
              value={v.blocks_to_swap ?? 0}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "blocks_to_swap"], n ?? 0)
              }
            />
          </Row>
          <Row
            label="Caching Batch Size"
            description="VAE / 文本嵌入预缓存阶段的批大小。"
            errors={errorMap.get("backend.diffusion_pipe.caching_batch_size")}
          >
            <IntInput
              min={1}
              value={v.caching_batch_size ?? 1}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "caching_batch_size"], n ?? 1)
              }
            />
          </Row>
          <Row label="Compile" description="启用 torch.compile 加速（PyTorch 2.x）。">
            <ToggleSwitch
              checked={v.compile ?? false}
              onCheckedChange={(b) =>
                set(["backend", "diffusion_pipe", "compile"], b)
              }
            />
          </Row>
          <Row
            label="Gradient Clipping"
            description="梯度范数裁剪上限。"
            errors={errorMap.get("backend.diffusion_pipe.gradient_clipping")}
          >
            <FloatInput
              step={0.1}
              value={v.gradient_clipping ?? 1.0}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "gradient_clipping"], n ?? 1.0)
              }
            />
          </Row>
          <Row label="Partition Method" description="pipeline 分片策略。">
            <EnumSelect
              value={v.partition_method ?? "parameters"}
              onChange={(t) =>
                set(["backend", "diffusion_pipe", "partition_method"], t)
              }
              options={PARTITION_METHOD_OPTIONS}
            />
          </Row>
          <Row
            label="Steps Per Print"
            description="每多少步 flush 一次训练日志。"
            errors={errorMap.get("backend.diffusion_pipe.steps_per_print")}
          >
            <IntInput
              min={1}
              value={v.steps_per_print ?? 1}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "steps_per_print"], n ?? 1)
              }
            />
          </Row>
          <Row
            label="Partition Split"
            description="manual 切分时各 stage 的层数列表，逗号分隔（长度 = pipeline_stages - 1）。"
            errors={errorMap.get("backend.diffusion_pipe.partition_split")}
          >
            <TextInput
              className="w-64"
              value={(v.partition_split ?? []).join(",")}
              onChange={(s) => {
                const list = s
                  .split(",")
                  .map((x) => x.trim())
                  .filter((x) => x.length > 0)
                  .map((x) => parseInt(x, 10))
                  .filter((n) => !Number.isNaN(n))
                set(
                  ["backend", "diffusion_pipe", "partition_split"],
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
              checked={v.reentrant_activation_checkpointing ?? false}
              onCheckedChange={(b) =>
                set(
                  ["backend", "diffusion_pipe", "reentrant_activation_checkpointing"],
                  b,
                )
              }
            />
          </Row>
          <Row
            label="Force Constant LR"
            description="忽略 scheduler，强制使用恒定 LR（resume 调试用）。"
            errors={errorMap.get("backend.diffusion_pipe.force_constant_lr")}
          >
            <FloatInput
              step={1e-5}
              value={v.force_constant_lr ?? null}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "force_constant_lr"], n)
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="Uncond Fraction"
            description="CFG 风格训练：丢弃 caption 的步数比例（0..1）。"
            errors={errorMap.get("backend.diffusion_pipe.uncond_fraction")}
          >
            <FloatInput
              step={0.05}
              value={v.uncond_fraction ?? 0}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "uncond_fraction"], n ?? 0)
              }
            />
          </Row>
          <Row
            label="X-axis Examples"
            description="Tensorboard X 轴用 examples 而不是 steps。"
          >
            <ToggleSwitch
              checked={v.x_axis_examples ?? false}
              onCheckedChange={(b) =>
                set(["backend", "diffusion_pipe", "x_axis_examples"], b)
              }
            />
          </Row>
          <Row
            label="Logging Steps"
            description="每多少步写一次 wandb / tensorboard。"
            errors={errorMap.get("backend.diffusion_pipe.logging_steps")}
          >
            <IntInput
              min={1}
              value={v.logging_steps ?? 1}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "logging_steps"], n ?? 1)
              }
            />
          </Row>
        </SubGroup>

        <SubGroup label="混合 image / video">
          <Row
            label="Image Micro Batch Size"
            description="混合训练时单 GPU 图像 micro batch。"
            errors={errorMap.get(
              "backend.diffusion_pipe.image_micro_batch_size_per_gpu",
            )}
          >
            <IntInput
              min={1}
              value={v.image_micro_batch_size_per_gpu ?? null}
              onChange={(n) =>
                set(
                  ["backend", "diffusion_pipe", "image_micro_batch_size_per_gpu"],
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
              "backend.diffusion_pipe.image_eval_micro_batch_size_per_gpu",
            )}
          >
            <IntInput
              min={1}
              value={v.image_eval_micro_batch_size_per_gpu ?? null}
              onChange={(n) =>
                set(
                  [
                    "backend",
                    "diffusion_pipe",
                    "image_eval_micro_batch_size_per_gpu",
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
              value={v.video_clip_mode ?? "single_beginning"}
              onChange={(s) =>
                set(["backend", "diffusion_pipe", "video_clip_mode"], s)
              }
              options={DP_VIDEO_CLIP_MODE_OPTIONS}
            />
          </Row>
        </SubGroup>

        <SubGroup label="dtype / 时间步采样">
          <Row label="Transformer dtype">
            <EnumSelect
              value={v.transformer_dtype ?? ""}
              onChange={(s) =>
                set(
                  ["backend", "diffusion_pipe", "transformer_dtype"],
                  s || null,
                )
              }
              options={DP_TRANSFORMER_DTYPE_OPTIONS}
            />
          </Row>
          <Row label="Diffusion model dtype">
            <EnumSelect
              value={v.diffusion_model_dtype ?? ""}
              onChange={(s) =>
                set(
                  ["backend", "diffusion_pipe", "diffusion_model_dtype"],
                  s || null,
                )
              }
              options={DP_DIFFUSION_DTYPE_OPTIONS}
            />
          </Row>
          <Row label="Timestep sample method">
            <EnumSelect
              value={v.timestep_sample_method ?? ""}
              onChange={(s) =>
                set(
                  ["backend", "diffusion_pipe", "timestep_sample_method"],
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
            errors={errorMap.get("backend.diffusion_pipe.eval_every_n_epochs")}
          >
            <div className="flex items-center gap-3">
              <ToggleSwitch
                checked={evalEnabled}
                onCheckedChange={(b) =>
                  set(
                    ["backend", "diffusion_pipe", "eval_every_n_epochs"],
                    b ? 1 : null,
                  )
                }
              />
              {evalEnabled && (
                <IntInput
                  min={1}
                  value={v.eval_every_n_epochs ?? 1}
                  onChange={(n) =>
                    set(
                      ["backend", "diffusion_pipe", "eval_every_n_epochs"],
                      n ?? 1,
                    )
                  }
                />
              )}
            </div>
          </Row>
          <Row
            label="每 N 步验证"
            errors={errorMap.get("backend.diffusion_pipe.eval_every_n_steps")}
          >
            <IntInput
              min={1}
              value={v.eval_every_n_steps ?? null}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "eval_every_n_steps"], n)
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="每 N 样本验证"
            errors={errorMap.get("backend.diffusion_pipe.eval_every_n_examples")}
          >
            <IntInput
              min={1}
              value={v.eval_every_n_examples ?? null}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "eval_every_n_examples"], n)
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="首步前先评估"
            description="第一步训练之前先跑一次评估。"
          >
            <ToggleSwitch
              checked={v.eval_before_first_step ?? false}
              onCheckedChange={(b) =>
                set(["backend", "diffusion_pipe", "eval_before_first_step"], b)
              }
            />
          </Row>
          <Row
            label="评估批大小"
            description="单 GPU 的 micro-batch 大小。"
            errors={errorMap.get(
              "backend.diffusion_pipe.eval_micro_batch_size_per_gpu",
            )}
          >
            <IntInput
              min={1}
              value={v.eval_micro_batch_size_per_gpu ?? 1}
              onChange={(n) =>
                set(
                  ["backend", "diffusion_pipe", "eval_micro_batch_size_per_gpu"],
                  n ?? 1,
                )
              }
            />
          </Row>
          <Row
            label="评估梯度累积"
            description="评估期梯度累积步数（影响 perplexity / loss 平均口径）。"
            errors={errorMap.get(
              "backend.diffusion_pipe.eval_gradient_accumulation_steps",
            )}
          >
            <IntInput
              min={1}
              value={v.eval_gradient_accumulation_steps ?? 1}
              onChange={(n) =>
                set(
                  [
                    "backend",
                    "diffusion_pipe",
                    "eval_gradient_accumulation_steps",
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
              checked={v.disable_block_swap_for_eval ?? false}
              onCheckedChange={(b) =>
                set(
                  ["backend", "diffusion_pipe", "disable_block_swap_for_eval"],
                  b,
                )
              }
            />
          </Row>
          <Row
            label="Eval Datasets"
            description="独立评估数据集列表（JSON 数组，每项 {name, config_path}）。"
            errors={errorMap.get("backend.diffusion_pipe.eval_datasets")}
          >
            <textarea
              value={JSON.stringify(v.eval_datasets ?? [], null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value || "[]")
                  if (Array.isArray(parsed)) {
                    set(
                      ["backend", "diffusion_pipe", "eval_datasets"],
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
            label="checkpoint_every_n_epochs"
            errors={errorMap.get(
              "backend.diffusion_pipe.checkpoint_every_n_epochs",
            )}
          >
            <IntInput
              min={1}
              value={v.checkpoint_every_n_epochs ?? null}
              onChange={(n) =>
                set(
                  ["backend", "diffusion_pipe", "checkpoint_every_n_epochs"],
                  n,
                )
              }
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="checkpoint_every_n_minutes"
            errors={errorMap.get(
              "backend.diffusion_pipe.checkpoint_every_n_minutes",
            )}
          >
            <IntInput
              min={1}
              value={v.checkpoint_every_n_minutes ?? null}
              onChange={(n) =>
                set(
                  ["backend", "diffusion_pipe", "checkpoint_every_n_minutes"],
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
              checked={v.enable_wandb ?? false}
              onCheckedChange={(b) =>
                set(["backend", "diffusion_pipe", "enable_wandb"], b)
              }
            />
          </Row>
          <Row label="Tracker 名称" description="W&amp;B project / tracker 名。">
            <TextInput
              className="w-64"
              value={v.tracker_name ?? ""}
              onChange={(s) =>
                set(["backend", "diffusion_pipe", "tracker_name"], s || null)
              }
              placeholder="（可选）"
            />
          </Row>
          <Row label="Run 名称" description="W&amp;B run 名；留空自动派生。">
            <TextInput
              className="w-64"
              value={v.run_name ?? ""}
              onChange={(s) =>
                set(["backend", "diffusion_pipe", "run_name"], s || null)
              }
              placeholder="（可选）"
            />
          </Row>
        </SubGroup>

        <SubGroup label="AR Bucket">
          <Row
            label="最小宽高比"
            errors={errorMap.get("backend.diffusion_pipe.min_ar")}
          >
            <FloatInput
              step={0.1}
              value={v.min_ar ?? 0.5}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "min_ar"], n ?? 0.5)
              }
            />
          </Row>
          <Row
            label="最大宽高比"
            errors={errorMap.get("backend.diffusion_pipe.max_ar")}
          >
            <FloatInput
              step={0.1}
              value={v.max_ar ?? 2.0}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "max_ar"], n ?? 2.0)
              }
            />
          </Row>
          <Row
            label="AR Bucket 数"
            errors={errorMap.get("backend.diffusion_pipe.num_ar_buckets")}
          >
            <IntInput
              min={1}
              value={v.num_ar_buckets ?? 7}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "num_ar_buckets"], n ?? 7)
              }
            />
          </Row>
        </SubGroup>

        <SubGroup label="模型路径">
          <Row
            label="model_paths"
            description="dp [model] 段的额外路径键值。每行一对，例如 transformer_path = /path/to.safetensors。"
            errors={errorMap.get("backend.diffusion_pipe.model_paths")}
          >
            <KeyValueTextArea
              rows={5}
              value={v.model_paths}
              onChange={(next) =>
                set(["backend", "diffusion_pipe", "model_paths"], next)
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
            errors={errorMap.get("backend.diffusion_pipe.cache_shuffle_num")}
          >
            <IntInput
              min={0}
              value={v.cache_shuffle_num ?? 0}
              onChange={(n) =>
                set(["backend", "diffusion_pipe", "cache_shuffle_num"], n ?? 0)
              }
            />
          </Row>
          <Row
            label="跳过空 Caption"
            description="忽略 caption 为空的样本。"
          >
            <ToggleSwitch
              checked={v.skip_empty_caption ?? true}
              onCheckedChange={(b) =>
                set(["backend", "diffusion_pipe", "skip_empty_caption"], b)
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

import { memo } from "react"
import { PARTITION_METHOD_OPTIONS } from "../options"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
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
    value: NonNullable<RecipeFormValue["backend"]>["diffusion_pipe"]
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

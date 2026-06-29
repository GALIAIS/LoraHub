import { memo } from "react"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { BACKEND_BADGE } from "../backend-meta"
import { BACKEND_OPTIONS } from "../options"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import {
  EnumSelect,
  IntInput,
  KeyValueTextArea,
  PathInput,
  Row,
  Section,
  ToggleSwitch,
} from "../widgets"

const BACKEND_DESCRIPTIONS: Record<string, string> = {
  kohya:
    "kohya-ss/sd-scripts。LoRA / DreamBooth 训练后端。",
  "diffusion-pipe":
    "tdrussell/diffusion-pipe。图像与视频训练后端。",
  anima_lora:
    "sorryhyun/anima_lora。Anima DiT 训练后端。",
  ai_toolkit:
    "ostris/ai-toolkit。Krea2 等模型训练后端。",
}

const REPO_LABEL: Record<string, string> = {
  kohya: "sd-scripts 路径",
  "diffusion-pipe": "diffusion-pipe 路径",
  anima_lora: "anima_lora 路径",
  ai_toolkit: "ai-toolkit 路径",
}

const REPO_PLACEHOLDER: Record<string, string> = {
  kohya: "设置默认值",
  "diffusion-pipe": "设置默认值",
  anima_lora: "./external/anima_lora",
  ai_toolkit: "./external/ai_toolkit",
}

const GPU_DISPATCH_OPTIONS = [
  { value: "settings", label: "跟随设置" },
  { value: "one-job-per-gpu", label: "一任务一 GPU" },
  { value: "distributed", label: "单任务多 GPU" },
]

const DISTRIBUTED_STRATEGY_OPTIONS = [
  { value: "ddp", label: "DDP 数据并行" },
  { value: "fsdp", label: "FSDP 参数分片" },
  { value: "deepspeed_zero", label: "DeepSpeed ZeRO" },
]

const FSDP_SHARDING_OPTIONS = [
  { value: "full_shard", label: "Full shard" },
  { value: "shard_grad_op", label: "Shard grad op" },
  { value: "no_reshard", label: "No shard" },
]

const FSDP_WRAP_OPTIONS = [
  { value: "size_based", label: "按参数量自动包裹" },
  { value: "transformer", label: "Transformer 层包裹" },
  { value: "none", label: "不自动包裹" },
]

const FSDP_STATE_DICT_OPTIONS = [
  { value: "full_state_dict", label: "Full state dict" },
  { value: "sharded_state_dict", label: "Sharded state dict" },
  { value: "local_state_dict", label: "Local state dict" },
]

const ZERO_STAGE_OPTIONS = [
  { value: "2", label: "ZeRO-2" },
  { value: "3", label: "ZeRO-3" },
]

const ZERO_OFFLOAD_OPTIONS = [
  { value: "none", label: "不 offload" },
  { value: "cpu", label: "CPU offload" },
]

export const BackendFields = memo(function BackendFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["backend"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const type = v.type ?? "kohya"
  const badge = BACKEND_BADGE[type as keyof typeof BACKEND_BADGE]
  const description =
    BACKEND_DESCRIPTIONS[type] ?? BACKEND_DESCRIPTIONS.kohya
  const repoLabel = REPO_LABEL[type] ?? "仓库路径"
  const repoPlaceholder = REPO_PLACEHOLDER[type] ?? "（使用设置中的默认值)"
  const isDistributed = v.gpuDispatch?.mode === "distributed"
  const distributedStrategy = v.distributed?.strategy ?? "ddp"

  return (
    <>
      <Row label="后端" description={description}>
        <div className="flex items-center gap-2 flex-wrap">
          <EnumSelect
            value={type}
            onChange={(t) => set(["backend", "type"], t)}
            options={BACKEND_OPTIONS}
          />
          {badge && (
            <Badge
              variant="outline"
              className={`rounded-[2px] uppercase text-[10px] ${badge.toneClass}`}
            >
              {badge.label}
            </Badge>
          )}
          {type === "anima_lora" && (
            <Badge
              variant="outline"
              className="rounded-[2px] uppercase text-[10px] border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
              title="随 LoraHub 分发"
            >
              vendored
            </Badge>
          )}
          {type === "ai_toolkit" && (
            <Badge
              variant="outline"
              className="rounded-[2px] uppercase text-[10px] border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
              title="随 LoraHub 分发"
            >
              vendored
            </Badge>
          )}
        </div>
      </Row>
      <Row
        label="GPU 调度"
        description={
          type === "kohya"
            ? "kohya 使用一任务一 GPU。"
            : "选择任务到 GPU 的分配方式。"
        }
        errors={errorMap.get("backend.gpuDispatch.mode")}
      >
        <div className="flex flex-wrap items-center gap-2">
          <EnumSelect
            value={v.gpuDispatch?.mode ?? "settings"}
            onChange={(mode) => {
              if (mode === "settings") {
                set(["backend", "gpuDispatch"], undefined)
                set(["backend", "distributed", "strategy"], "ddp")
              } else {
                set(["backend", "gpuDispatch", "mode"], mode)
                if (mode !== "distributed") {
                  set(["backend", "distributed", "strategy"], "ddp")
                }
              }
            }}
            options={GPU_DISPATCH_OPTIONS}
          />
          {v.gpuDispatch?.mode === "distributed" && (
            <IntInput
              value={v.gpuDispatch?.numGpus ?? null}
              min={1}
              onChange={(n) => set(["backend", "gpuDispatch", "numGpus"], n)}
              placeholder="全部"
              className="w-24"
            />
          )}
        </div>
      </Row>
      {isDistributed && (
        <>
          <Row
            label="分布式策略"
            description={
              type === "anima_lora"
                ? "DDP 复制模型；FSDP/ZeRO 分片参数或优化器状态。"
                : "当前后端使用 DDP。"
            }
            errors={errorMap.get("backend.distributed.strategy")}
          >
            <EnumSelect
              value={distributedStrategy}
              onChange={(s) => set(["backend", "distributed", "strategy"], s)}
              options={DISTRIBUTED_STRATEGY_OPTIONS}
            />
          </Row>
          {distributedStrategy === "fsdp" && (
            <>
              <Row
                label="FSDP 分片"
                errors={errorMap.get("backend.distributed.fsdp.shardingStrategy")}
              >
                <EnumSelect
                  value={v.distributed?.fsdp?.shardingStrategy ?? "full_shard"}
                  onChange={(s) =>
                    set(["backend", "distributed", "fsdp", "shardingStrategy"], s)
                  }
                  options={FSDP_SHARDING_OPTIONS}
                />
              </Row>
              <Row
                label="FSDP 包裹"
                description="FSDP 自动包裹策略。"
                errors={errorMap.get("backend.distributed.fsdp.autoWrapPolicy")}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <EnumSelect
                    value={v.distributed?.fsdp?.autoWrapPolicy ?? "size_based"}
                    onChange={(s) =>
                      set(["backend", "distributed", "fsdp", "autoWrapPolicy"], s)
                    }
                    options={FSDP_WRAP_OPTIONS}
                  />
                  {(v.distributed?.fsdp?.autoWrapPolicy ?? "size_based") ===
                    "size_based" && (
                    <IntInput
                      value={v.distributed?.fsdp?.minNumParams ?? 100_000_000}
                      min={0}
                      onChange={(n) =>
                        set(
                          ["backend", "distributed", "fsdp", "minNumParams"],
                          n ?? 100_000_000,
                        )
                      }
                      className="w-36"
                    />
                  )}
                </div>
              </Row>
              <Row
                label="FSDP 保存"
                errors={errorMap.get("backend.distributed.fsdp.stateDictType")}
              >
                <EnumSelect
                  value={v.distributed?.fsdp?.stateDictType ?? "full_state_dict"}
                  onChange={(s) =>
                    set(["backend", "distributed", "fsdp", "stateDictType"], s)
                  }
                  options={FSDP_STATE_DICT_OPTIONS}
                />
              </Row>
              <Row
                label="FSDP CPU offload"
                description="将 FSDP 参数 offload 到 CPU。"
                errors={errorMap.get("backend.distributed.fsdp.cpuOffload")}
              >
                <ToggleSwitch
                  checked={!!v.distributed?.fsdp?.cpuOffload}
                  onCheckedChange={(c) =>
                    set(["backend", "distributed", "fsdp", "cpuOffload"], c)
                  }
                />
              </Row>
            </>
          )}
          {distributedStrategy === "deepspeed_zero" && (
            <>
              <Row
                label="ZeRO stage"
                errors={errorMap.get("backend.distributed.zero.stage")}
              >
                <EnumSelect
                  value={String(v.distributed?.zero?.stage ?? 2)}
                  onChange={(s) =>
                    set(["backend", "distributed", "zero", "stage"], parseInt(s, 10))
                  }
                  options={ZERO_STAGE_OPTIONS}
                />
              </Row>
              <Row
                label="ZeRO offload"
                description="ZeRO 优化器 / 参数 offload。"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <EnumSelect
                    value={v.distributed?.zero?.offloadOptimizer ?? "none"}
                    onChange={(s) =>
                      set(["backend", "distributed", "zero", "offloadOptimizer"], s)
                    }
                    options={ZERO_OFFLOAD_OPTIONS}
                  />
                  <EnumSelect
                    value={v.distributed?.zero?.offloadParam ?? "none"}
                    onChange={(s) =>
                      set(["backend", "distributed", "zero", "offloadParam"], s)
                    }
                    options={ZERO_OFFLOAD_OPTIONS}
                  />
                </div>
              </Row>
              <Row
                label="ZeRO overlap comm"
                errors={errorMap.get("backend.distributed.zero.overlapComm")}
              >
                <ToggleSwitch
                  checked={v.distributed?.zero?.overlapComm ?? true}
                  onCheckedChange={(c) =>
                    set(["backend", "distributed", "zero", "overlapComm"], c)
                  }
                />
              </Row>
            </>
          )}
        </>
      )}
      <Section title="后端高级" subtitle="仓库路径 / Python / 额外参数">
        <Row label={repoLabel} errors={errorMap.get("backend.repoPath")}>
          <PathInput
            value={v.repoPath ?? v.sdScriptsPath ?? ""}
            onChange={(s) => set(["backend", "repoPath"], s || null)}
            placeholder={repoPlaceholder}
          />
        </Row>
        <Row label="Python 解释器" errors={errorMap.get("backend.pythonExecutable")}>
          <PathInput
            value={v.pythonExecutable ?? ""}
            onChange={(s) => set(["backend", "pythonExecutable"], s || null)}
            placeholder={
              type === "anima_lora" || type === "ai_toolkit"
                ? ".venv/bin/python"
                : "设置默认值"
            }
          />
        </Row>
        <Row
          label="锁定版本"
          description="兼容旧配置。当前安装器不读取该字段。"
        >
          <Input
            value={v.pinVersion ?? ""}
            className="font-mono w-64"
            onChange={(e) => set(["backend", "pinVersion"], e.target.value || null)}
            placeholder="未启用"
            disabled
          />
        </Row>
        <Row
          label="额外参数"
          description={
            type === "kohya"
              ? "透传给 sd-scripts 的额外 CLI flag。每行一条,key=value。bool flag 写 key=true 即可,store_true 由 compiler 兼容。"
              : type === "anima_lora"
                ? "写入 Anima TOML 顶层。每行 key=value。"
                : type === "ai_toolkit"
                  ? "写入 ai-toolkit YAML。支持点号路径,每行 key=value。"
                  : "追加到 diffusion-pipe TOML 顶层。每行 key=value。"
          }
        >
          <KeyValueTextArea
            value={
              v.extraArgs
                ? Object.fromEntries(
                    Object.entries(v.extraArgs).map(([k, val]) => [
                      k,
                      val === true
                        ? "true"
                        : val === false
                          ? "false"
                          : val == null
                            ? ""
                            : String(val),
                    ]),
                  )
                : {}
            }
            onChange={(next) => {
              const out: Record<string, unknown> = {}
              for (const [k, val] of Object.entries(next)) {
                const trimmed = val.trim()
                if (trimmed === "") {
                  out[k] = true
                } else if (trimmed === "true") {
                  out[k] = true
                } else if (trimmed === "false") {
                  out[k] = false
                } else {
                  out[k] = trimmed
                }
              }
              set(["backend", "extraArgs"], out)
            }}
            placeholder={
              type === "kohya"
                ? "network_train_unet_only=true\ngradient_accumulation_steps=4"
                : "key=value"
            }
            rows={4}
          />
        </Row>
      </Section>
    </>
  )
})

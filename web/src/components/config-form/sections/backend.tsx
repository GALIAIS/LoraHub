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
  ToggleSwitch,
} from "../widgets"

const BACKEND_DESCRIPTIONS: Record<string, string> = {
  kohya:
    "kohya-ss/sd-scripts。SD1.5 / SD2 / SDXL / SD3 / FLUX / Lumina / HunyuanImage / Anima 通用 LoRA / DreamBooth 训练。",
  "diffusion-pipe":
    "tdrussell/diffusion-pipe。DeepSpeed 流水并行,涵盖图像与视频(Wan / HunyuanVideo / LTX / Cosmos 等)。",
  anima_lora:
    "sorryhyun/anima_lora(随 LoraHub vendored)。仅训练 Anima DiT,带 OrthoLoRA / T-LoRA / Hydra / postfix / EasyControl / IP-Adapter / DMD turbo 蒸馏。",
}

const REPO_LABEL: Record<string, string> = {
  kohya: "sd-scripts 路径",
  "diffusion-pipe": "diffusion-pipe 路径",
  anima_lora: "anima_lora 路径",
}

const REPO_PLACEHOLDER: Record<string, string> = {
  kohya: "（使用设置中的默认值)",
  "diffusion-pipe": "（使用设置中的默认值)",
  anima_lora: "（默认 ./external/anima_lora,通常无需填写)",
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
              title="anima_lora 源码随 LoraHub 一起分发,不需要单独 clone"
            >
              vendored
            </Badge>
          )}
        </div>
      </Row>
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
            type === "anima_lora"
              ? "（默认 .venv/bin/python — 用 uv sync 后自动指向)"
              : "（使用设置中的默认值)"
          }
        />
      </Row>
      <Row
        label="GPU 调度"
        description={
          type === "kohya"
            ? "kohya 当前仅支持一任务一 GPU。多卡机器可并行跑多个训练任务。"
            : "默认跟随「设置 → 概览」。单任务多 GPU 只会使用同型号/同显存 GPU 组，4080 + V100 这类异构卡默认不混跑。"
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
                ? "DDP 复制模型到每张卡；FSDP/ZeRO 会分片参数或优化器状态，能降低单卡显存峰值但通信开销更高。"
                : "当前只有 anima_lora 接入 FSDP / DeepSpeed ZeRO；其他后端请保持 DDP。"
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
                description="默认按大模块参数量自动包裹，更适合 anima_lora 这种非标准 Transformers DiT。"
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
                description="只有显存仍不够时再开；会明显增加 CPU/PCIe 压力。"
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
                description="ZeRO-3 + 参数 offload 能继续降显存，但速度通常明显下降。"
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
      <Row
        label="锁定版本"
        description="schema-only 字段：保留以兼容旧 YAML,但当前 installer 不会读取。如需锁定到特定 git ref,手动 cd 到后端仓库后 git checkout。"
      >
        <Input
          value={v.pinVersion ?? ""}
          className="font-mono w-64"
          onChange={(e) => set(["backend", "pinVersion"], e.target.value || null)}
          placeholder="(未实装,仅 YAML 占位)"
          disabled
        />
      </Row>
      <Row
        label="额外参数"
        description={
          type === "kohya"
            ? "透传给 sd-scripts 的额外 CLI flag。每行一条,key=value。bool flag 写 key=true 即可,store_true 由 compiler 兼容。"
            : type === "anima_lora"
              ? "写入 _lorahub_anima_config.toml 顶层的额外字段。每行一条,key=value。"
              : "追加到 diffusion-pipe TOML 顶层的额外字段。每行一条,key=value。"
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
    </>
  )
})

/**
 * anima_lora-specific config section.
 *
 * Only visible when `backend.type === "anima_lora"`. Mirrors every field
 * on `AnimaLoraOptions` in `lorahub/core/config/schema.py`.
 *
 * Layout follows the upstream merge chain so users edit in roughly the
 * same order they'd find a knob in `configs/methods/lora.toml` /
 * `configs/base.toml`:
 *   1. Method + preset (the two core choices)
 *   2. Network / optim / sampling — the train.py argv overrides
 *   3. Memory + attn + compile (the perf tradeoff knobs)
 *   4. Method sub-config (visible only when the matching method is selected)
 *   5. Turbo distillation (orthogonal — when populated, switches the whole
 *      pipeline from train.py to scripts/distill_turbo.py)
 */
import { memo, useCallback } from "react"
import { Sparkles } from "lucide-react"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import {
  EnumSelect,
  FloatInput,
  PathInput,
  Row,
  Section,
  ToggleSwitch,
} from "../widgets"

const METHOD_OPTIONS = [
  { value: "lora", label: "lora — LoRA + OrthoLoRA + T-LoRA(默认堆叠)" },
  { value: "postfix", label: "postfix — 自由参数 / 条件正交后缀" },
  { value: "chimera", label: "chimera — 双池路由 MoE" },
  { value: "easycontrol", label: "easycontrol — 自注意力图像条件" },
  {
    value: "ip_adapter",
    label: "ip_adapter — 图像交叉注意力(PE-Core encoder)",
  },
] as const

const PRESET_OPTIONS = [
  { value: "default", label: "default — 标准 24GB" },
  { value: "low_vram", label: "low_vram — 8GB(grad ckpt + unsloth offload)" },
  { value: "graft", label: "graft — blocks_to_swap=20" },
  { value: "half", label: "half — 50% 数据(实验)" },
  { value: "quarter", label: "quarter — 25% 数据" },
  { value: "tenth", label: "tenth — 10% 数据" },
  { value: "debug", label: "debug — 0.1% 数据(打通管线)" },
] as const

const TIMESTEP_OPTIONS = [
  { value: "sigmoid", label: "sigmoid(默认)" },
  { value: "uniform", label: "uniform" },
  { value: "logit_normal", label: "logit_normal" },
] as const

const ATTN_OPTIONS = [
  { value: "flash", label: "flash(默认,需装 flash-attn)" },
  { value: "torch", label: "torch — SDPA(无 flash-attn 时选这个)" },
  { value: "flex", label: "flex — FlexAttention" },
  { value: "sageattn", label: "sageattn(仅推理)" },
  { value: "xformers", label: "xformers" },
] as const

const COMPILE_MODE_OPTIONS = [
  { value: "", label: "（关闭,默认)" },
  { value: "blocks", label: "blocks — 分块编译(可与 grad ckpt 共存)" },
  {
    value: "full",
    label: "full — 全图编译(与 grad ckpt / blocks_to_swap 互斥)",
  },
] as const

const COMPILE_INDUCTOR_OPTIONS = [
  { value: "", label: "（默认)" },
  { value: "default", label: "default" },
  { value: "reduce-overhead", label: "reduce-overhead(推荐)" },
  { value: "max-autotune", label: "max-autotune" },
] as const

const MIXED_PRECISION_OPTIONS = [
  { value: "bf16", label: "bf16(默认)" },
  { value: "fp16", label: "fp16" },
  { value: "fp32", label: "fp32" },
] as const

const OPTIMIZER_OPTIONS = [
  { value: "AdamW", label: "AdamW(默认)" },
  { value: "AdamW8bit", label: "AdamW8bit" },
  { value: "Lion", label: "Lion" },
  { value: "Prodigy", label: "Prodigy" },
] as const

const LR_SCHEDULER_OPTIONS = [
  { value: "constant", label: "constant(默认)" },
  { value: "cosine", label: "cosine" },
  { value: "cosine_with_restarts", label: "cosine_with_restarts" },
  { value: "linear", label: "linear" },
  { value: "polynomial", label: "polynomial" },
] as const

export const BackendAnimaLoraFields = memo(function BackendAnimaLoraFields({
  value = {},
  set,
  errorMap,
}: {
  value: NonNullable<ConfigFormValue["backend"]>["animaLora"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const method = v.method ?? "lora"

  // Switching method clears the previously-active sub-config so the
  // model_validator on the Python side doesn't reject "method=postfix
  // but ipAdapter sub-config also set" (it'd accept it but it's noise).
  const onMethodChange = useCallback(
    (next: string) => {
      set(["backend", "animaLora", "method"], next)
    },
    [set],
  )

  return (
    <>
      <Row
        label="训练方法"
        required
        description="lora 是默认堆叠 LoRA+OrthoLoRA+T-LoRA;其他四种是上游论文级算法,选了对应方法后下方会展开它的子配置。"
        errors={errorMap.get("backend.animaLora.method")}
      >
        <EnumSelect value={method} onChange={onMethodChange} options={METHOD_OPTIONS} />
      </Row>
      <Row
        label="硬件预设"
        description="对应 anima_lora/configs/presets.toml 的 section。debug 预设只取 0.1% 数据,适合打通管线。"
        errors={errorMap.get("backend.animaLora.preset")}
      >
        <EnumSelect
          value={v.preset ?? "default"}
          onChange={(p) => set(["backend", "animaLora", "preset"], p)}
          options={PRESET_OPTIONS}
        />
      </Row>

      <Row label="输出名" errors={errorMap.get("backend.animaLora.outputName")}>
        <PathInput
          value={v.outputName ?? ""}
          onChange={(s) => set(["backend", "animaLora", "outputName"], s || undefined)}
          placeholder="anima_lora"
        />
      </Row>

      {/* === 通用网络参数 === */}
      <Section
        icon={<Sparkles className="size-3.5" />}
        title="网络容量"
        subtitle="LoRA rank / alpha"
      >
        <Row label="network_dim(rank)" errors={errorMap.get("backend.animaLora.networkDim")}>
          <FloatInput
            value={v.networkDim}
            onChange={(n) => set(["backend", "animaLora", "networkDim"], n)}
            placeholder="16"
            min={1}
          />
        </Row>
        <Row label="network_alpha" errors={errorMap.get("backend.animaLora.networkAlpha")}>
          <FloatInput
            value={v.networkAlpha}
            onChange={(n) => set(["backend", "animaLora", "networkAlpha"], n)}
            placeholder="16"
            min={1}
          />
        </Row>
        <Row
          label="只训练 unet"
          description="anima_lora 默认开启 — text encoder 不训练。"
        >
          <ToggleSwitch
            checked={v.networkTrainUnetOnly ?? true}
            onCheckedChange={(c) => set(["backend", "animaLora", "networkTrainUnetOnly"], c)}
          />
        </Row>
      </Section>

      {/* === 优化器 + 调度 === */}
      <Section title="优化器 / 学习率 / 调度">
        <Row label="optimizer_type">
          <EnumSelect
            value={v.optimizerType ?? "AdamW"}
            onChange={(s) => set(["backend", "animaLora", "optimizerType"], s)}
            options={OPTIMIZER_OPTIONS}
          />
        </Row>
        <Row label="lr_scheduler">
          <EnumSelect
            value={v.lrScheduler ?? "constant"}
            onChange={(s) => set(["backend", "animaLora", "lrScheduler"], s)}
            options={LR_SCHEDULER_OPTIONS}
          />
        </Row>
        <Row label="learning_rate" errors={errorMap.get("backend.animaLora.learningRate")}>
          <FloatInput
            value={v.learningRate}
            onChange={(n) => set(["backend", "animaLora", "learningRate"], n)}
            placeholder="5e-5"
            step={1e-6}
          />
        </Row>
        <Row label="max_train_epochs">
          <FloatInput
            value={v.maxTrainEpochs}
            onChange={(n) => set(["backend", "animaLora", "maxTrainEpochs"], n)}
            placeholder="8"
            min={1}
          />
        </Row>
        <Row label="save_every_n_epochs">
          <FloatInput
            value={v.saveEveryNEpochs}
            onChange={(n) => set(["backend", "animaLora", "saveEveryNEpochs"], n)}
            placeholder="2"
            min={1}
          />
        </Row>
        <Row label="checkpointing_epochs" description="保存 optimizer state 的频率(用于断点续训)">
          <FloatInput
            value={v.checkpointingEpochs}
            onChange={(n) => set(["backend", "animaLora", "checkpointingEpochs"], n)}
            placeholder="2"
            min={1}
          />
        </Row>
        <Row label="caption_dropout_rate">
          <FloatInput
            value={v.captionDropoutRate}
            onChange={(n) => set(["backend", "animaLora", "captionDropoutRate"], n)}
            placeholder="0.1"
            step={0.01}
            min={0}
            max={1}
          />
        </Row>
      </Section>

      {/* === 流匹配采样 === */}
      <Section title="流匹配采样" subtitle="Anima DiT 的 timestep + 损失权重">
        <Row label="timestep_sampling">
          <EnumSelect
            value={v.timestepSampling ?? "sigmoid"}
            onChange={(s) => set(["backend", "animaLora", "timestepSampling"], s)}
            options={TIMESTEP_OPTIONS}
          />
        </Row>
        <Row label="sigmoid_scale">
          <FloatInput
            value={v.sigmoidScale}
            onChange={(n) => set(["backend", "animaLora", "sigmoidScale"], n)}
            placeholder="1.0"
            step={0.1}
          />
        </Row>
        <Row label="discrete_flow_shift">
          <FloatInput
            value={v.discreteFlowShift}
            onChange={(n) => set(["backend", "animaLora", "discreteFlowShift"], n)}
            placeholder="1.0"
            step={0.1}
          />
        </Row>
        <Row
          label="vr_loss_weight"
          description="可选 AsymFlow §5.2 方差减少损失。+40% step 计算成本,留空关闭。"
        >
          <FloatInput
            value={v.vrLossWeight ?? undefined}
            onChange={(n) =>
              set(["backend", "animaLora", "vrLossWeight"], n ?? null)
            }
            placeholder="（关闭)"
            step={0.1}
            min={0}
          />
        </Row>
      </Section>

      {/* === 缓存 === */}
      <Section title="缓存" subtitle="latent / TE / LLM adapter 输出落盘">
        <Row label="cache_latents">
          <ToggleSwitch
            checked={v.cacheLatents ?? true}
            onCheckedChange={(c) => set(["backend", "animaLora", "cacheLatents"], c)}
          />
        </Row>
        <Row label="cache_latents_to_disk">
          <ToggleSwitch
            checked={v.cacheLatentsToDisk ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cacheLatentsToDisk"], c)
            }
          />
        </Row>
        <Row label="cache_text_encoder_outputs">
          <ToggleSwitch
            checked={v.cacheTextEncoderOutputs ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cacheTextEncoderOutputs"], c)
            }
          />
        </Row>
        <Row label="cache_text_encoder_outputs_to_disk">
          <ToggleSwitch
            checked={v.cacheTextEncoderOutputsToDisk ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cacheTextEncoderOutputsToDisk"], c)
            }
          />
        </Row>
        <Row label="cache_llm_adapter_outputs">
          <ToggleSwitch
            checked={v.cacheLlmAdapterOutputs ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cacheLlmAdapterOutputs"], c)
            }
          />
        </Row>
        <Row label="use_shuffled_caption_variants">
          <ToggleSwitch
            checked={v.useShuffledCaptionVariants ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "useShuffledCaptionVariants"], c)
            }
          />
        </Row>
        <Row label="static_token_count" description="Anima 必须 4096">
          <FloatInput
            value={v.staticTokenCount}
            onChange={(n) => set(["backend", "animaLora", "staticTokenCount"], n)}
            placeholder="4096"
            min={1}
          />
        </Row>
        <Row label="vae_chunk_size">
          <FloatInput
            value={v.vaeChunkSize}
            onChange={(n) => set(["backend", "animaLora", "vaeChunkSize"], n)}
            placeholder="64"
            min={1}
          />
        </Row>
        <Row label="vae_disable_cache">
          <ToggleSwitch
            checked={v.vaeDisableCache ?? false}
            onCheckedChange={(c) => set(["backend", "animaLora", "vaeDisableCache"], c)}
          />
        </Row>
      </Section>

      {/* === 注意力 / 编译 === */}
      <Section title="注意力 / torch.compile" subtitle="性能权衡旋钮">
        <Row label="attn_mode" description="不装 flash-attn 时选 torch">
          <EnumSelect
            value={v.attnMode ?? "flash"}
            onChange={(s) => set(["backend", "animaLora", "attnMode"], s)}
            options={ATTN_OPTIONS}
          />
        </Row>
        <Row
          label="compile_mode"
          description="full 与 gradient_checkpointing / blocks_to_swap 互斥(LoraHub 编译期会校验)"
        >
          <EnumSelect
            value={v.compileMode ?? ""}
            onChange={(s) =>
              set(["backend", "animaLora", "compileMode"], s ? s : null)
            }
            options={COMPILE_MODE_OPTIONS}
          />
        </Row>
        <Row label="compile_inductor_mode">
          <EnumSelect
            value={v.compileInductorMode ?? ""}
            onChange={(s) =>
              set(
                ["backend", "animaLora", "compileInductorMode"],
                s ? s : null,
              )
            }
            options={COMPILE_INDUCTOR_OPTIONS}
          />
        </Row>
        <Row label="use_custom_down_autograd" description="anima_lora 自定义内存优化 autograd,默认开">
          <ToggleSwitch
            checked={v.useCustomDownAutograd ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "useCustomDownAutograd"], c)
            }
          />
        </Row>
      </Section>

      {/* === 显存 / offload === */}
      <Section title="显存 / offload" subtitle="低显存训练相关">
        <Row label="blocks_to_swap" description="0=关。graft preset 默认 20。">
          <FloatInput
            value={v.blocksToSwap}
            onChange={(n) => set(["backend", "animaLora", "blocksToSwap"], n)}
            placeholder="0"
            min={0}
          />
        </Row>
        <Row label="gradient_checkpointing">
          <ToggleSwitch
            checked={v.gradientCheckpointing ?? false}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "gradientCheckpointing"], c)
            }
          />
        </Row>
        <Row label="unsloth_offload_checkpointing" description="低显存预设的杀手锏">
          <ToggleSwitch
            checked={v.unslothOffloadCheckpointing ?? false}
            onCheckedChange={(c) =>
              set(
                ["backend", "animaLora", "unslothOffloadCheckpointing"],
                c,
              )
            }
          />
        </Row>
        <Row label="cpu_offload_checkpointing">
          <ToggleSwitch
            checked={v.cpuOffloadCheckpointing ?? false}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cpuOffloadCheckpointing"], c)
            }
          />
        </Row>
        <Row label="mixed_precision">
          <EnumSelect
            value={v.mixedPrecision ?? "bf16"}
            onChange={(s) => set(["backend", "animaLora", "mixedPrecision"], s)}
            options={MIXED_PRECISION_OPTIONS}
          />
        </Row>
      </Section>

      {/* === Method = lora 子配置 === */}
      {method === "lora" && (
        <Section
          icon={<Sparkles className="size-3.5" />}
          title="method=lora 子配置"
          subtitle="OrthoLoRA + T-LoRA 默认堆叠"
          defaultOpen
        >
          <Row label="use_ortho" description="OrthoLoRA — SVD 参数化 + 正交正则">
            <ToggleSwitch
              checked={v.lora?.useOrtho ?? true}
              onCheckedChange={(c) =>
                set(["backend", "animaLora", "lora", "useOrtho"], c)
              }
            />
          </Row>
          <Row label="use_timestep_mask" description="T-LoRA — 时间步相关 rank mask">
            <ToggleSwitch
              checked={v.lora?.useTimestepMask ?? true}
              onCheckedChange={(c) =>
                set(["backend", "animaLora", "lora", "useTimestepMask"], c)
              }
            />
          </Row>
          <Row label="min_rank">
            <FloatInput
              value={v.lora?.minRank}
              onChange={(n) =>
                set(["backend", "animaLora", "lora", "minRank"], n)
              }
              placeholder="8"
              min={1}
            />
          </Row>
          <Row label="alpha_rank_scale">
            <FloatInput
              value={v.lora?.alphaRankScale}
              onChange={(n) =>
                set(["backend", "animaLora", "lora", "alphaRankScale"], n)
              }
              placeholder="1.0"
              step={0.1}
            />
          </Row>
        </Section>
      )}

      {/* === Method = postfix 子配置 === */}
      {method === "postfix" && (
        <Section title="method=postfix 子配置" defaultOpen>
          <Row label="mode">
            <EnumSelect
              value={v.postfix?.mode ?? "cond"}
              onChange={(s) => set(["backend", "animaLora", "postfix", "mode"], s)}
              options={[
                { value: "cond", label: "cond — caption-conditional" },
                { value: "postfix", label: "postfix — 自由 K×D 张量" },
              ]}
            />
          </Row>
          <Row label="cond_hidden_dim">
            <FloatInput
              value={v.postfix?.condHiddenDim}
              onChange={(n) =>
                set(["backend", "animaLora", "postfix", "condHiddenDim"], n)
              }
              placeholder="1024"
              min={1}
            />
          </Row>
          <Row label="lambda_init">
            <FloatInput
              value={v.postfix?.lambdaInit}
              onChange={(n) =>
                set(["backend", "animaLora", "postfix", "lambdaInit"], n)
              }
              placeholder="0.3"
              step={0.05}
            />
          </Row>
          <Row label="te_cache_dir">
            <PathInput
              value={v.postfix?.teCacheDir ?? ""}
              onChange={(s) =>
                set(
                  ["backend", "animaLora", "postfix", "teCacheDir"],
                  s || null,
                )
              }
              placeholder="post_image_dataset/lora"
            />
          </Row>
        </Section>
      )}

      {/* === Method = chimera 子配置 === */}
      {method === "chimera" && (
        <Section title="method=chimera 子配置" defaultOpen>
          <Row label="balance_w_content">
            <FloatInput
              value={v.chimera?.balanceWContent}
              onChange={(n) =>
                set(["backend", "animaLora", "chimera", "balanceWContent"], n)
              }
              placeholder="2e-7"
              step={1e-8}
            />
          </Row>
          <Row label="balance_w_freq">
            <FloatInput
              value={v.chimera?.balanceWFreq}
              onChange={(n) =>
                set(["backend", "animaLora", "chimera", "balanceWFreq"], n)
              }
              placeholder="5e-7"
              step={1e-8}
            />
          </Row>
          <Row label="balance_loss_warmup_ratio">
            <FloatInput
              value={v.chimera?.balanceLossWarmupRatio}
              onChange={(n) =>
                set(
                  ["backend", "animaLora", "chimera", "balanceLossWarmupRatio"],
                  n,
                )
              }
              placeholder="0.4"
              step={0.05}
              min={0}
              max={1}
            />
          </Row>
        </Section>
      )}

      {/* === Method = easycontrol 子配置 === */}
      {method === "easycontrol" && (
        <Section title="method=easycontrol 子配置" defaultOpen>
          <Row label="b_cond_init" description="softmax gate 初始值,-10 让 step 0 等同 baseline DiT">
            <FloatInput
              value={v.easycontrol?.bCondInit}
              onChange={(n) =>
                set(["backend", "animaLora", "easycontrol", "bCondInit"], n)
              }
              placeholder="-10"
              step={0.5}
            />
          </Row>
          <Row label="cond_token_count">
            <FloatInput
              value={v.easycontrol?.condTokenCount}
              onChange={(n) =>
                set(
                  ["backend", "animaLora", "easycontrol", "condTokenCount"],
                  n,
                )
              }
              placeholder="4096"
              min={1}
            />
          </Row>
          <Row label="apply_ffn_lora">
            <ToggleSwitch
              checked={v.easycontrol?.applyFfnLora ?? true}
              onCheckedChange={(c) =>
                set(
                  ["backend", "animaLora", "easycontrol", "applyFfnLora"],
                  c,
                )
              }
            />
          </Row>
          <Row label="drop_p" description="image-CFG dropout">
            <FloatInput
              value={v.easycontrol?.dropP}
              onChange={(n) =>
                set(["backend", "animaLora", "easycontrol", "dropP"], n)
              }
              placeholder="0.1"
              step={0.01}
              min={0}
              max={1}
            />
          </Row>
        </Section>
      )}

      {/* === Method = ip_adapter 子配置 === */}
      {method === "ip_adapter" && (
        <Section title="method=ip_adapter 子配置" defaultOpen>
          <Row label="encoder">
            <EnumSelect
              value={v.ipAdapter?.encoder ?? "PE-Core-L14-336"}
              onChange={(s) =>
                set(["backend", "animaLora", "ipAdapter", "encoder"], s)
              }
              options={[
                { value: "PE-Core-L14-336", label: "PE-Core-L14-336(默认)" },
                { value: "PE-Core-G14-448", label: "PE-Core-G14-448" },
              ]}
            />
          </Row>
          <Row label="ip_scale">
            <FloatInput
              value={v.ipAdapter?.ipScale}
              onChange={(n) =>
                set(["backend", "animaLora", "ipAdapter", "ipScale"], n)
              }
              placeholder="1.0"
              step={0.1}
            />
          </Row>
          <Row label="gate_lr" description="门 LR,通常 10× 全局 LR">
            <FloatInput
              value={v.ipAdapter?.gateLr}
              onChange={(n) =>
                set(["backend", "animaLora", "ipAdapter", "gateLr"], n)
              }
              placeholder="1e-3"
              step={1e-4}
            />
          </Row>
          <Row label="image_drop_p">
            <FloatInput
              value={v.ipAdapter?.imageDropP}
              onChange={(n) =>
                set(["backend", "animaLora", "ipAdapter", "imageDropP"], n)
              }
              placeholder="0.05"
              step={0.01}
              min={0}
              max={1}
            />
          </Row>
        </Section>
      )}

      {/* === Turbo 蒸馏(独立路径) === */}
      <Section
        title="Turbo / DMD 蒸馏"
        subtitle="开启后会切换到 distill_turbo.py 路径(忽略 method/preset),输出 4-step LoRA"
      >
        <Row
          label="启用 turbo 蒸馏"
          description="勾选后下方字段才会写入 turbo 子配置;否则该字段为 null,保持普通训练。"
        >
          <ToggleSwitch
            checked={!!v.turbo}
            onCheckedChange={(c) =>
              set(
                ["backend", "animaLora", "turbo"],
                c
                  ? {
                      iterations: 1000,
                      studentRank: 48,
                      studentAlpha: 48,
                      studentSteps: 4,
                      teacherCfg: 4,
                    }
                  : undefined,
              )
            }
          />
        </Row>
        {v.turbo && (
          <>
            <Row label="iterations">
              <FloatInput
                value={v.turbo.iterations}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "iterations"], n)
                }
                placeholder="1000"
                min={1}
              />
            </Row>
            <Row label="student_rank / student_alpha" description="学生 LoRA 容量">
              <div className="flex gap-2 items-center">
                <FloatInput
                  value={v.turbo.studentRank}
                  onChange={(n) =>
                    set(["backend", "animaLora", "turbo", "studentRank"], n)
                  }
                  placeholder="48"
                  min={1}
                />
                <span className="text-muted-foreground">/</span>
                <FloatInput
                  value={v.turbo.studentAlpha}
                  onChange={(n) =>
                    set(["backend", "animaLora", "turbo", "studentAlpha"], n)
                  }
                  placeholder="48"
                  min={1}
                />
              </div>
            </Row>
            <Row label="student_steps" description="推理步数,蒸馏后用 --infer_steps N">
              <FloatInput
                value={v.turbo.studentSteps}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "studentSteps"], n)
                }
                placeholder="4"
                min={1}
              />
            </Row>
            <Row label="teacher_cfg" description="教师 CFG,会被烤进学生(推理时 --cfg 1.0)">
              <FloatInput
                value={v.turbo.teacherCfg}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "teacherCfg"], n)
                }
                placeholder="4"
                step={0.5}
              />
            </Row>
            <Row label="student_lr">
              <FloatInput
                value={v.turbo.studentLr}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "studentLr"], n)
                }
                placeholder="5e-6"
                step={1e-7}
              />
            </Row>
            <Row label="fake_lr">
              <FloatInput
                value={v.turbo.fakeLr}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "fakeLr"], n)
                }
                placeholder="5e-5"
                step={1e-6}
              />
            </Row>
            <Row label="save_every">
              <FloatInput
                value={v.turbo.saveEvery}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "saveEvery"], n)
                }
                placeholder="250"
                min={1}
              />
            </Row>
          </>
        )}
      </Section>
    </>
  )
})

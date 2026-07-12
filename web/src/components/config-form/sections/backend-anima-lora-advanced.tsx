import { Lock } from "lucide-react"
import type { ReactNode } from "react"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import {
  EnumSelect,
  FloatInput,
  IntInput,
  PathInput,
  Row,
  Section,
  ToggleSwitch,
} from "../widgets"

type AnimaLoraValue = NonNullable<
  NonNullable<ConfigFormValue["backend"]>["animaLora"]
>

export function AnimaLoraLockedDefaultsSection({
  value,
  set,
  errorMap,
  lockBadgeFor,
}: {
  value: AnimaLoraValue
  set: Setter
  errorMap: ErrorMap
  lockBadgeFor: (field: string) => ReactNode
}) {
  return (
    <Section
      icon={<Lock className="size-3.5" />}
      title="固定字段"
      subtitle="base.toml 字段与风险项"
    >
      <Row
        label="Masked Loss"
        labelBadge={lockBadgeFor("maskedLoss")}
        description="Anima masked loss。"
      >
        <ToggleSwitch
          checked={value.maskedLoss ?? true}
          onCheckedChange={(c) => set(["backend", "animaLora", "maskedLoss"], c)}
        />
      </Row>
      <Row
        label="torch.compile"
        labelBadge={lockBadgeFor("torchCompile")}
        description="启用 torch.compile。"
      >
        <ToggleSwitch
          checked={value.torchCompile ?? true}
          onCheckedChange={(c) => set(["backend", "animaLora", "torchCompile"], c)}
        />
      </Row>
      <Row
        label="跳过缓存校验"
        labelBadge={lockBadgeFor("skipCacheCheck")}
        description="跳过缓存哈希校验,只影响启动速度。"
      >
        <ToggleSwitch
          checked={value.skipCacheCheck ?? true}
          onCheckedChange={(c) =>
            set(["backend", "animaLora", "skipCacheCheck"], c)
          }
        />
      </Row>
      <Row
        label="DataLoader pin_memory"
        labelBadge={lockBadgeFor("dataloaderPinMemory")}
        description="DataLoader pin_memory。"
      >
        <ToggleSwitch
          checked={value.dataloaderPinMemory ?? true}
          onCheckedChange={(c) =>
            set(["backend", "animaLora", "dataloaderPinMemory"], c)
          }
        />
      </Row>
      <Row
        label="持久化 DataLoader workers"
        labelBadge={lockBadgeFor("persistentDataLoaderWorkers")}
        description="persistent_workers。"
      >
        <ToggleSwitch
          checked={value.persistentDataLoaderWorkers ?? false}
          onCheckedChange={(c) =>
            set(["backend", "animaLora", "persistentDataLoaderWorkers"], c)
          }
        />
      </Row>
      <Row
        label="裁剪交叉注意力 KV"
        labelBadge={lockBadgeFor("trimCrossattnKv")}
        description="启用 cross-attention KV trimming。"
      >
        <ToggleSwitch
          checked={value.trimCrossattnKv ?? false}
          onCheckedChange={(c) =>
            set(["backend", "animaLora", "trimCrossattnKv"], c)
          }
        />
      </Row>
      <Row
        label="半精度 VAE"
        labelBadge={lockBadgeFor("noHalfVae")}
        description="启用 half VAE。"
      >
        <ToggleSwitch
          checked={value.noHalfVae ?? false}
          onCheckedChange={(c) => set(["backend", "animaLora", "noHalfVae"], c)}
        />
      </Row>
      <Row
        label="保存精度"
        labelBadge={lockBadgeFor("savePrecision")}
        description="checkpoint 保存精度。"
      >
        <EnumSelect
          value={value.savePrecision ?? "bf16"}
          onChange={(s) => set(["backend", "animaLora", "savePrecision"], s)}
          options={[
            { value: "bf16", label: "bf16 · 默认" },
            { value: "fp16", label: "fp16" },
            { value: "fp32", label: "fp32" },
          ]}
        />
      </Row>
      <Row
        label="保存格式"
        labelBadge={lockBadgeFor("saveModelAs")}
        description="checkpoint 保存格式。"
      >
        <EnumSelect
          value={value.saveModelAs ?? "safetensors"}
          onChange={(s) => set(["backend", "animaLora", "saveModelAs"], s)}
          options={[{ value: "safetensors", label: "safetensors" }]}
        />
      </Row>
      <Row label="日志记录步数" description="每 N 步记录一次训练日志。">
        <FloatInput
          value={value.logEveryNSteps}
          onChange={(n) => set(["backend", "animaLora", "logEveryNSteps"], n)}
          placeholder="2"
          min={1}
        />
      </Row>

      <Row
        label="保留 token 数"
        labelBadge={lockBadgeFor("keepTokens")}
        description="caption shuffle 时保留前 N 个 token。"
      >
        <FloatInput
          value={value.keepTokens}
          onChange={(n) => set(["backend", "animaLora", "keepTokens"], n)}
          placeholder="3"
          min={0}
        />
      </Row>
      <Row
        label="caption 文件后缀"
        labelBadge={lockBadgeFor("captionExtension")}
        description="caption 文件后缀。"
      >
        <PathInput
          value={value.captionExtension ?? ""}
          onChange={(s) =>
            set(["backend", "animaLora", "captionExtension"], s || ".txt")
          }
          placeholder=".txt"
        />
      </Row>
      <Row
        label="验证集大小"
        labelBadge={lockBadgeFor("validationSplitNum")}
        description="验证集样本数。0 表示关闭。"
      >
        <FloatInput
          value={value.validationSplitNum}
          onChange={(n) => set(["backend", "animaLora", "validationSplitNum"], n)}
          placeholder="0"
          min={0}
        />
      </Row>
      <Row
        label="CMMD 验证"
        description="按验证集计算 val_loss / CMMD。"
        errors={errorMap.get("backend.animaLora.useCmmd")}
      >
        <ToggleSwitch
          checked={value.useCmmd ?? false}
          onCheckedChange={(c) => set(["backend", "animaLora", "useCmmd"], c)}
        />
      </Row>
      <Row label="验证随机种子" description="留空时使用训练随机种子。">
        <IntInput value={value.validationSeed ?? null} onChange={(n) => set(["backend", "animaLora", "validationSeed"], n)} placeholder="（训练种子）" />
      </Row>
      <Row label="验证采样步数">
        <IntInput min={1} value={value.validationSampleSteps ?? null} onChange={(n) => set(["backend", "animaLora", "validationSampleSteps"], n)} placeholder="（默认）" />
      </Row>
      <Row label="验证 CFG">
        <FloatInput min={0.1} step={0.1} value={value.validationCfgScale ?? null} onChange={(n) => set(["backend", "animaLora", "validationCfgScale"], n)} placeholder="（默认）" />
      </Row>
      <Row
        label="多分辨率分桶"
        labelBadge={lockBadgeFor("enableBucket")}
        description="启用多分辨率 bucketing。"
      >
        <ToggleSwitch
          checked={value.enableBucket ?? true}
          onCheckedChange={(c) => set(["backend", "animaLora", "enableBucket"], c)}
        />
      </Row>
      <Row
        label="路径匹配模式"
        description="fnmatch pattern。"
      >
        <PathInput
          value={value.pathPattern ?? ""}
          onChange={(s) => set(["backend", "animaLora", "pathPattern"], s || "*")}
          placeholder="*"
        />
      </Row>
    </Section>
  )
}

export function AnimaLoraTurboSection({
  value,
  set,
}: {
  value: AnimaLoraValue
  set: Setter
}) {
  return (
    <Section
      title="Turbo / DMD 蒸馏"
      subtitle="distill_turbo.py"
    >
      <Row
        label="启用 turbo 蒸馏"
        description="启用 turbo 子配置。"
      >
        <ToggleSwitch
          checked={!!value.turbo}
          onCheckedChange={(c) =>
            set(
              ["backend", "animaLora", "turbo"],
              c
                  ? {
                      iterations: 1000,
                      batchSize: 1,
                      seed: -1,
                      useCustomDownAutograd: true,
                      studentRank: 48,
                      studentAlpha: 48,
                      fakeRank: 64,
                      fakeAlpha: 64,
                      attnMode: "torch",
                      studentSteps: 4,
                      teacherCfg: 4,
                      tauCaMinGap: 0.05,
                      tauCaSkipAboveT: 0.95,
                      studentLr: 5e-6,
                      fakeLr: 5e-5,
                      fakeStepsPerStudentStep: 2,
                      alphaWarmupSteps: 100,
                      weightDecay: 0,
                      gradClip: 1,
                      tDistribution: "uniform",
                      sigmoidScale: 1,
                      saveEvery: 250,
                      logInterval: 5,
                    }
                : undefined,
            )
          }
        />
      </Row>
      {value.turbo && (
        <>
          <Row label="迭代次数">
            <IntInput
              value={value.turbo.iterations}
              onChange={(n) =>
                set(["backend", "animaLora", "turbo", "iterations"], n)
              }
              placeholder="1000"
              min={1}
            />
          </Row>
          <Row label="批大小">
            <IntInput min={1} value={value.turbo.batchSize ?? 1} onChange={(n) => set(["backend", "animaLora", "turbo", "batchSize"], n ?? 1)} />
          </Row>
          <Row label="随机种子" description="-1 表示启动时随机生成。">
            <IntInput value={value.turbo.seed ?? -1} onChange={(n) => set(["backend", "animaLora", "turbo", "seed"], n ?? -1)} />
          </Row>
          <Row label="自定义下投影 autograd">
            <ToggleSwitch checked={value.turbo.useCustomDownAutograd ?? true} onCheckedChange={(checked) => set(["backend", "animaLora", "turbo", "useCustomDownAutograd"], checked)} />
          </Row>
          <Row label="学生 rank / alpha" description="学生 LoRA 容量。">
            <div className="flex gap-2 items-center">
              <IntInput
                value={value.turbo.studentRank}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "studentRank"], n)
                }
                placeholder="48"
                min={1}
              />
              <span className="text-muted-foreground">/</span>
              <FloatInput
                value={value.turbo.studentAlpha}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "studentAlpha"], n)
                }
                placeholder="48"
                min={1}
              />
            </div>
          </Row>
          <Row label="Fake rank / alpha" description="辅助分布匹配 LoRA 容量。">
            <div className="flex gap-2 items-center">
              <IntInput min={1} value={value.turbo.fakeRank ?? 64} onChange={(n) => set(["backend", "animaLora", "turbo", "fakeRank"], n ?? 64)} />
              <span className="text-muted-foreground">/</span>
              <FloatInput min={0.0001} value={value.turbo.fakeAlpha ?? 64} onChange={(n) => set(["backend", "animaLora", "turbo", "fakeAlpha"], n ?? 64)} />
            </div>
          </Row>
          <Row label="注意力模式">
            <EnumSelect
              value={value.turbo.attnMode ?? "torch"}
              onChange={(next) => set(["backend", "animaLora", "turbo", "attnMode"], next)}
              options={[
                { value: "torch", label: "PyTorch SDPA · 默认" },
                { value: "flash", label: "Flash Attention" },
                { value: "flex", label: "Flex Attention" },
                { value: "sageattn", label: "SageAttention" },
                { value: "xformers", label: "xFormers" },
              ]}
            />
          </Row>
          <Row label="学生推理步数" description="infer_steps。">
            <IntInput
              value={value.turbo.studentSteps}
              onChange={(n) =>
                set(["backend", "animaLora", "turbo", "studentSteps"], n)
              }
              placeholder="4"
              min={1}
            />
          </Row>
          <Row label="教师 CFG" description="teacher cfg。">
            <FloatInput
              value={value.turbo.teacherCfg}
              onChange={(n) =>
                set(["backend", "animaLora", "turbo", "teacherCfg"], n)
              }
              placeholder="4"
              step={0.5}
            />
          </Row>
          <Row label="CA 最小时间步间隔">
            <FloatInput min={0} max={0.9999} step={0.01} value={value.turbo.tauCaMinGap ?? 0.05} onChange={(n) => set(["backend", "animaLora", "turbo", "tauCaMinGap"], n ?? 0.05)} />
          </Row>
          <Row label="CA 跳过阈值">
            <FloatInput min={0.0001} max={1} step={0.01} value={value.turbo.tauCaSkipAboveT ?? 0.95} onChange={(n) => set(["backend", "animaLora", "turbo", "tauCaSkipAboveT"], n ?? 0.95)} />
          </Row>
          <Row label="学生学习率">
            <FloatInput
              value={value.turbo.studentLr}
              onChange={(n) =>
                set(["backend", "animaLora", "turbo", "studentLr"], n)
              }
              placeholder="5e-6"
              step={1e-7}
            />
          </Row>
          <Row label="Fake 学习率">
            <FloatInput
              value={value.turbo.fakeLr}
              onChange={(n) =>
                set(["backend", "animaLora", "turbo", "fakeLr"], n)
              }
              placeholder="5e-5"
              step={1e-6}
            />
          </Row>
          <Row label="每次学生更新的 Fake 步数">
            <IntInput min={1} value={value.turbo.fakeStepsPerStudentStep ?? 2} onChange={(n) => set(["backend", "animaLora", "turbo", "fakeStepsPerStudentStep"], n ?? 2)} />
          </Row>
          <Row label="Alpha 预热步数">
            <IntInput min={0} value={value.turbo.alphaWarmupSteps ?? 100} onChange={(n) => set(["backend", "animaLora", "turbo", "alphaWarmupSteps"], n ?? 100)} />
          </Row>
          <Row label="权重衰减">
            <FloatInput min={0} step={0.001} value={value.turbo.weightDecay ?? 0} onChange={(n) => set(["backend", "animaLora", "turbo", "weightDecay"], n ?? 0)} />
          </Row>
          <Row label="梯度裁剪">
            <FloatInput min={0.0001} step={0.1} value={value.turbo.gradClip ?? 1} onChange={(n) => set(["backend", "animaLora", "turbo", "gradClip"], n ?? 1)} />
          </Row>
          <Row label="时间步分布">
            <EnumSelect value={value.turbo.tDistribution ?? "uniform"} onChange={(next) => set(["backend", "animaLora", "turbo", "tDistribution"], next)} options={[{ value: "uniform", label: "Uniform · 默认" }, { value: "sigmoid", label: "Sigmoid" }]} />
          </Row>
          {value.turbo.tDistribution === "sigmoid" && (
            <Row label="Sigmoid 缩放">
              <FloatInput min={0.0001} step={0.1} value={value.turbo.sigmoidScale ?? 1} onChange={(n) => set(["backend", "animaLora", "turbo", "sigmoidScale"], n ?? 1)} />
            </Row>
          )}
          <Row label="保存间隔" description="每 N 次迭代保存一次。">
            <IntInput
              value={value.turbo.saveEvery}
              onChange={(n) =>
                set(["backend", "animaLora", "turbo", "saveEvery"], n)
              }
              placeholder="250"
              min={1}
            />
          </Row>
          <Row label="日志间隔" description="每 N 次迭代记录一次指标。">
            <IntInput min={1} value={value.turbo.logInterval ?? 5} onChange={(n) => set(["backend", "animaLora", "turbo", "logInterval"], n ?? 5)} />
          </Row>
        </>
      )}
    </Section>
  )
}

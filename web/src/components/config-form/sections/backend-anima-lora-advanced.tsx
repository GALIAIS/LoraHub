import { Lock } from "lucide-react"
import type { ReactNode } from "react"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import {
  EnumSelect,
  FloatInput,
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
      title="上游默认 / 锁定字段"
      subtitle="anima_lora base.toml 写死的字段。带 🔒 是 upstream 无法 override 的;带 ⚠️ 可改但有副作用。"
    >
      <Row
        label="Masked Loss"
        labelBadge={lockBadgeFor("maskedLoss")}
        description="Anima 训练管线硬依赖,关掉是无效操作。"
      >
        <ToggleSwitch
          checked={value.maskedLoss ?? true}
          onCheckedChange={(c) => set(["backend", "animaLora", "maskedLoss"], c)}
        />
      </Row>
      <Row
        label="torch.compile"
        labelBadge={lockBadgeFor("torchCompile")}
        description="static_token_count 性能收益的前提,upstream 训练循环假定开启。"
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
        description="DataLoader pin_memory 一直开;upstream 没提供反向 flag。"
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
        description="减少 epoch 边界 stall,但长跑可能泄漏 file handle。"
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
        description="启用 KV trimming · 短 caption 加速约 10–15 %。"
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
        description="true 半精度 VAE 省显存,但偶尔在边缘数据集产生 NaN。"
      >
        <ToggleSwitch
          checked={value.noHalfVae ?? false}
          onCheckedChange={(c) => set(["backend", "animaLora", "noHalfVae"], c)}
        />
      </Row>
      <Row
        label="保存精度"
        labelBadge={lockBadgeFor("savePrecision")}
        description="bf16 是 upstream 默认且匹配训练 dtype。"
      >
        <EnumSelect
          value={value.savePrecision ?? "bf16"}
          onChange={(s) => set(["backend", "animaLora", "savePrecision"], s)}
          options={[
            { value: "bf16", label: "bf16 · 默认" },
            { value: "fp16", label: "fp16" },
            { value: "fp32", label: "fp32 · 2× 体积，无质量收益" },
          ]}
        />
      </Row>
      <Row
        label="保存格式"
        labelBadge={lockBadgeFor("saveModelAs")}
        description="Anima 只能加载 safetensors。"
      >
        <EnumSelect
          value={value.saveModelAs ?? "safetensors"}
          onChange={(s) => set(["backend", "animaLora", "saveModelAs"], s)}
          options={[{ value: "safetensors", label: "safetensors · 锁定" }]}
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
        description="caption shuffle 保前 N 个 tag。改 < 3 trigger word 不再可靠。"
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
        description="caption 文件后缀。改了所有图片会被跳过。"
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
        description="留出验证集大小；0 是 LoraHub 默认，关闭验证。大于 0 时需要开启 CMMD 验证。"
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
        description="开启后按验证集计算 val_loss/CMMD；关闭时 validationSplitNum 应为 0。"
        errors={errorMap.get("backend.animaLora.useCmmd")}
      >
        <ToggleSwitch
          checked={value.useCmmd ?? false}
          onCheckedChange={(c) => set(["backend", "animaLora", "useCmmd"], c)}
        />
      </Row>
      <Row
        label="多分辨率分桶"
        labelBadge={lockBadgeFor("enableBucket")}
        description="多分辨率 bucketing,Anima static-shape compile 硬约束。"
      >
        <ToggleSwitch
          checked={value.enableBucket ?? true}
          onCheckedChange={(c) => set(["backend", "animaLora", "enableBucket"], c)}
        />
      </Row>
      <Row
        label="路径匹配模式"
        description="fnmatch 模式;* 全部图,char_a/*|char_b/* OR-合并子文件夹。"
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
      subtitle="开启后会切换到 distill_turbo.py 路径(忽略 method/preset),输出 4-step LoRA"
    >
      <Row
        label="启用 turbo 蒸馏"
        description="勾选后下方字段才会写入 turbo 子配置;否则该字段为 null,保持普通训练。"
      >
        <ToggleSwitch
          checked={!!value.turbo}
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
      {value.turbo && (
        <>
          <Row label="迭代次数">
            <FloatInput
              value={value.turbo.iterations}
              onChange={(n) =>
                set(["backend", "animaLora", "turbo", "iterations"], n)
              }
              placeholder="1000"
              min={1}
            />
          </Row>
          <Row label="学生 rank / alpha" description="学生 LoRA 容量。">
            <div className="flex gap-2 items-center">
              <FloatInput
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
          <Row label="学生推理步数" description="蒸馏后用 --infer_steps N。">
            <FloatInput
              value={value.turbo.studentSteps}
              onChange={(n) =>
                set(["backend", "animaLora", "turbo", "studentSteps"], n)
              }
              placeholder="4"
              min={1}
            />
          </Row>
          <Row label="教师 CFG" description="教师 CFG,会被烤进学生（推理时 --cfg 1.0）。">
            <FloatInput
              value={value.turbo.teacherCfg}
              onChange={(n) =>
                set(["backend", "animaLora", "turbo", "teacherCfg"], n)
              }
              placeholder="4"
              step={0.5}
            />
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
          <Row label="保存间隔" description="每 N 次迭代保存一次。">
            <FloatInput
              value={value.turbo.saveEvery}
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
  )
}

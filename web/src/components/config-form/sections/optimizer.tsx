import { memo } from "react"
import { Label } from "@/components/ui/label"
import { LR_SCHEDULE_OPTIONS, OPTIMIZER_OPTIONS } from "../options"
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

export const OptimizerFields = memo(function OptimizerFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["optimizer"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const lr = v.lr ?? {}
  const [b1 = 0.9, b2 = 0.999] = (v.betas as number[] | undefined) ?? []
  return (
    <>
      <Row label="优化器">
        <EnumSelect
          value={v.type ?? "adamw8bit"}
          onChange={(t) => set(["optimizer", "type"], t)}
          options={OPTIMIZER_OPTIONS}
        />
      </Row>
      <Row label="U-Net 学习率" description="SDXL 角色 LoRA 推荐 1e-4。" errors={errorMap.get("optimizer.lr.unet")}>
        <FloatInput
          step={0.00001}
          value={lr.unet ?? 1e-4}
          onChange={(n) => set(["optimizer", "lr", "unet"], n ?? 1e-4)}
        />
      </Row>
      <Row label="文本编码器学习率" errors={errorMap.get("optimizer.lr.textEncoder")}>
        <FloatInput
          step={0.00001}
          value={lr.textEncoder ?? 5e-5}
          onChange={(n) => set(["optimizer", "lr", "textEncoder"], n ?? 5e-5)}
        />
      </Row>
      <Row label="学习率调度">
        <EnumSelect
          value={v.schedule ?? "cosine_with_restarts"}
          onChange={(s) => set(["optimizer", "schedule"], s)}
          options={LR_SCHEDULE_OPTIONS}
        />
      </Row>
      <Row label="预热步数">
        <IntInput
          min={0}
          value={v.warmupSteps ?? 100}
          onChange={(n) => set(["optimizer", "warmupSteps"], n ?? 0)}
        />
      </Row>

      <details className="rounded-[4px] border border-border/40 bg-muted/10 px-3 py-2 group">
        <summary className="cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.18em]">
          高级（betas / weight_decay / eps / optimizer_args）
        </summary>
        <div className="mt-3 space-y-3.5">
          <Row
            label="Betas"
            description="AdamW / Lion 风格的一阶/二阶动量。默认 (0.9, 0.999)。"
            errors={errorMap.get("optimizer.betas")}
          >
            <div className="flex items-center gap-2">
              <div>
                <Label className="text-[11px] text-muted-foreground">β₁</Label>
                <FloatInput
                  step={0.001}
                  value={b1}
                  onChange={(n) =>
                    set(["optimizer", "betas"], [n ?? 0.9, b2])
                  }
                />
              </div>
              <div>
                <Label className="text-[11px] text-muted-foreground">β₂</Label>
                <FloatInput
                  step={0.001}
                  value={b2}
                  onChange={(n) =>
                    set(["optimizer", "betas"], [b1, n ?? 0.999])
                  }
                />
              </div>
            </div>
          </Row>
          <Row
            label="Weight Decay"
            description="L2 权重衰减系数。AdamW 默认 0。"
            errors={errorMap.get("optimizer.weightDecay")}
          >
            <FloatInput
              step={0.001}
              value={v.weightDecay ?? 0}
              onChange={(n) => set(["optimizer", "weightDecay"], n ?? 0)}
            />
          </Row>
          <Row
            label="Eps"
            description="Adam ε（数值稳定项）。默认 1e-8。"
            errors={errorMap.get("optimizer.eps")}
          >
            <FloatInput
              step={1e-9}
              value={v.eps ?? 1e-8}
              onChange={(n) => set(["optimizer", "eps"], n ?? 1e-8)}
            />
          </Row>
          <Row
            label="optimizerArgs"
            description="额外 key=value，每行一对。覆盖上面三项；用于 Lion / Prodigy 等的私有参数。"
            errors={errorMap.get("optimizer.optimizerArgs")}
          >
            <KeyValueTextArea
              value={v.optimizerArgs}
              onChange={(next) => set(["optimizer", "optimizerArgs"], next)}
              placeholder={"momentum = 0.9\ndecouple = True"}
            />
          </Row>
          <Row
            label="maxGradNorm"
            description="梯度范数裁剪上限；0 关闭裁剪。"
            errors={errorMap.get("optimizer.maxGradNorm")}
          >
            <FloatInput
              step={0.1}
              value={v.maxGradNorm ?? 1.0}
              onChange={(n) => set(["optimizer", "maxGradNorm"], n ?? 1.0)}
            />
          </Row>
          <Row
            label="schedulerModule"
            description="自定义 LR scheduler module（kohya `--lr_scheduler_type`）。"
            errors={errorMap.get("optimizer.schedulerModule")}
          >
            <TextInput
              className="w-64"
              value={v.schedulerModule ?? ""}
              onChange={(s) =>
                set(["optimizer", "schedulerModule"], s || null)
              }
              placeholder="（可选）"
            />
          </Row>
          <Row
            label="schedulerArgs"
            description="scheduler 私有 kwargs（kohya `--lr_scheduler_args`）。"
            errors={errorMap.get("optimizer.schedulerArgs")}
          >
            <KeyValueTextArea
              value={v.schedulerArgs}
              onChange={(next) => set(["optimizer", "schedulerArgs"], next)}
              placeholder={"factor = 0.5\npatience = 5"}
            />
          </Row>
          <Row
            label="schedulerNumCycles"
            description="cosine_with_restarts 重启次数。"
            errors={errorMap.get("optimizer.schedulerNumCycles")}
          >
            <IntInput
              min={1}
              value={v.schedulerNumCycles ?? 1}
              onChange={(n) => set(["optimizer", "schedulerNumCycles"], n ?? 1)}
            />
          </Row>
          <Row
            label="schedulerPower"
            description="polynomial 衰减幂。"
            errors={errorMap.get("optimizer.schedulerPower")}
          >
            <FloatInput
              step={0.1}
              value={v.schedulerPower ?? 1.0}
              onChange={(n) => set(["optimizer", "schedulerPower"], n ?? 1.0)}
            />
          </Row>
          <Row
            label="schedulerTimescale"
            description="inverse_sqrt 时间常数；留空使用默认。"
            errors={errorMap.get("optimizer.schedulerTimescale")}
          >
            <IntInput
              min={1}
              value={v.schedulerTimescale ?? null}
              onChange={(n) => set(["optimizer", "schedulerTimescale"], n)}
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="schedulerMinLrRatio"
            description="cosine 最小 LR 比例（kohya `--lr_scheduler_min_lr_ratio`）。"
            errors={errorMap.get("optimizer.schedulerMinLrRatio")}
          >
            <FloatInput
              step={0.01}
              value={v.schedulerMinLrRatio ?? null}
              onChange={(n) => set(["optimizer", "schedulerMinLrRatio"], n)}
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="gradientRelease"
            description="dp 分块释放梯度以节省显存。"
          >
            <ToggleSwitch
              checked={v.gradientRelease ?? false}
              onCheckedChange={(b) => set(["optimizer", "gradientRelease"], b)}
            />
          </Row>
        </div>
      </details>
    </>
  )
})

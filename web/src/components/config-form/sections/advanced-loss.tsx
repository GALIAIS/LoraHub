/**
 * Advanced loss-shaping — fields beyond LossFields' core knobs.
 *
 * The core LossFields section covers min_snr_gamma / noise_offset /
 * ip_noise_gamma / loss_type / debiased / masked / v-pred. This advanced
 * panel surfaces every other LossConfig field added in the schema rewrite
 * so users can tune them without hand-writing YAML.
 */
import { memo } from "react"
import { HUBER_SCHEDULE_OPTIONS } from "../options"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import { EnumSelect, FloatInput, IntInput, Row, ToggleSwitch } from "../widgets"

export const AdvancedLossFields = memo(function AdvancedLossFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["loss"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const k = (n: string) => `loss.${n}` as const
  return (
    <>
      <Row
        label="noiseOffsetRandomStrength"
        description="对 noise_offset 启用随机强度。"
      >
        <ToggleSwitch
          checked={v.noiseOffsetRandomStrength ?? false}
          onCheckedChange={(b) =>
            set(["loss", "noiseOffsetRandomStrength"], b)
          }
        />
      </Row>
      <Row
        label="multiresNoiseIterations"
        description="多分辨率噪声迭代次数；留空关闭。"
        errors={errorMap.get(k("multiresNoiseIterations"))}
      >
        <IntInput
          min={1}
          value={v.multiresNoiseIterations ?? null}
          onChange={(n) => set(["loss", "multiresNoiseIterations"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="multiresNoiseDiscount"
        description="多分辨率噪声衰减系数 · 取值 0..1。"
        errors={errorMap.get(k("multiresNoiseDiscount"))}
      >
        <FloatInput
          step={0.05}
          value={v.multiresNoiseDiscount ?? 0.3}
          onChange={(n) =>
            set(["loss", "multiresNoiseDiscount"], n ?? 0.3)
          }
        />
      </Row>
      <Row
        label="adaptiveNoiseScale"
        description="自适应噪声尺度。"
        errors={errorMap.get(k("adaptiveNoiseScale"))}
      >
        <FloatInput
          step={0.01}
          value={v.adaptiveNoiseScale ?? null}
          onChange={(n) => set(["loss", "adaptiveNoiseScale"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="ipNoiseGammaRandomStrength"
        description="对 ip_noise_gamma 启用随机强度。"
      >
        <ToggleSwitch
          checked={v.ipNoiseGammaRandomStrength ?? false}
          onCheckedChange={(b) =>
            set(["loss", "ipNoiseGammaRandomStrength"], b)
          }
        />
      </Row>
      <Row label="zeroTerminalSnr" description="启用 zero terminal SNR 训练。">
        <ToggleSwitch
          checked={v.zeroTerminalSnr ?? false}
          onCheckedChange={(b) => set(["loss", "zeroTerminalSnr"], b)}
        />
      </Row>
      <Row
        label="minTimestep"
        description="时间步采样下限；留空使用默认。"
        errors={errorMap.get(k("minTimestep"))}
      >
        <IntInput
          min={0}
          value={v.minTimestep ?? null}
          onChange={(n) => set(["loss", "minTimestep"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="maxTimestep"
        description="时间步采样上限；留空使用默认。"
        errors={errorMap.get(k("maxTimestep"))}
      >
        <IntInput
          min={0}
          value={v.maxTimestep ?? null}
          onChange={(n) => set(["loss", "maxTimestep"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="huberSchedule" errors={errorMap.get(k("huberSchedule"))}>
        <EnumSelect
          value={v.huberSchedule ?? ""}
          onChange={(s) => set(["loss", "huberSchedule"], s || null)}
          options={HUBER_SCHEDULE_OPTIONS}
        />
      </Row>
      <Row label="huberC" errors={errorMap.get(k("huberC"))}>
        <FloatInput
          step={0.01}
          value={v.huberC ?? null}
          onChange={(n) => set(["loss", "huberC"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="huberScale" errors={errorMap.get(k("huberScale"))}>
        <FloatInput
          step={0.01}
          value={v.huberScale ?? null}
          onChange={(n) => set(["loss", "huberScale"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="vPredLikeLoss"
        description="把 ε 损失整形得像 v-prediction。"
        errors={errorMap.get(k("vPredLikeLoss"))}
      >
        <FloatInput
          step={0.01}
          value={v.vPredLikeLoss ?? null}
          onChange={(n) => set(["loss", "vPredLikeLoss"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="pseudoHuberC"
        description="dp 顶层 pseudo Huber 常数。"
        errors={errorMap.get(k("pseudoHuberC"))}
      >
        <FloatInput
          step={0.01}
          value={v.pseudoHuberC ?? null}
          onChange={(n) => set(["loss", "pseudoHuberC"], n)}
          placeholder="（默认）"
        />
      </Row>
    </>
  )
})

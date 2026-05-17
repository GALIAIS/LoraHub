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
        label="noise_offset_random_strength"
        description="对 noise_offset 启用随机强度。"
      >
        <ToggleSwitch
          checked={v.noise_offset_random_strength ?? false}
          onCheckedChange={(b) =>
            set(["loss", "noise_offset_random_strength"], b)
          }
        />
      </Row>
      <Row
        label="multires_noise_iterations"
        description="多分辨率噪声迭代次数；留空关闭。"
        errors={errorMap.get(k("multires_noise_iterations"))}
      >
        <IntInput
          min={1}
          value={v.multires_noise_iterations ?? null}
          onChange={(n) => set(["loss", "multires_noise_iterations"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="multires_noise_discount"
        description="多分辨率噪声衰减系数（0..1）。"
        errors={errorMap.get(k("multires_noise_discount"))}
      >
        <FloatInput
          step={0.05}
          value={v.multires_noise_discount ?? 0.3}
          onChange={(n) =>
            set(["loss", "multires_noise_discount"], n ?? 0.3)
          }
        />
      </Row>
      <Row
        label="adaptive_noise_scale"
        description="自适应噪声尺度。"
        errors={errorMap.get(k("adaptive_noise_scale"))}
      >
        <FloatInput
          step={0.01}
          value={v.adaptive_noise_scale ?? null}
          onChange={(n) => set(["loss", "adaptive_noise_scale"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="ip_noise_gamma_random_strength"
        description="对 ip_noise_gamma 启用随机强度。"
      >
        <ToggleSwitch
          checked={v.ip_noise_gamma_random_strength ?? false}
          onCheckedChange={(b) =>
            set(["loss", "ip_noise_gamma_random_strength"], b)
          }
        />
      </Row>
      <Row label="zero_terminal_snr" description="启用 zero terminal SNR 训练。">
        <ToggleSwitch
          checked={v.zero_terminal_snr ?? false}
          onCheckedChange={(b) => set(["loss", "zero_terminal_snr"], b)}
        />
      </Row>
      <Row
        label="min_timestep"
        description="时间步采样下限；留空使用默认。"
        errors={errorMap.get(k("min_timestep"))}
      >
        <IntInput
          min={0}
          value={v.min_timestep ?? null}
          onChange={(n) => set(["loss", "min_timestep"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="max_timestep"
        description="时间步采样上限；留空使用默认。"
        errors={errorMap.get(k("max_timestep"))}
      >
        <IntInput
          min={0}
          value={v.max_timestep ?? null}
          onChange={(n) => set(["loss", "max_timestep"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="huber_schedule" errors={errorMap.get(k("huber_schedule"))}>
        <EnumSelect
          value={v.huber_schedule ?? ""}
          onChange={(s) => set(["loss", "huber_schedule"], s || null)}
          options={HUBER_SCHEDULE_OPTIONS}
        />
      </Row>
      <Row label="huber_c" errors={errorMap.get(k("huber_c"))}>
        <FloatInput
          step={0.01}
          value={v.huber_c ?? null}
          onChange={(n) => set(["loss", "huber_c"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row label="huber_scale" errors={errorMap.get(k("huber_scale"))}>
        <FloatInput
          step={0.01}
          value={v.huber_scale ?? null}
          onChange={(n) => set(["loss", "huber_scale"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="v_pred_like_loss"
        description="把 ε 损失整形得像 v-prediction。"
        errors={errorMap.get(k("v_pred_like_loss"))}
      >
        <FloatInput
          step={0.01}
          value={v.v_pred_like_loss ?? null}
          onChange={(n) => set(["loss", "v_pred_like_loss"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="pseudo_huber_c"
        description="dp 顶层 pseudo Huber 常数。"
        errors={errorMap.get(k("pseudo_huber_c"))}
      >
        <FloatInput
          step={0.01}
          value={v.pseudo_huber_c ?? null}
          onChange={(n) => set(["loss", "pseudo_huber_c"], n)}
          placeholder="（默认）"
        />
      </Row>
    </>
  )
})

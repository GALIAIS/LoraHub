import { memo } from "react"
import { LOSS_TYPE_OPTIONS } from "../options"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { EnumSelect, FloatInput, Row, ToggleSwitch } from "../widgets"

/**
 * Loss-shaping hyperparameters mapping to LossConfig in schema.py.
 *
 * Two of the floats (`min_snr_gamma`, `ip_noise_gamma`) are nullable —
 * a "未设置" toggle gates them so users explicitly opt in instead of
 * sending stale stale defaults. The remaining numeric fields default to
 * 0 / 1.0 to match sd-scripts.
 */
export const LossFields = memo(function LossFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["loss"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const minSnrEnabled = v.min_snr_gamma !== null && v.min_snr_gamma !== undefined
  const ipNoiseEnabled = v.ip_noise_gamma !== null && v.ip_noise_gamma !== undefined

  return (
    <>
      <Row
        label="Min-SNR Gamma"
        description="启用 Min-SNR 加权（推荐 5.0）。仅在勾选时写入；否则走 sd-scripts 默认。"
        errors={errorMap.get("loss.min_snr_gamma")}
      >
        <div className="flex items-center gap-3">
          <ToggleSwitch
            checked={minSnrEnabled}
            onCheckedChange={(b) =>
              set(["loss", "min_snr_gamma"], b ? 5.0 : null)
            }
          />
          {minSnrEnabled && (
            <FloatInput
              step={0.1}
              value={v.min_snr_gamma ?? 5.0}
              onChange={(n) => set(["loss", "min_snr_gamma"], n)}
            />
          )}
        </div>
      </Row>
      <Row
        label="Noise Offset"
        description="给输入噪声加常数偏移以增强对比度（kohya 推荐 0.05~0.1）。"
        errors={errorMap.get("loss.noise_offset")}
      >
        <FloatInput
          step={0.01}
          value={v.noise_offset ?? 0}
          onChange={(n) => set(["loss", "noise_offset"], n ?? 0)}
        />
      </Row>
      <Row
        label="IP Noise Gamma"
        description="可选的 input perturbation noise 强度。"
        errors={errorMap.get("loss.ip_noise_gamma")}
      >
        <div className="flex items-center gap-3">
          <ToggleSwitch
            checked={ipNoiseEnabled}
            onCheckedChange={(b) =>
              set(["loss", "ip_noise_gamma"], b ? 0.1 : null)
            }
          />
          {ipNoiseEnabled && (
            <FloatInput
              step={0.01}
              value={v.ip_noise_gamma ?? 0.1}
              onChange={(n) => set(["loss", "ip_noise_gamma"], n)}
            />
          )}
        </div>
      </Row>
      <Row
        label="Prior Loss Weight"
        description="DreamBooth 风格正则项权重。仅在使用 prior preservation 时调整。"
        errors={errorMap.get("loss.prior_loss_weight")}
      >
        <FloatInput
          step={0.1}
          value={v.prior_loss_weight ?? 1.0}
          onChange={(n) => set(["loss", "prior_loss_weight"], n ?? 1.0)}
        />
      </Row>
      <Row label="损失类型" errors={errorMap.get("loss.loss_type")}>
        <EnumSelect
          value={v.loss_type ?? "l2"}
          onChange={(t) => set(["loss", "loss_type"], t)}
          options={LOSS_TYPE_OPTIONS}
        />
      </Row>
      <Row
        label="Debiased Estimation"
        description="启用 debiased estimation loss（kohya `--debiased_estimation_loss`）。"
      >
        <ToggleSwitch
          checked={v.debiased_estimation ?? false}
          onCheckedChange={(b) => set(["loss", "debiased_estimation"], b)}
        />
      </Row>
      <Row
        label="Masked Loss"
        description="按 alpha 通道遮罩计算损失（仅在数据集自带遮罩时启用）。"
      >
        <ToggleSwitch
          checked={v.masked_loss ?? false}
          onCheckedChange={(b) => set(["loss", "masked_loss"], b)}
        />
      </Row>
      <Row
        label="V-Pred 损失缩放"
        description="`--scale_v_pred_loss_like_noise_pred`，与 v_parameterization 搭配使用。"
      >
        <ToggleSwitch
          checked={v.scale_v_pred_loss_like_noise_pred ?? false}
          onCheckedChange={(b) =>
            set(["loss", "scale_v_pred_loss_like_noise_pred"], b)
          }
        />
      </Row>
      <Row
        label="V-Parameterization"
        description="启用 v-prediction 训练目标（SD2.x 768、部分 SDXL 微调需要）。"
      >
        <ToggleSwitch
          checked={v.v_parameterization ?? false}
          onCheckedChange={(b) => set(["loss", "v_parameterization"], b)}
        />
      </Row>
    </>
  )
})

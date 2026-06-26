import { memo } from "react"
import { LOSS_TYPE_OPTIONS } from "../options"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
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
  value: ConfigFormValue["loss"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const minSnrEnabled = v.minSnrGamma !== null && v.minSnrGamma !== undefined
  const ipNoiseEnabled = v.ipNoiseGamma !== null && v.ipNoiseGamma !== undefined

  return (
    <>
      <Row
        label="Min-SNR Gamma"
        description="启用 Min-SNR 加权。仅在勾选时写入。"
        errors={errorMap.get("loss.minSnrGamma")}
      >
        <div className="flex items-center gap-3">
          <ToggleSwitch
            checked={minSnrEnabled}
            onCheckedChange={(b) =>
              set(["loss", "minSnrGamma"], b ? 5.0 : null)
            }
          />
          {minSnrEnabled && (
            <FloatInput
              step={0.1}
              value={v.minSnrGamma ?? 5.0}
              onChange={(n) => set(["loss", "minSnrGamma"], n)}
            />
          )}
        </div>
      </Row>
      <Row
        label="Noise Offset"
        description="给输入噪声加常数偏移。"
        errors={errorMap.get("loss.noiseOffset")}
      >
        <FloatInput
          step={0.01}
          value={v.noiseOffset ?? 0}
          onChange={(n) => set(["loss", "noiseOffset"], n ?? 0)}
        />
      </Row>
      <Row
        label="IP Noise Gamma"
        description="可选的 input perturbation noise 强度。"
        errors={errorMap.get("loss.ipNoiseGamma")}
      >
        <div className="flex items-center gap-3">
          <ToggleSwitch
            checked={ipNoiseEnabled}
            onCheckedChange={(b) =>
              set(["loss", "ipNoiseGamma"], b ? 0.1 : null)
            }
          />
          {ipNoiseEnabled && (
            <FloatInput
              step={0.01}
              value={v.ipNoiseGamma ?? 0.1}
              onChange={(n) => set(["loss", "ipNoiseGamma"], n)}
            />
          )}
        </div>
      </Row>
      <Row
        label="Prior Loss Weight"
        description="DreamBooth 风格正则项权重。仅在使用 prior preservation 时调整。"
        errors={errorMap.get("loss.priorLossWeight")}
      >
        <FloatInput
          step={0.1}
          value={v.priorLossWeight ?? 1.0}
          onChange={(n) => set(["loss", "priorLossWeight"], n ?? 1.0)}
        />
      </Row>
      <Row label="损失类型" errors={errorMap.get("loss.lossType")}>
        <EnumSelect
          value={v.lossType ?? "l2"}
          onChange={(t) => set(["loss", "lossType"], t)}
          options={LOSS_TYPE_OPTIONS}
        />
      </Row>
      <Row
        label="Debiased Estimation"
        description="启用 debiased estimation loss（kohya `--debiased_estimation_loss`）。"
      >
        <ToggleSwitch
          checked={v.debiasedEstimation ?? false}
          onCheckedChange={(b) => set(["loss", "debiasedEstimation"], b)}
        />
      </Row>
      <Row
        label="Masked Loss"
        description="按 alpha 通道遮罩计算损失（仅在数据集自带遮罩时启用）。"
      >
        <ToggleSwitch
          checked={v.maskedLoss ?? false}
          onCheckedChange={(b) => set(["loss", "maskedLoss"], b)}
        />
      </Row>
      <Row
        label="V-Pred 损失缩放"
        description="`--scale_v_pred_loss_like_noise_pred`，与 v_parameterization 搭配使用。"
      >
        <ToggleSwitch
          checked={v.scaleVPredLossLikeNoisePred ?? false}
          onCheckedChange={(b) =>
            set(["loss", "scaleVPredLossLikeNoisePred"], b)
          }
        />
      </Row>
      <Row
        label="V-Parameterization"
        description="启用 v-prediction 训练目标（SD2.x 768、部分 SDXL 微调需要）。"
      >
        <ToggleSwitch
          checked={v.vParameterization ?? false}
          onCheckedChange={(b) => set(["loss", "vParameterization"], b)}
        />
      </Row>
    </>
  )
})

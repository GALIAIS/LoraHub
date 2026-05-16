import { memo } from "react"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { FloatInput, IntInput, Row } from "../widgets"

/**
 * Validation cadence + held-out split editor.
 *
 * `dataset.val_split` lives on DatasetConfig but is conceptually paired with
 * the ValidationConfig knobs, so the editor surfaces both together. Setting
 * val_split=0 disables validation entirely; the compiler skips emitting
 * validation argv in that case.
 */
export const ValidationFields = memo(function ValidationFields({
  value,
  set,
  errorMap,
}: {
  value: RecipeFormValue
  set: Setter
  errorMap: ErrorMap
}) {
  const valSplit = value.dataset?.val_split ?? 0
  const v = value.validation ?? {}
  return (
    <>
      <Row
        label="留出比例"
        description="0 关闭验证；上限 0.5。例如 0.1 表示 10% 数据留作验证集。"
        errors={errorMap.get("dataset.val_split")}
      >
        <FloatInput
          step={0.01}
          value={valSplit}
          onChange={(n) =>
            set(["dataset", "val_split"], Math.max(0, Math.min(0.49, n ?? 0)))
          }
        />
      </Row>
      <Row
        label="每 N 回合验证一次"
        description="仅在留出比例 > 0 时生效。"
        errors={errorMap.get("validation.every_n_epochs")}
      >
        <IntInput
          min={1}
          value={v.every_n_epochs ?? 1}
          onChange={(n) => set(["validation", "every_n_epochs"], n ?? 1)}
        />
      </Row>
      <Row
        label="最大验证样本数"
        description="可选。验证集很大时用来限制每次评估的步数；留空则全跑。"
        errors={errorMap.get("validation.max_samples")}
      >
        <IntInput
          min={1}
          value={v.max_samples ?? null}
          onChange={(n) => set(["validation", "max_samples"], n)}
          placeholder="（不限）"
        />
      </Row>
    </>
  )
})

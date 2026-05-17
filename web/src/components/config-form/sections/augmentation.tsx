/**
 * AugmentationConfig editor — image augmentation (kohya only).
 *
 * dp doesn't currently consume any of these fields; the dp compiler
 * ignores them silently.
 */
import { memo } from "react"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import { Row, TextInput, ToggleSwitch } from "../widgets"

export const AugmentationFields = memo(function AugmentationFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["augmentation"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row label="水平翻转" description="随机水平翻转图像（人物角色慎用）。">
        <ToggleSwitch
          checked={v.flip ?? false}
          onCheckedChange={(b) => set(["augmentation", "flip"], b)}
        />
      </Row>
      <Row label="颜色扰动" description="启用颜色抖动增强。">
        <ToggleSwitch
          checked={v.color ?? false}
          onCheckedChange={(b) => set(["augmentation", "color"], b)}
        />
      </Row>
      <Row label="随机裁剪" description="启用随机裁剪。">
        <ToggleSwitch
          checked={v.random_crop ?? false}
          onCheckedChange={(b) => set(["augmentation", "random_crop"], b)}
        />
      </Row>
      <Row
        label="face_crop_aug_range"
        description="kohya 三元组 `min_face_size,target_size,max_face_size`。"
        errors={errorMap.get("augmentation.face_crop_aug_range")}
      >
        <TextInput
          className="w-64"
          value={v.face_crop_aug_range ?? ""}
          onChange={(s) =>
            set(["augmentation", "face_crop_aug_range"], s || null)
          }
          placeholder="例如 256,512,1024"
        />
      </Row>
      <Row
        label="alpha_mask"
        description="使用图像 alpha 通道作为 masked-loss 掩码。"
      >
        <ToggleSwitch
          checked={v.alpha_mask ?? false}
          onCheckedChange={(b) => set(["augmentation", "alpha_mask"], b)}
        />
      </Row>
    </>
  )
})

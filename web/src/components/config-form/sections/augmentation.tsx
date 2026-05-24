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
      <Row label="水平翻转" description="随机水平翻转图像 · 角色训练慎用。">
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
      <Row label="随机裁剪" description="启用随机裁剪增强。">
        <ToggleSwitch
          checked={v.randomCrop ?? false}
          onCheckedChange={(b) => set(["augmentation", "randomCrop"], b)}
        />
      </Row>
      <Row
        label="faceCropAugRange"
        description="kohya 三元组 `min_face_size,target_size,max_face_size`。"
        errors={errorMap.get("augmentation.faceCropAugRange")}
      >
        <TextInput
          className="w-64"
          value={v.faceCropAugRange ?? ""}
          onChange={(s) =>
            set(["augmentation", "faceCropAugRange"], s || null)
          }
          placeholder="例如 256,512,1024"
        />
      </Row>
      <Row
        label="alphaMask"
        description="使用图像 alpha 通道作为 masked-loss 掩码。"
      >
        <ToggleSwitch
          checked={v.alphaMask ?? false}
          onCheckedChange={(b) => set(["augmentation", "alphaMask"], b)}
        />
      </Row>
    </>
  )
})

import { memo } from "react"
import { NETWORK_DTYPE_OPTIONS, NETWORK_TYPE_OPTIONS } from "../options"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { EnumSelect, FloatInput, IntInput, PathInput, Row, ToggleSwitch } from "../widgets"

export const NetworkFields = memo(function NetworkFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["network"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const type = v.type ?? "lora"
  // Pure LoRA / DoRA in sd-scripts don't expose conv layers; the schema
  // rejects conv_dim / conv_alpha on those types, so hide the fields.
  const supportsConv = type === "locon" || type === "loha"
  const swnEnabled =
    v.scale_weight_norms !== null && v.scale_weight_norms !== undefined
  return (
    <>
      <Row label="网络类型">
        <EnumSelect
          value={type}
          onChange={(t) => set(["network", "type"], t)}
          options={NETWORK_TYPE_OPTIONS}
        />
      </Row>
      <Row
        label="Rank（秩）"
        description="越高容量越大，显存占用也越大。SDXL 角色推荐 32。"
        errors={errorMap.get("network.rank")}
      >
        <IntInput
          min={1}
          max={512}
          value={v.rank ?? 32}
          onChange={(n) => set(["network", "rank"], n ?? 32)}
        />
      </Row>
      <Row
        label="Alpha（缩放）"
        description="实际学习率缩放因子。常见做法是 alpha = rank / 2。"
        errors={errorMap.get("network.alpha")}
      >
        <IntInput
          min={1}
          value={v.alpha ?? 16}
          onChange={(n) => set(["network", "alpha"], n ?? 16)}
        />
      </Row>
      <Row label="训练 U-Net" description="训练 U-Net（视觉变化所必需）。">
        <ToggleSwitch
          checked={v.target_unet ?? true}
          onCheckedChange={(b) => set(["network", "target_unet"], b)}
        />
      </Row>
      <Row
        label="训练文本编码器"
        description="一并训练文本编码器，速度更慢；有助于风格 / 概念的泛化。"
      >
        <ToggleSwitch
          checked={v.target_text_encoder ?? false}
          onCheckedChange={(b) => set(["network", "target_text_encoder"], b)}
        />
      </Row>

      <details className="rounded-[4px] border border-border/40 bg-muted/10 px-3 py-2 group">
        <summary className="cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.18em]">
          高级（dropout / conv / weight-norm）
        </summary>
        <div className="mt-3 space-y-3.5">
          {supportsConv && (
            <>
              <Row
                label="Conv Rank"
                description="LyCORIS conv 层秩。仅对 locon / loha 生效。"
                errors={errorMap.get("network.conv_dim")}
              >
                <IntInput
                  min={1}
                  max={512}
                  value={v.conv_dim ?? null}
                  onChange={(n) => set(["network", "conv_dim"], n)}
                  placeholder="（默认）"
                />
              </Row>
              <Row
                label="Conv Alpha"
                description="LyCORIS conv 层 alpha。留空则与 alpha 相同。"
                errors={errorMap.get("network.conv_alpha")}
              >
                <IntInput
                  min={1}
                  value={v.conv_alpha ?? null}
                  onChange={(n) => set(["network", "conv_alpha"], n)}
                  placeholder="（与 alpha 相同）"
                />
              </Row>
            </>
          )}
          <Row
            label="Network Dropout"
            description="整网 dropout 概率（0..1）。"
            errors={errorMap.get("network.network_dropout")}
          >
            <FloatInput
              step={0.05}
              value={v.network_dropout ?? 0}
              onChange={(n) => set(["network", "network_dropout"], n ?? 0)}
            />
          </Row>
          <Row
            label="Rank Dropout"
            description="低秩矩阵中按秩维度的 dropout（0..1）。"
            errors={errorMap.get("network.rank_dropout")}
          >
            <FloatInput
              step={0.05}
              value={v.rank_dropout ?? 0}
              onChange={(n) => set(["network", "rank_dropout"], n ?? 0)}
            />
          </Row>
          <Row
            label="Module Dropout"
            description="按 LoRA 模块整体 dropout（0..1）。"
            errors={errorMap.get("network.module_dropout")}
          >
            <FloatInput
              step={0.05}
              value={v.module_dropout ?? 0}
              onChange={(n) => set(["network", "module_dropout"], n ?? 0)}
            />
          </Row>
          <Row
            label="Scale Weight Norms"
            description="`--scale_weight_norms` 最大范数；可选启用以约束权重大小。"
            errors={errorMap.get("network.scale_weight_norms")}
          >
            <div className="flex items-center gap-3">
              <ToggleSwitch
                checked={swnEnabled}
                onCheckedChange={(b) =>
                  set(["network", "scale_weight_norms"], b ? 1.0 : null)
                }
              />
              {swnEnabled && (
                <FloatInput
                  step={0.1}
                  value={v.scale_weight_norms ?? 1.0}
                  onChange={(n) => set(["network", "scale_weight_norms"], n)}
                />
              )}
            </div>
          </Row>
          <Row
            label="init_from"
            description="基于已有 LoRA 续训（kohya `--network_weights`，dp `init_from_existing`）。"
            errors={errorMap.get("network.init_from")}
          >
            <PathInput
              value={v.init_from ?? ""}
              onChange={(s) => set(["network", "init_from"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          <Row
            label="dim_from_weights"
            description="kohya 从已加载权重读取 rank。"
            errors={errorMap.get("network.dim_from_weights")}
          >
            <PathInput
              value={v.dim_from_weights ?? ""}
              onChange={(s) => set(["network", "dim_from_weights"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          <Row label="dtype" description="dp 训练时 LoRA 参数 dtype（kohya 忽略此字段）。">
            <EnumSelect
              value={v.dtype ?? ""}
              onChange={(s) => set(["network", "dtype"], s || null)}
              options={NETWORK_DTYPE_OPTIONS}
            />
          </Row>
          <Row
            label="base_weights"
            description="训练前合并的 LoRA 路径列表，每行一个（kohya）。"
            errors={errorMap.get("network.base_weights")}
          >
            <textarea
              value={(v.base_weights ?? []).join("\n")}
              onChange={(e) => {
                const lines = e.target.value
                  .split("\n")
                  .map((s) => s.trim())
                  .filter((s) => s.length > 0)
                set(["network", "base_weights"], lines)
              }}
              rows={3}
              className="font-mono w-full max-w-2xl rounded-[4px] border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none placeholder:text-muted-foreground/60 focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
              placeholder="/path/to/style.safetensors"
            />
          </Row>
          <Row
            label="base_weights_multiplier"
            description="对应 base_weights 的合并强度，每行一个浮点数。长度必须与上一项一致。"
            errors={errorMap.get("network.base_weights_multiplier")}
          >
            <textarea
              value={(v.base_weights_multiplier ?? []).join("\n")}
              onChange={(e) => {
                const nums = e.target.value
                  .split("\n")
                  .map((s) => s.trim())
                  .filter((s) => s.length > 0)
                  .map((s) => parseFloat(s))
                  .filter((n) => !Number.isNaN(n))
                set(["network", "base_weights_multiplier"], nums)
              }}
              rows={3}
              className="font-mono w-full max-w-2xl rounded-[4px] border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none placeholder:text-muted-foreground/60 focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
              placeholder="0.5"
            />
          </Row>
          <Row
            label="fuse_adapters"
            description="dp 训练前融合的 LoRA 列表（JSON 数组，每项 {path, multiplier}）。"
            errors={errorMap.get("network.fuse_adapters")}
          >
            <textarea
              value={JSON.stringify(v.fuse_adapters ?? [], null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value || "[]")
                  if (Array.isArray(parsed)) {
                    set(["network", "fuse_adapters"], parsed)
                  }
                } catch {
                  // 当用户编辑中途 JSON 不合法时静默忽略，让其继续输入。
                }
              }}
              rows={4}
              className="font-mono w-full max-w-2xl rounded-[4px] border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none placeholder:text-muted-foreground/60 focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
              placeholder={'[{"path": "/path/to.safetensors", "multiplier": 1.0}]'}
            />
          </Row>
          <Row
            label="module_lr"
            description="Anima/Wan 多组件模型的 per-submodule LR。留空走全局 unet LR。"
          >
            <PerModuleLREditor value={v.module_lr ?? null} set={set} errorMap={errorMap} />
          </Row>
        </div>
      </details>
    </>
  )
})

const PerModuleLREditor = memo(function PerModuleLREditor({
  value,
  set,
  errorMap,
}: {
  value: NonNullable<ConfigFormValue["network"]>["module_lr"]
  set: Setter
  errorMap: ErrorMap
}) {
  const enabled = value !== null && value !== undefined
  const v = value ?? {}
  return (
    <div className="space-y-2">
      <ToggleSwitch
        checked={enabled}
        onCheckedChange={(b) =>
          set(["network", "module_lr"], b ? {} : null)
        }
      />
      {enabled && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
          {(["llm_adapter", "self_attn", "cross_attn", "mlp", "mod"] as const).map(
            (key) => (
              <div key={key}>
                <div className="text-[11px] text-muted-foreground">{key}</div>
                <FloatInput
                  step={1e-5}
                  value={v[key] ?? null}
                  onChange={(n) => set(["network", "module_lr", key], n)}
                  placeholder="（继承）"
                />
                {errorMap.get(`network.module_lr.${key}`)?.map((m, i) => (
                  <div key={i} className="text-[10px] text-destructive">
                    {m}
                  </div>
                ))}
              </div>
            ),
          )}
        </div>
      )}
    </div>
  )
})

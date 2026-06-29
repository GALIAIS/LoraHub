import { memo } from "react"
import { NETWORK_DTYPE_OPTIONS, NETWORK_TYPE_OPTIONS } from "../options"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { EnumSelect, FloatInput, IntInput, PathInput, Row, ToggleSwitch } from "../widgets"

export const NetworkFields = memo(function NetworkFields({
  value = {},
  set,
  errorMap,
  backendType,
}: {
  value: ConfigFormValue["network"]
  set: Setter
  errorMap: ErrorMap
  backendType?: "kohya" | "diffusion-pipe" | "anima_lora" | "ai_toolkit"
}) {
  const v = value ?? {}
  const type = v.type ?? "lora"
  const isDiffusionPipe = backendType === "diffusion-pipe"
  const isAiToolkit = backendType === "ai_toolkit"
  // Pure LoRA / DoRA in sd-scripts don't expose conv layers; the schema
  // rejects conv_dim / conv_alpha on those types, so hide the fields.
  const supportsConv = !isDiffusionPipe && !isAiToolkit && (type === "locon" || type === "loha")
  const swnEnabled =
    v.scaleWeightNorms !== null && v.scaleWeightNorms !== undefined
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
        description="控制 LoRA 容量；数值越高显存占用越大。"
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
      {!isDiffusionPipe && (
        <>
          <Row label="训练 U-Net" description="训练视觉网络。">
            <ToggleSwitch
              checked={v.targetUnet ?? true}
              onCheckedChange={(b) => set(["network", "targetUnet"], b)}
            />
          </Row>
          <Row
            label="训练文本编码器"
            description="同时训练文本编码器。"
          >
            <ToggleSwitch
              checked={v.targetTextEncoder ?? false}
              onCheckedChange={(b) => set(["network", "targetTextEncoder"], b)}
            />
          </Row>
        </>
      )}

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
                errors={errorMap.get("network.convDim")}
              >
                <IntInput
                  min={1}
                  max={512}
                  value={v.convDim ?? null}
                  onChange={(n) => set(["network", "convDim"], n)}
                  placeholder="（默认）"
                />
              </Row>
              <Row
                label="Conv Alpha"
                description="LyCORIS conv 层 alpha。留空则与 alpha 相同。"
                errors={errorMap.get("network.convAlpha")}
              >
                <IntInput
                  min={1}
                  value={v.convAlpha ?? null}
                  onChange={(n) => set(["network", "convAlpha"], n)}
                  placeholder="（与 alpha 相同）"
                />
              </Row>
            </>
          )}
          {!isDiffusionPipe && !isAiToolkit && (
            <>
              <Row
                label="Network Dropout"
                description="整网 dropout 概率（0..1）。"
                errors={errorMap.get("network.networkDropout")}
              >
                <FloatInput
                  step={0.05}
                  value={v.networkDropout ?? 0}
                  onChange={(n) => set(["network", "networkDropout"], n ?? 0)}
                />
              </Row>
              <Row
                label="Rank Dropout"
                description="低秩矩阵中按秩维度的 dropout（0..1）。"
                errors={errorMap.get("network.rankDropout")}
              >
                <FloatInput
                  step={0.05}
                  value={v.rankDropout ?? 0}
                  onChange={(n) => set(["network", "rankDropout"], n ?? 0)}
                />
              </Row>
              <Row
                label="Module Dropout"
                description="按 LoRA 模块整体 dropout（0..1）。"
                errors={errorMap.get("network.moduleDropout")}
              >
                <FloatInput
                  step={0.05}
                  value={v.moduleDropout ?? 0}
                  onChange={(n) => set(["network", "moduleDropout"], n ?? 0)}
                />
              </Row>
              <Row
                label="Scale Weight Norms"
                description="最大范数。"
                errors={errorMap.get("network.scaleWeightNorms")}
              >
                <div className="flex items-center gap-3">
                  <ToggleSwitch
                    checked={swnEnabled}
                    onCheckedChange={(b) =>
                      set(["network", "scaleWeightNorms"], b ? 1.0 : null)
                    }
                  />
                  {swnEnabled && (
                    <FloatInput
                      step={0.1}
                      value={v.scaleWeightNorms ?? 1.0}
                      onChange={(n) => set(["network", "scaleWeightNorms"], n)}
                    />
                  )}
                </div>
              </Row>
            </>
          )}
          <Row
            label="initFrom"
            description="基于已有 LoRA 续训（kohya `--network_weights`，dp `init_from_existing`）。"
            errors={errorMap.get("network.initFrom")}
          >
            <PathInput
              value={v.initFrom ?? ""}
              onChange={(s) => set(["network", "initFrom"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          <Row
            label="dimFromWeights"
            description="kohya 从已加载权重读取 rank。"
            errors={errorMap.get("network.dimFromWeights")}
          >
            <PathInput
              value={v.dimFromWeights ?? ""}
              onChange={(s) => set(["network", "dimFromWeights"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          {isDiffusionPipe && (
            <Row label="dtype" description="LoRA 参数 dtype。">
              <EnumSelect
                value={v.dtype ?? ""}
                onChange={(s) => set(["network", "dtype"], s || null)}
                options={NETWORK_DTYPE_OPTIONS}
              />
            </Row>
          )}
          {!isDiffusionPipe && !isAiToolkit && (
            <>
              <Row
                label="baseWeights"
                description="训练前合并的 LoRA 路径列表，每行一个。"
                errors={errorMap.get("network.baseWeights")}
              >
                <textarea
                  value={(v.baseWeights ?? []).join("\n")}
                  onChange={(e) => {
                    const lines = e.target.value
                      .split("\n")
                      .map((s) => s.trim())
                      .filter((s) => s.length > 0)
                    set(["network", "baseWeights"], lines)
                  }}
                  rows={3}
                  className="font-mono w-full max-w-2xl rounded-[4px] border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none placeholder:text-muted-foreground/60 focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
                  placeholder="/path/to/style.safetensors"
                />
              </Row>
              <Row
                label="baseWeightsMultiplier"
                description="对应 baseWeights 的合并强度，每行一个浮点数。"
                errors={errorMap.get("network.baseWeightsMultiplier")}
              >
                <textarea
                  value={(v.baseWeightsMultiplier ?? []).join("\n")}
                  onChange={(e) => {
                    const nums = e.target.value
                      .split("\n")
                      .map((s) => s.trim())
                      .filter((s) => s.length > 0)
                      .map((s) => parseFloat(s))
                      .filter((n) => !Number.isNaN(n))
                    set(["network", "baseWeightsMultiplier"], nums)
                  }}
                  rows={3}
                  className="font-mono w-full max-w-2xl rounded-[4px] border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none placeholder:text-muted-foreground/60 focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
                  placeholder="0.5"
                />
              </Row>
            </>
          )}
          {isDiffusionPipe && (
            <Row
              label="fuseAdapters"
              description="训练前融合的 LoRA 列表（JSON 数组，每项 {path, multiplier}）。"
              errors={errorMap.get("network.fuseAdapters")}
            >
              <textarea
                value={JSON.stringify(v.fuseAdapters ?? [], null, 2)}
                onChange={(e) => {
                  try {
                    const parsed = JSON.parse(e.target.value || "[]")
                    if (Array.isArray(parsed)) {
                      set(["network", "fuseAdapters"], parsed)
                    }
                  } catch {
                    // 用户编辑 JSON 中途可能不合法，保持当前值。
                  }
                }}
                rows={4}
                className="font-mono w-full max-w-2xl rounded-[4px] border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none placeholder:text-muted-foreground/60 focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
                placeholder={'[{"path": "/path/to.safetensors", "multiplier": 1.0}]'}
              />
            </Row>
          )}
          {!isDiffusionPipe && !isAiToolkit && (
            <Row
              label="moduleLr"
              description="多组件模型的 per-submodule LR。留空走全局 U-Net LR。"
            >
              <PerModuleLREditor value={v.moduleLr ?? null} set={set} errorMap={errorMap} />
            </Row>
          )}
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
  value: NonNullable<ConfigFormValue["network"]>["moduleLr"]
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
          set(["network", "moduleLr"], b ? {} : null)
        }
      />
      {enabled && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
          {(["llmAdapter", "selfAttn", "crossAttn", "mlp", "mod"] as const).map(
            (key) => (
              <div key={key}>
                <div className="text-[11px] text-muted-foreground">{key}</div>
                <FloatInput
                  step={1e-5}
                  value={v[key] ?? null}
                  onChange={(n) => set(["network", "moduleLr", key], n)}
                  placeholder="（继承）"
                />
                {errorMap.get(`network.moduleLr.${key}`)?.map((m, i) => (
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

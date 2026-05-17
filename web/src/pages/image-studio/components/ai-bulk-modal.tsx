import { useState } from "react"
import { X, Sparkles } from "lucide-react"
import type { AiBulkTab } from "./types"

interface AiBulkModalProps {
  paths: string[]
  datasetPath: string
  onClose: () => void
  onStart: (tab: AiBulkTab, params: Record<string, unknown>) => void
}

const tabs: { id: AiBulkTab; label: string }[] = [
  { id: "smart-caption", label: "智能标注" },
  { id: "vlm-caption", label: "VLM 标注" },
  { id: "quality-score", label: "质量评分" },
  { id: "wd14", label: "WD14 标注" },
  { id: "trigger-words", label: "触发词建议" },
]

export function AiBulkModal({ paths, datasetPath, onClose, onStart }: AiBulkModalProps) {
  const [activeTab, setActiveTab] = useState<AiBulkTab>("smart-caption")
  const [device, setDevice] = useState("auto")
  const [mergeStrategy, setMergeStrategy] = useState("append")
  const [taggerModel, setTaggerModel] = useState("wd-swinv2-v3")
  const [generalThreshold, setGeneralThreshold] = useState(0.35)
  const [characterThreshold, setCharacterThreshold] = useState(0.85)
  const [overwrite, setOverwrite] = useState(false)

  const handleStart = () => {
    const base = { device, paths }
    switch (activeTab) {
      case "smart-caption":
        onStart(activeTab, { ...base, mergeStrategy, path: datasetPath })
        break
      case "vlm-caption":
        onStart(activeTab, { ...base, path: datasetPath })
        break
      case "quality-score":
        onStart(activeTab, { ...base, path: datasetPath })
        break
      case "wd14":
        onStart(activeTab, {
          ...base,
          path: datasetPath,
          model_id: taggerModel,
          general: generalThreshold,
          character: characterThreshold,
          overwrite,
        })
        break
      case "trigger-words":
        onStart(activeTab, { ...base, path: datasetPath })
        break
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="w-[32rem] max-h-[80vh] flex flex-col rounded-lg border bg-popover shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-2">
            <Sparkles className="size-4 text-primary" />
            <h3 className="text-sm font-semibold">AI 批量操作</h3>
            <span className="text-xs text-muted-foreground">
              {paths.length > 0 ? `${paths.length} 张已选` : "全部图片"}
            </span>
          </div>
          <button type="button" onClick={onClose} className="rounded p-1 hover:bg-muted">
            <X className="size-4" />
          </button>
        </div>

        <div className="flex border-b">
          {tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setActiveTab(t.id)}
              className={`px-3 py-2 text-xs transition-colors ${
                activeTab === t.id
                  ? "border-b-2 border-primary font-medium"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {/* Common device selector */}
          <label className="flex items-center gap-2 mb-3">
            <span className="text-xs text-muted-foreground w-16">设备</span>
            <select
              value={device}
              onChange={(e) => setDevice(e.target.value)}
              className="rounded border bg-background px-2 py-1 text-xs flex-1"
            >
              <option value="auto">自动</option>
              <option value="cuda">CUDA (GPU)</option>
              <option value="cpu">CPU</option>
            </select>
          </label>

          {activeTab === "smart-caption" && (
            <div className="flex flex-col gap-3">
              <p className="text-xs text-muted-foreground">
                使用 WD14 标签 + VLM 视觉模型生成综合描述
              </p>
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">合并策略</span>
                <select
                  value={mergeStrategy}
                  onChange={(e) => setMergeStrategy(e.target.value)}
                  className="rounded border bg-background px-2 py-1 text-xs flex-1"
                >
                  <option value="append">追加</option>
                  <option value="replace">替换</option>
                  <option value="prepend">前置</option>
                </select>
              </label>
            </div>
          )}

          {activeTab === "vlm-caption" && (
            <p className="text-xs text-muted-foreground">
              仅使用视觉语言模型生成自然语言描述
            </p>
          )}

          {activeTab === "quality-score" && (
            <p className="text-xs text-muted-foreground">
              使用 AI 模型对图片质量进行评分 (优/中/差)
            </p>
          )}

          {activeTab === "wd14" && (
            <div className="flex flex-col gap-3">
              <p className="text-xs text-muted-foreground">
                运行 WD14/JoyTag 标注器生成标签
              </p>
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">模型</span>
                <select
                  value={taggerModel}
                  onChange={(e) => setTaggerModel(e.target.value)}
                  className="rounded border bg-background px-2 py-1 text-xs flex-1"
                >
                  <option value="wd-swinv2-v3">WD SwinV2 v3</option>
                  <option value="wd-vit-v3">WD ViT v3</option>
                  <option value="wd-convnext-v3">WD ConvNext v3</option>
                  <option value="joytag">JoyTag</option>
                </select>
              </label>
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">通用阈值</span>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={generalThreshold}
                  onChange={(e) => setGeneralThreshold(Number(e.target.value))}
                  className="rounded border bg-background px-2 py-1 text-xs w-20"
                />
              </label>
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">角色阈值</span>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={characterThreshold}
                  onChange={(e) => setCharacterThreshold(Number(e.target.value))}
                  className="rounded border bg-background px-2 py-1 text-xs w-20"
                />
              </label>
              <label className="flex items-center gap-1.5 text-xs">
                <input
                  type="checkbox"
                  checked={overwrite}
                  onChange={(e) => setOverwrite(e.target.checked)}
                  className="size-3"
                />
                覆盖已有标签
              </label>
            </div>
          )}

          {activeTab === "trigger-words" && (
            <p className="text-xs text-muted-foreground">
              分析数据集图片，建议合适的触发词
            </p>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t px-4 py-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-3 py-1.5 text-xs hover:bg-muted"
          >
            取消
          </button>
          <button
            type="button"
            onClick={handleStart}
            className="rounded-md bg-primary px-4 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
          >
            开始执行
          </button>
        </div>
      </div>
    </div>
  )
}
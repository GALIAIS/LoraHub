import { useEffect, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { X, Sparkles } from "lucide-react"
import { api } from "@/lib/api"
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

// Server-side fallback when ``/api/tagging/wd14/models`` hasn't
// resolved yet — same id the backend defaults to. Without this the
// select would render with an empty value attribute on the very
// first render and the form would post that to /api/tagging/tag,
// which is exactly the 401 we just got bitten by.
const FALLBACK_DEFAULT_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"

export function AiBulkModal({ paths, datasetPath, onClose, onStart }: AiBulkModalProps) {
  const [activeTab, setActiveTab] = useState<AiBulkTab>("smart-caption")
  const [device, setDevice] = useState("auto")
  const [mergeStrategy, setMergeStrategy] = useState("replace")
  const [taggerModel, setTaggerModel] = useState<string>(FALLBACK_DEFAULT_MODEL)
  const [generalThreshold, setGeneralThreshold] = useState(0.35)
  const [characterThreshold, setCharacterThreshold] = useState(0.85)
  const [overwrite, setOverwrite] = useState(false)
  const [captionMode, setCaptionMode] = useState<"general" | "style" | "character">("style")
  // "vlm" — multimodal model sees the image (best quality, more
  // expensive, requires a vision-capable model + quota).
  // "tags" — text-only LLM composes from the WD14 tag list. Useful
  // when the configured VLM is rate-limited / quota-exhausted.
  const [captionSource, setCaptionSource] = useState<"vlm" | "tags">("vlm")
  const [triggerWord, setTriggerWord] = useState("")
  const [stripStyleTags, setStripStyleTags] = useState(true)
  // Shared "skip already-processed" toggle for the three tabs that
  // don't have an explicit overwrite/skip control of their own
  // (smart-caption, vlm-caption, quality-score, trigger-words). WD14
  // already exposes this via its own ``overwrite`` checkbox below.
  // Default ON so the common case ("don't redo work") just works.
  const [skipDone, setSkipDone] = useState(true)

  // Pull the curated SmilingWolf catalogue from the server. The
  // dropdown was previously hard-coded to short names (e.g.
  // ``wd-eva02-large-v3``) that the HF resolver couldn't find,
  // tripping a 401 mid-tag-request. The single source of truth is
  // ``lorahub/core/tagging/wd14.py:WD14_MODEL_CATALOG``.
  const wd14Models = useQuery({
    queryKey: ["wd14-models"],
    queryFn: api.listWd14Models,
    staleTime: 60 * 60 * 1000,  // catalogue is effectively static
  })

  // Once the canonical default arrives, swap in if the user hasn't
  // touched the picker yet — the fallback short-circuits a flicker
  // from "loading" to "wrong default" between mount and resolution.
  useEffect(() => {
    if (wd14Models.data?.default && taggerModel === FALLBACK_DEFAULT_MODEL) {
      setTaggerModel(wd14Models.data.default)
    }
  }, [wd14Models.data?.default, taggerModel])

  const wd14Options = wd14Models.data?.models ?? [
    { id: FALLBACK_DEFAULT_MODEL, label: "v3 · EvaCLIP-Large(推荐)" },
  ]

  const handleStart = () => {
    const base = { device, paths }
    switch (activeTab) {
      case "smart-caption":
        onStart(activeTab, {
          ...base,
          mergeStrategy,
          path: datasetPath,
          captionMode,
          captionSource,
          triggerWord: triggerWord.trim() || undefined,
          stripStyleTags,
          skipExisting: skipDone,
        })
        break
      case "vlm-caption":
        onStart(activeTab, {
          ...base,
          path: datasetPath,
          skipAnnotated: skipDone,
        })
        break
      case "quality-score":
        onStart(activeTab, {
          ...base,
          path: datasetPath,
          skipScored: skipDone,
        })
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
        onStart(activeTab, {
          ...base,
          path: datasetPath,
          skipAnalyzed: skipDone,
        })
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

          {/* Shared "skip done" toggle. WD14 has its own overwrite control
              right inside the wd14 panel below, so we only show this for
              the four non-WD14 tabs. */}
          {activeTab !== "wd14" && (
            <label className="flex items-center gap-1.5 text-xs mb-3">
              <input
                type="checkbox"
                checked={skipDone}
                onChange={(e) => setSkipDone(e.target.checked)}
                className="size-3"
              />
              <span>
                跳过已{
                  activeTab === "quality-score" ? "评分" :
                  activeTab === "trigger-words" ? "分析" :
                  "标注"
                }的图片
              </span>
              <span className="text-muted-foreground/70">
                ({
                  activeTab === "quality-score" ? "已有 AI 质量评分" :
                  activeTab === "trigger-words" ? "已生成触发词建议" :
                  "已有非空 .txt"
                })
              </span>
            </label>
          )}

          {activeTab === "smart-caption" && (
            <div className="flex flex-col gap-3">
              <p className="text-xs text-muted-foreground">
                WD14 标签 + LLM 综合标注，按训练用途自动调整 prompt
              </p>
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">LLM 输入</span>
                <select
                  value={captionSource}
                  onChange={(e) => setCaptionSource(e.target.value as typeof captionSource)}
                  className="rounded border bg-background px-2 py-1 text-xs flex-1"
                >
                  <option value="vlm">视觉模型（看图）· 质量最高</option>
                  <option value="tags">仅 WD14 标签 · 不上传图片，省额度/兼容文本模型</option>
                </select>
              </label>
              <p className="text-[11px] text-muted-foreground/80 -mt-1.5 pl-[4.5rem]">
                {captionSource === "tags"
                  ? "LLM 不会看到图片，仅根据 WD14 给出的 tag 列表撰写描述。提示词已针对此场景优化，避免凭空虚构。"
                  : "多模态模型直接看图，质量最佳。若服务商额度耗尽或当前模型不支持视觉，可切换为「仅标签」。"}
              </p>
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">训练用途</span>
                <select
                  value={captionMode}
                  onChange={(e) => setCaptionMode(e.target.value as typeof captionMode)}
                  className="rounded border bg-background px-2 py-1 text-xs flex-1"
                >
                  <option value="style">风格 LoRA（不写画风词）</option>
                  <option value="character">角色 LoRA（不写角色特征）</option>
                  <option value="general">通用（描述全部内容）</option>
                </select>
              </label>
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">触发词</span>
                <input
                  type="text"
                  value={triggerWord}
                  onChange={(e) => setTriggerWord(e.target.value)}
                  placeholder="例如 anima style"
                  className="rounded border bg-background px-2 py-1 text-xs flex-1"
                />
              </label>
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">合并策略</span>
                <select
                  value={mergeStrategy}
                  onChange={(e) => setMergeStrategy(e.target.value)}
                  className="rounded border bg-background px-2 py-1 text-xs flex-1"
                >
                  <option value="replace">替换（推荐）</option>
                  <option value="append">追加</option>
                  <option value="prepend">前置</option>
                </select>
              </label>
              {captionMode === "style" && (
                <label className="flex items-center gap-1.5 text-xs">
                  <input
                    type="checkbox"
                    checked={stripStyleTags}
                    onChange={(e) => setStripStyleTags(e.target.checked)}
                    className="size-3"
                  />
                  自动剔除画风/质量类 WD14 标签（anime / illustration / masterpiece 等）
                </label>
              )}
            </div>
          )}

          {activeTab === "vlm-caption" && (
            <p className="text-xs text-muted-foreground">
              仅使用视觉语言模型生成自然语言描述
            </p>
          )}

          {activeTab === "quality-score" && (
            <p className="text-xs text-muted-foreground">
              使用 AI 模型对图片质量进行评分（优 / 中 / 差）
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
                  {wd14Options.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
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
            <div className="flex flex-col gap-2">
              <p className="text-xs text-muted-foreground">
                逐图分析视觉内容，给出 1-3 个适合作为 LoRA 触发词的短语。
                批次完成后会在工具栏下方汇总数据集层面的高频候选词，可点击复制。
              </p>
              <p className="text-xs text-muted-foreground/70">
                只写入 store（不修改 .txt），可在右侧检查器看到每张图的建议触发词。
              </p>
            </div>
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
import { useEffect, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { X, Sparkles, BookOpen } from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import type { AiBulkTab } from "./types"
import { TriggerPicker } from "./library/trigger-picker"
import {
  CaptionPromptPicker,
  type CaptionPromptValue,
} from "./caption-prompt-picker"

interface AiBulkModalProps {
  paths: string[]
  datasetPath: string
  onClose: () => void
  onStart: (tab: AiBulkTab, params: Record<string, unknown>) => void
}

const tabs: { id: AiBulkTab; label: string }[] = [
  { id: "smart-caption", label: "标注" },
  { id: "quality-score", label: "质量评分" },
  { id: "trigger-words", label: "触发词候选" },
]
type AnnotationMode = "tag" | "nl" | "tag-llm" | "tag-vlm" | "toriigate"

// Server-side fallback when ``/api/tagging/wd14/models`` hasn't
// resolved yet — same id the backend defaults to. Without this the
// select would render with an empty value attribute on the very
// first render and the form would post that to /api/tagging/tag,
// which is exactly the 401 we just got bitten by.
const FALLBACK_DEFAULT_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"

export function AiBulkModal({ paths, datasetPath, onClose, onStart }: AiBulkModalProps) {
  const [activeTab, setActiveTab] = useState<AiBulkTab>("smart-caption")
  const [annotationMode, setAnnotationMode] = useState<AnnotationMode>("tag-vlm")
  const [device, setDevice] = useState("auto")
  const [mergeStrategy, setMergeStrategy] = useState("replace")
  const [taggerModel, setTaggerModel] = useState<string>(FALLBACK_DEFAULT_MODEL)
  const [generalThreshold, setGeneralThreshold] = useState(0.35)
  const [characterThreshold, setCharacterThreshold] = useState(0.85)
  const [overwrite, setOverwrite] = useState(false)
  const [captionMode, setCaptionMode] = useState<"general" | "style" | "character">("style")
  const [captionPrompt, setCaptionPrompt] =
    useState<CaptionPromptValue>("style")
  const [promptTemplate, setPromptTemplate] = useState<string | undefined>()
  // "vlm" — multimodal model sees the image (best quality, more
  // expensive, requires a vision-capable model + quota).
  // "tags" — text-only LLM composes from the WD14 tag list. Useful
  // when the configured VLM is rate-limited / quota-exhausted.
  const [captionSource] = useState<"vlm" | "tags">("vlm")
  const [triggerWord, setTriggerWord] = useState("")
  const [triggerPickerOpen, setTriggerPickerOpen] = useState(false)
  const [stripStyleTags, setStripStyleTags] = useState(true)
  // Disabling WD14 reduces the caption to "trigger word + LLM nl_text"
  // (or just the trigger in style mode where the LLM is also off).
  // Useful for users who want a clean trigger-only training set or
  // who do their own tagging upstream.
  const [useWd14, setUseWd14] = useState(true)
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
    { id: FALLBACK_DEFAULT_MODEL, label: "v3 · EvaCLIP-Large" },
  ]

  const handleStart = () => {
    const base = { device, paths }
    switch (activeTab) {
      case "smart-caption":
        if (annotationMode === "tag") {
          onStart("wd14", {
            ...base,
            path: datasetPath,
            model_id: taggerModel,
            general: generalThreshold,
            character: characterThreshold,
            overwrite,
          })
          break
        }
        onStart(activeTab, {
          ...base,
          mergeStrategy,
          path: datasetPath,
          captionMode,
          promptTemplate,
          captionSource:
            annotationMode === "tag-llm"
              ? "tags"
              : annotationMode === "toriigate"
                ? "toriigate"
                : annotationMode === "nl"
                  ? "vlm"
                  : captionSource,
          triggerWord: triggerWord.trim() || undefined,
          stripStyleTags,
          useWd14: annotationMode !== "nl" && useWd14,
          skipExisting: skipDone,
        })
        break
      case "quality-score":
        onStart(activeTab, {
          ...base,
          path: datasetPath,
          skipScored: skipDone,
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

          {activeTab !== "smart-caption" || annotationMode !== "tag" ? (
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
                  activeTab === "trigger-words" ? "已生成触发词候选" :
                  "已有非空 .txt"
                })
              </span>
            </label>
          ) : null}

          {activeTab === "smart-caption" && (
            <div className="flex flex-col gap-3">
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">模式</span>
                <select
                  value={annotationMode}
                  onChange={(e) => setAnnotationMode(e.target.value as AnnotationMode)}
                  className="rounded border bg-background px-2 py-1 text-xs flex-1"
                >
                  <option value="tag">TAG 模式</option>
                  <option value="nl">NL 模式</option>
                  <option value="tag-llm">TAG+LLM 模式</option>
                  <option value="tag-vlm">TAG+VLM 模式</option>
                  <option value="toriigate">ToriiGate 模式</option>
                </select>
              </label>
              {annotationMode === "tag" ? (
                <>
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
                </>
              ) : (
                <>
                  <p className="text-xs text-muted-foreground">
                    {annotationMode === "nl"
                      ? "视觉模型直接生成自然语言 caption，不调用 WD14。"
                      : annotationMode === "tag-llm"
                        ? "先生成 TAG，再由 LLM 按训练用途重写 caption。"
                        : annotationMode === "toriigate"
                          ? "使用 ToriiGate 官方格式生成 caption。"
                          : "先生成 TAG，再由 VLM 看图按训练用途重写 caption。"}
                  </p>
                  {annotationMode !== "nl" && (
                    <label className="flex items-center gap-1.5 text-xs">
                <input
                  type="checkbox"
                  checked={useWd14}
                  onChange={(e) => setUseWd14(e.target.checked)}
                  className="size-3"
                />
                <span>{annotationMode === "toriigate" ? "使用 WD14 参考标签" : "启用 WD14 标签预处理"}</span>
                <span className="text-muted-foreground/70">
                  （关闭后仅用 LLM 描述 + 触发词，不调用 WD14 模型）
                </span>
              </label>
                  )}
              <p className="text-[11px] text-muted-foreground/80 -mt-1.5 pl-[4.5rem]">
                {annotationMode === "nl"
                  ? "自然语言模式适合快速生成描述，不输出 booru tag。"
                  : !useWd14 && captionMode === "style"
                  ? "WD14 关闭 + 风格模式：caption 仅含触发词，不调用 LLM。"
                  : !useWd14
                    ? "WD14 已关闭：LLM 直接看图描述并自行写 caption（无标签参考）。"
                  : captionMode === "style"
                      ? "风格模式：LLM 看图（或 WD14 标签）后输出『描述 + 修正后的标签』，过滤掉画风词与矛盾标签。caption 末尾不再硬拼接原始 WD14。"
                      : captionMode === "character"
                        ? "角色模式：LLM 输出『描述 + 修正后的标签』，过滤掉外貌身份词（hair / eyes / skin / body）与矛盾标签。"
                        : annotationMode === "tag-llm"
                          ? "LLM 不会看到图片，仅根据 WD14 标签列表撰写。提示词已针对此场景优化，避免凭空虚构。"
                          : annotationMode === "toriigate"
                            ? "使用 ToriiGate 官方 short caption 格式，适合 ToriiGate-0.5 服务商。"
                          : "多模态模型直接读取图片。若服务商额度耗尽或当前模型不支持视觉，可切换为「仅标签」。"}
              </p>
              {annotationMode !== "nl" && (
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">训练用途</span>
                <CaptionPromptPicker
                  value={captionPrompt}
                  onChange={(next) => {
                    setCaptionPrompt(next.value)
                    setCaptionMode(next.captionMode)
                    setPromptTemplate(next.promptTemplate)
                  }}
                  className="flex-1"
                />
              </label>
              )}
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">触发词</span>
                <div className="relative flex-1 flex items-center gap-1">
                  <input
                    type="text"
                    value={triggerWord}
                    onChange={(e) => setTriggerWord(e.target.value)}
                    placeholder="例如 anima style"
                    className="rounded border bg-background px-2 py-1 text-xs flex-1"
                  />
                  <button
                    type="button"
                    onClick={() => setTriggerPickerOpen((v) => !v)}
                    className="flex items-center gap-1 rounded border px-2 py-1 text-[11px] hover:bg-accent"
                    title="从工具库选触发词"
                  >
                    <BookOpen className="size-3" />
                    工具库
                  </button>
                  {triggerPickerOpen && (
                    <TriggerPicker
                      onSelect={(t) => setTriggerWord(t)}
                      onClose={() => setTriggerPickerOpen(false)}
                    />
                  )}
                </div>
              </div>
              <label className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16">合并策略</span>
                <select
                  value={mergeStrategy}
                  onChange={(e) => setMergeStrategy(e.target.value)}
                  className="rounded border bg-background px-2 py-1 text-xs flex-1"
                >
                  <option value="replace">替换</option>
                  <option value="append">追加</option>
                  <option value="prepend">前置</option>
                </select>
              </label>
              {annotationMode !== "nl" && captionMode === "style" && useWd14 && (
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
                </>
              )}
            </div>
          )}

          {activeTab === "quality-score" && (
            <p className="text-xs text-muted-foreground">
              使用 AI 模型对图片质量进行评分（优 / 中 / 差）
            </p>
          )}

          {activeTab === "trigger-words" && (
            <div className="flex flex-col gap-2">
              <p className="text-xs text-muted-foreground">
                逐图分析视觉内容，生成 1-3 个触发词候选短语。
                批次完成后会在工具栏下方汇总数据集层面的高频候选词，可点击复制。
              </p>
              <p className="text-xs text-muted-foreground/70">
                只写入 store（不修改 .txt），可在右侧检查器查看每张图的触发词候选。
              </p>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t px-4 py-3">
          <Button
            type="button"
            onClick={onClose}
            variant="ghost"
            size="sm"
          >
            取消
          </Button>
          <Button
            type="button"
            onClick={handleStart}
            size="sm"
          >
            开始执行
          </Button>
        </div>
      </div>
    </div>
  )
}

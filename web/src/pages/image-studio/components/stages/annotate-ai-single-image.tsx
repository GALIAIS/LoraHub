import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ImageIcon, Loader2, Play, Sparkles, Tags } from "lucide-react"
import { toast } from "sonner"

import {
  api,
  imageStudioVlmAnimaRewrite,
  imageStudioWd14Prefilter,
  type Wd14PrefilterResult,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { FALLBACK_DEFAULT_MODEL, Field, Row } from "./annotate-ai-shared"

// --------------------------------------------------------------------------- //
// ai-wd14-prefilter — 单图测试入口（调试用）
// --------------------------------------------------------------------------- //

export function AiWd14PrefilterTool({ datasetPath }: { datasetPath: string }) {
  const [imagePath, setImagePath] = useState("")
  const [taggerModel, setTaggerModel] = useState<string>(FALLBACK_DEFAULT_MODEL)
  const [device, setDevice] = useState("auto")
  const [generalThreshold, setGeneralThreshold] = useState(0.35)
  const [characterThreshold, setCharacterThreshold] = useState(0.85)
  const [captionMode, setCaptionMode] =
    useState<"general" | "style" | "character">("style")
  const [captionSource, setCaptionSource] = useState<"vlm" | "tags">("vlm")
  const [triggerWord, setTriggerWord] = useState("")
  const [stripStyleTags, setStripStyleTags] = useState(true)
  const [result, setResult] = useState<Wd14PrefilterResult | null>(null)

  // 与 ai-bulk-modal 同源的真实模型列表
  const wd14Models = useQuery({
    queryKey: ["wd14-models"],
    queryFn: api.listWd14Models,
    staleTime: 60 * 60 * 1000,
  })
  useEffect(() => {
    if (wd14Models.data?.default && taggerModel === FALLBACK_DEFAULT_MODEL) {
      setTaggerModel(wd14Models.data.default)
    }
  }, [wd14Models.data?.default, taggerModel])
  const modelOptions = wd14Models.data?.models ?? [
    { id: FALLBACK_DEFAULT_MODEL, label: "v3 · EvaCLIP-Large" },
  ]

  const mutation = useMutation({
    mutationFn: () =>
      imageStudioWd14Prefilter({
        path: imagePath.trim(),
        taggerModel,
        device,
        generalThreshold,
        characterThreshold,
        captionMode,
        captionSource,
        triggerWord: triggerWord.trim() || undefined,
        stripStyleTags,
      }),
    onSuccess: (data) => setResult(data),
    onError: (err) =>
      toast.error("prefilter 失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  return (
    <div className="h-full overflow-y-auto p-4 max-w-3xl grid gap-3 lg:grid-cols-2">
      <section className="rounded-md border border-border/60 bg-card">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Tags className="size-3.5" />
          <span className="text-xs font-medium">WD14 单步出标签 · 单图测试</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            智能 caption 第一步独立出来：给一张图，拿到 WD14 标签 +
            assemble 好的 prompt。结果可直接给「VLM Anima 重写」工具用。
          </p>
          <Row label="图片路径">
            <Input
              value={imagePath}
              onChange={(e) => setImagePath(e.target.value)}
              placeholder={`${datasetPath}/sample.png`}
              className="h-8 text-xs font-mono"
            />
          </Row>
          <Row label="模型">
            <Select
              value={taggerModel}
              onValueChange={(v) => v && setTaggerModel(v)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {modelOptions.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Row>
          <Row label="设备">
            <Select value={device} onValueChange={(v) => v && setDevice(v)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">自动</SelectItem>
                <SelectItem value="cuda">CUDA</SelectItem>
                <SelectItem value="cpu">CPU</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <div className="grid grid-cols-2 gap-2">
            <Row label="通用阈值">
              <Input
                type="number"
                step="0.05"
                min={0}
                max={1}
                value={generalThreshold}
                onChange={(e) => setGeneralThreshold(Number(e.target.value))}
                className="h-8 text-xs font-mono"
              />
            </Row>
            <Row label="角色阈值">
              <Input
                type="number"
                step="0.05"
                min={0}
                max={1}
                value={characterThreshold}
                onChange={(e) => setCharacterThreshold(Number(e.target.value))}
                className="h-8 text-xs font-mono"
              />
            </Row>
          </div>
          <Row label="LLM 输入">
            <Select
              value={captionSource}
              onValueChange={(v) =>
                v && setCaptionSource(v as typeof captionSource)
              }
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="vlm">视觉模型</SelectItem>
                <SelectItem value="tags">仅 WD14 标签</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Row label="训练用途">
            <Select
              value={captionMode}
              onValueChange={(v) => v && setCaptionMode(v as typeof captionMode)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="style">风格</SelectItem>
                <SelectItem value="character">角色</SelectItem>
                <SelectItem value="general">通用</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Row label="触发词">
            <Input
              value={triggerWord}
              onChange={(e) => setTriggerWord(e.target.value)}
              placeholder="anima style / @charA"
              className="h-8 text-xs"
            />
          </Row>
          {captionMode === "style" && (
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={stripStyleTags} onCheckedChange={setStripStyleTags} />
              剔除画风类标签
            </label>
          )}
          <Button
            size="sm"
            disabled={!imagePath.trim() || mutation.isPending}
            onClick={() => mutation.mutate()}
            className="w-full gap-1"
          >
            {mutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Play className="size-3" />
            )}
            运行 WD14 prefilter
          </Button>
        </div>
      </section>
      <PrefilterResultCard result={result} />
    </div>
  )
}

function PrefilterResultCard({ result }: { result: Wd14PrefilterResult | null }) {
  if (!result) {
    return (
      <section className="rounded-md border border-dashed border-border/60 p-3 text-xs text-muted-foreground flex items-center justify-center min-h-32">
        点左侧「运行」后结果会显示在这里。可直接拷给「VLM Anima 重写」。
      </section>
    )
  }
  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <ImageIcon className="size-3.5" />
        <span className="text-xs font-medium">prefilter 结果</span>
      </div>
      <div className="p-3 space-y-2 text-xs">
        <Field label="rating">{result.ratingName ?? "—"}</Field>
        <Field label="general tags">
          <span className="font-mono break-words">
            {result.generalTags.join(", ")}
          </span>
        </Field>
        <Field label="character tags">
          <span className="font-mono break-words">
            {result.characterTags.join(", ") || "—"}
          </span>
        </Field>
        <Field label="prompt">
          <pre className="font-mono whitespace-pre-wrap text-[11px] bg-muted/30 rounded p-2 max-h-64 overflow-auto">
            {result.promptText}
          </pre>
        </Field>
        <Button
          size="sm"
          variant="outline"
          className="w-full text-[11px]"
          onClick={() => {
            void navigator.clipboard.writeText(JSON.stringify(result, null, 2))
            toast.success("已拷贝 prefilter JSON")
          }}
        >
          复制 JSON
        </Button>
      </div>
    </section>
  )
}

// --------------------------------------------------------------------------- //
// ai-vlm-anima-rewrite — 单图测试，把 prefilter 的产出送到 VLM
// --------------------------------------------------------------------------- //

export function AiVlmAnimaRewriteTool({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [json, setJson] = useState("")
  const [mergeStrategy, setMergeStrategy] = useState("replace")
  const [result, setResult] = useState<{
    path: string
    wd14Tags: string
    caption: string
  } | null>(null)

  const mutation = useMutation({
    mutationFn: () => {
      let parsed: Wd14PrefilterResult
      try {
        parsed = JSON.parse(json) as Wd14PrefilterResult
      } catch {
        throw new Error("JSON 解析失败 — 请粘贴 prefilter 工具的完整输出")
      }
      return imageStudioVlmAnimaRewrite({
        path: parsed.path,
        mergeStrategy,
        captionMode: "style",
        captionSource: parsed.captionSource,
        stripStyleTags: parsed.stripStyleTags,
        ratingName: parsed.ratingName,
        generalTags: parsed.generalTags,
        characterTags: parsed.characterTags,
        promptText: parsed.promptText,
        dataUrl: parsed.dataUrl,
        skipLlm: parsed.skipLlm,
      })
    },
    onSuccess: (data) => {
      setResult({
        path: data.path,
        wd14Tags: data.wd14Tags,
        caption: data.caption,
      })
      qc.invalidateQueries({ queryKey: ["image-studio-captions-vocab", datasetPath] })
      qc.invalidateQueries({ queryKey: ["image-studio-audit-report", datasetPath] })
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    },
    onError: (err) =>
      toast.error("VLM 重写失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  return (
    <div className="h-full overflow-y-auto p-4 max-w-3xl space-y-3">
      <section className="rounded-md border border-border/60 bg-card">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Sparkles className="size-3.5" />
          <span className="text-xs font-medium">VLM Anima 重写 · 单图测试</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            把「WD14 单步出标签」工具复制出来的 JSON 粘到这里，调 VLM 写 caption
            并写入 <code>.txt</code>。 主要用来对比不同 prompt / merge 策略下的产出。
            数据集 <code>{datasetPath}</code> 仅用于校验路径前缀。
          </p>
          <textarea
            value={json}
            onChange={(e) => setJson(e.target.value)}
            placeholder='{"path":"...","ratingName":"...","generalTags":[...]...}'
            className="w-full h-40 rounded border bg-background px-2 py-1.5 text-[11px] font-mono resize-y"
          />
          <Row label="合并策略">
            <Select
              value={mergeStrategy}
              onValueChange={(v) => v && setMergeStrategy(v)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="replace">替换</SelectItem>
                <SelectItem value="append">追加</SelectItem>
                <SelectItem value="prepend">前置</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Button
            size="sm"
            disabled={!json.trim() || mutation.isPending}
            onClick={() => mutation.mutate()}
            className="w-full gap-1"
          >
            {mutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Play className="size-3" />
            )}
            写入 caption
          </Button>
        </div>
      </section>
      {result && (
        <section className="rounded-md border border-border/60 bg-card">
          <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
            <ImageIcon className="size-3.5" />
            <span className="text-xs font-medium">写入结果</span>
          </div>
          <div className="p-3 space-y-2 text-xs">
            <Field label="path">
              <code className="font-mono">{result.path}</code>
            </Field>
            <Field label="wd14 tags">
              <pre className="font-mono whitespace-pre-wrap text-[11px] bg-muted/30 rounded p-2 max-h-32 overflow-auto">
                {result.wd14Tags}
              </pre>
            </Field>
            <Field label="caption">
              <pre className="font-mono whitespace-pre-wrap text-[11px] bg-muted/30 rounded p-2 max-h-48 overflow-auto">
                {result.caption}
              </pre>
            </Field>
          </div>
        </section>
      )}
    </div>
  )
}

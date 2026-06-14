/**
 * CheckpointPlayback — fixed-prompt × fixed-seed sample timeline.
 *
 * For diffusion-model fine-tuning, the loss curve is famously a poor
 * proxy for visual quality. The honest signal is "what does this model
 * draw at step N for the same prompt + seed", scrolled side-by-side
 * across the training timeline. This panel does that.
 *
 * Grouping strategy:
 *   - Each sample image is parsed into ``(epoch?, step?, prompt?, seed?)``
 *     using filename heuristics that match the conventions used by the
 *     three backends:
 *       kohya:  ``<name>-<runId>_e<EE>_<NNNNNN>_p<II>_s<SEED>.png``
 *       dp:     ``samples/step_<N>/p<II>_s<SEED>.png``  (we emit step
 *                 from the parent dir when the file itself lacks it)
 *       anima:  ``sample/<runId>/<EE>-<NNNNNN>-<II>-<SEED>.png``
 *   - Rows are keyed by ``(promptIdx, seed)`` — the actual prompt text
 *     isn't in the filename, but the index suffices: row 0 is "the
 *     first prompt the user listed", row 1 is the second, etc.
 *   - Columns are sorted by step ascending; we don't deduplicate so a
 *     single row can have one cell per checkpoint.
 *
 * If the parser can't extract a step / prompt-index from any image
 * (older kohya runs, truly custom save_filename), the panel falls
 * back to a single chronological row labelled "未识别分组" — strictly
 * better than not rendering at all, and the existing samples-gallery
 * tab is the place users go for the unstructured view.
 */
import { useMemo } from "react"
import { ImageIcon, ZoomIn } from "lucide-react"
import { api, type JobFile } from "@/lib/api"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface ParsedSample {
  file: JobFile
  epoch: number | null
  step: number | null
  prompt: number | null
  seed: number | null
}

const FILENAME_RX = [
  // kohya canonical: ..._e000004_001500_p00_s12345.png  (the unsigned
  // ints can be padded with arbitrary zeros). We accept every component
  // as optional so a file missing one of them still parses the rest.
  /e(?:poch)?[-_]?(?<epoch>\d+).*?(?:s(?:tep)?[-_]?(?<step>\d+))?.*?p(?:rompt)?[-_]?(?<prompt>\d+).*?s(?:eed)?[-_]?(?<seed>\d+)/i,
  // anima compact: 0004-001500-00-12345.png — no letter prefixes.
  /^(?<epoch>\d+)-(?<step>\d+)-(?<prompt>\d+)-(?<seed>\d+)\b/,
  // anima_lora current: anima_lora_e000001_00_20260614010101_12345.png.
  /(?:^|_)e(?<epoch>\d+)_(?<prompt>\d+)_(?:\d{10,14})(?:_(?<seed>\d+))?/i,
  // anima_lora step path: anima_lora_001500_00_20260614010101_12345.png.
  /(?:^|_)(?<step>\d{3,})_(?<prompt>\d+)_(?:\d{10,14})(?:_(?<seed>\d+))?/i,
] as const

const SEED_ONLY_RX = /(?:^|[_-])(?<seed>\d{4,})\.[^.]+$/

function parseSample(file: JobFile): ParsedSample {
  const name = (file.path.split(/[\\/]/).pop() ?? file.path).toLowerCase()
  for (const rx of FILENAME_RX) {
    const m = name.match(rx)
    if (m?.groups) {
      const num = (k: string) =>
        m.groups?.[k] != null ? Number(m.groups[k]) : null
      return {
        file,
        epoch: num("epoch"),
        step: num("step"),
        prompt: num("prompt"),
        seed: num("seed"),
      }
    }
  }
  // Fallback: derive a step from the parent directory (`step_<N>` /
  // `<N>_steps`) — diffusion-pipe writes there. Promptlessly grouped.
  const parts = file.path.split(/[\\/]/)
  const stepFromDir = (() => {
    for (const part of parts) {
      const m = part.match(/^(?:step[-_]?)?(\d+)(?:[-_]?steps?)?$/i)
      if (m) return Number(m[1])
    }
    return null
  })()
  const seedFromName = name.match(SEED_ONLY_RX)?.groups?.seed
  const normalized = file.path.replace(/\\/g, "/")
  const epochFromName = normalized.match(/(?:^|[/_-])e(?:poch)?[-_]?(\d+)(?:[/_.-]|$)/i)?.[1]
  const stepFromName = normalized.match(/(?:^|[/_-])s(?:tep)?[-_]?(\d+)(?:[/_.-]|$)/i)?.[1]
  return {
    file,
    epoch: epochFromName ? Number(epochFromName) : null,
    step: stepFromName ? Number(stepFromName) : stepFromDir,
    prompt: null,
    seed: seedFromName ? Number(seedFromName) : null,
  }
}

function openExternal(url: string): void {
  window.open(url, "_blank", "noopener,noreferrer")
}

interface PlaybackRow {
  /** Stable key combining promptIdx + seed — rows are sorted by promptIdx asc. */
  rowId: string
  promptIdx: number | null
  seed: number | null
  /** Sorted ascending by step / epoch. */
  cells: ParsedSample[]
}

function groupSamples(samples: JobFile[]): {
  rows: PlaybackRow[]
  fallback: ParsedSample[]
  totalSteps: number[]
} {
  const parsed = samples.map(parseSample)
  const grouped = new Map<string, PlaybackRow>()
  const fallback: ParsedSample[] = []
  for (const s of parsed) {
    if (s.prompt == null && s.seed == null) {
      fallback.push(s)
      continue
    }
    const key = `${s.prompt ?? "?"}|${s.seed ?? "?"}`
    let row = grouped.get(key)
    if (!row) {
      row = {
        rowId: key,
        promptIdx: s.prompt,
        seed: s.seed,
        cells: [],
      }
      grouped.set(key, row)
    }
    row.cells.push(s)
  }
  // Sort each row's cells by step (or epoch when step missing).
  for (const row of grouped.values()) {
    row.cells.sort((a, b) => {
      const sa = a.step ?? (a.epoch ?? 0) * 1_000_000
      const sb = b.step ?? (b.epoch ?? 0) * 1_000_000
      return sa - sb
    })
  }
  const rows = Array.from(grouped.values()).sort((a, b) => {
    const pa = a.promptIdx ?? Number.MAX_SAFE_INTEGER
    const pb = b.promptIdx ?? Number.MAX_SAFE_INTEGER
    if (pa !== pb) return pa - pb
    return (a.seed ?? 0) - (b.seed ?? 0)
  })
  // Union of every step across rows — drives the column header.
  const stepSet = new Set<number>()
  for (const row of rows) {
    for (const c of row.cells) {
      const s = c.step ?? c.epoch
      if (s != null) stepSet.add(s)
    }
  }
  const totalSteps = Array.from(stepSet).sort((a, b) => a - b)
  return { rows, fallback, totalSteps }
}

export function CheckpointPlayback({
  jobId,
  samples,
  loading,
  triggerWord,
}: {
  jobId: string
  samples: JobFile[]
  loading: boolean
  /** 触发词，作为 LoRA 预览缩略图的橙色短角标（缩略图小，仅取前几位）。*/
  triggerWord?: string | null
}) {
  const { rows, fallback, totalSteps } = useMemo(
    () => groupSamples(samples),
    [samples],
  )
  // Column count drives the row width. Capped at 24 visible cells to
  // avoid an absurdly wide grid for thousand-step runs; older cells
  // scroll out the left side as the strip becomes scrollable.
  const totalRows = rows.length
  const totalCells = rows.reduce((acc, r) => acc + r.cells.length, 0)

  return (
    <Card
      className="analysis-fade-in-stagger overflow-hidden"
      style={{ ["--stagger-delay" as string]: "120ms" }}
    >
      <CardHeader className="py-2.5 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground inline-flex items-center gap-1.5">
          <ZoomIn className="size-3" /> 检查点回放 · 同 prompt × 同 seed
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70 tabular-nums">
          {loading
            ? "加载中…"
            : totalRows > 0
              ? `${totalRows} 组 prompt · ${totalCells} 张 · ${totalSteps.length} 个时间点`
              : `${samples.length} 张未分组`}
        </span>
      </CardHeader>
      <CardContent className="p-0">
        {samples.length === 0 && !loading && (
          <EmptyState />
        )}

        {totalRows > 0 && (
          <div className="max-h-[62vh] min-h-0 overflow-auto overscroll-contain">
            <table className="w-max border-separate border-spacing-0 text-[11px] tabular-nums">
              <thead>
                <tr className="text-muted-foreground/70">
                  <th className="sticky left-0 top-0 z-20 bg-muted/40 border-b border-r border-border/60 px-2.5 py-1.5 text-left text-[10px] uppercase tracking-[0.16em]">
                    prompt × seed
                  </th>
                  {totalSteps.map((s) => (
                    <th
                      key={s}
                      className={cn(
                        "sticky top-0 z-10 border-b border-border/60 bg-muted/40 px-1.5 py-1.5 text-center font-mono text-[10px] font-normal",
                        s === 0 && "bg-sky-500/10 text-sky-600 dark:text-sky-400",
                      )}
                    >
                      {s === 0 ? "基模" : Number.isInteger(s) ? `s${s}` : `e${s}`}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <PlaybackRowView
                    key={row.rowId}
                    jobId={jobId}
                    row={row}
                    columns={totalSteps}
                    triggerWord={triggerWord ?? null}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}

        {fallback.length > 0 && (
          <FallbackStrip
            jobId={jobId}
            entries={fallback}
            mixed={totalRows > 0}
          />
        )}
      </CardContent>
    </Card>
  )
}

function PlaybackRowView({
  jobId,
  row,
  columns,
  triggerWord,
}: {
  jobId: string
  row: PlaybackRow
  columns: number[]
  triggerWord: string | null
}) {
  // Index cells by their step / epoch key so we can render aligned
  // placeholders in columns where this row didn't sample.
  const byKey = useMemo(() => {
    const m = new Map<number, ParsedSample>()
    for (const c of row.cells) {
      const s = c.step ?? c.epoch
      if (s != null) m.set(s, c)
    }
    return m
  }, [row.cells])

  const label =
    row.promptIdx != null
      ? `p${row.promptIdx}${row.seed != null ? ` · s${row.seed}` : ""}`
      : row.seed != null
        ? `seed ${row.seed}`
        : row.rowId

  return (
    <tr>
      <th
        scope="row"
        className="sticky left-0 z-10 bg-background/95 border-b border-r border-border/60 px-2.5 py-1.5 text-left font-mono text-[11px] text-foreground/80"
      >
        {label}
      </th>
      {columns.map((col) => {
        const cell = byKey.get(col)
        return (
          <td
            key={col}
            className={cn(
              "border-b border-border/30 p-1 align-middle",
              col === 0 && "bg-sky-500/5",
            )}
          >
            {cell ? (
              <PlaybackCell
                jobId={jobId}
                sample={cell}
                isBaseline={col === 0}
                triggerWord={triggerWord}
              />
            ) : (
              <div className="size-16 rounded-[3px] border border-dashed border-border/40 bg-muted/20" />
            )}
          </td>
        )
      })}
    </tr>
  )
}

function PlaybackCell({
  jobId,
  sample,
  isBaseline,
  triggerWord,
}: {
  jobId: string
  sample: ParsedSample
  isBaseline?: boolean
  triggerWord?: string | null
}) {
  const url = api.jobFileUrl(jobId, sample.file.path)
  const name = sample.file.path.split(/[\\/]/).pop() ?? sample.file.path
  // 16×16 缩略图角标空间紧；触发词截到前 4 字符全大写。
  const loraBadgeFull = (triggerWord || "").trim() || "LORA"
  const loraBadge =
    loraBadgeFull.length > 4
      ? loraBadgeFull.slice(0, 4).toUpperCase()
      : loraBadgeFull.toUpperCase()
  return (
    <button
      type="button"
      onClick={() => openExternal(url)}
      className={cn(
        "group relative size-16 overflow-hidden rounded-[3px] border",
        "bg-muted/20 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40",
        isBaseline
          ? "border-sky-400/70 ring-1 ring-sky-400/30 hover:border-sky-500"
          : "border-amber-400/55 hover:border-amber-500/80 focus-visible:border-amber-500/80",
      )}
      title={
        isBaseline
          ? `[基模] ${name}`
          : `[${loraBadgeFull}] ${name}`
      }
    >
      <img
        src={url}
        alt={name}
        loading="lazy"
        className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.06]"
      />
      {isBaseline ? (
        <div className="pointer-events-none absolute inset-x-0 top-0 bg-sky-500/80 px-1 py-px text-center text-[8px] font-medium text-white leading-tight">
          BASE
        </div>
      ) : (
        <div
          className="pointer-events-none absolute left-0 top-0 max-w-full bg-amber-500/85 px-1 py-px text-[8px] font-medium uppercase tracking-tight text-white leading-tight rounded-br-[3px] truncate"
          title={loraBadgeFull}
        >
          {loraBadge}
        </div>
      )}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/65 to-transparent px-1 py-0.5 text-[9px] font-mono text-white opacity-0 transition group-hover:opacity-100">
        {sample.step != null
          ? `s${sample.step}`
          : sample.epoch != null
            ? `e${sample.epoch}`
            : ""}
      </div>
    </button>
  )
}

function FallbackStrip({
  jobId,
  entries,
  mixed,
}: {
  jobId: string
  entries: ParsedSample[]
  mixed: boolean
}) {
  return (
    <div className="border-t border-border/60 bg-muted/20 px-3.5 py-3">
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground mb-2">
        未识别分组（{entries.length}） {mixed && "· 文件名缺少 prompt / seed 信息"}
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {entries.map((s) => {
          const url = api.jobFileUrl(jobId, s.file.path)
          const name = s.file.path.split(/[\\/]/).pop() ?? s.file.path
          return (
            <button
              key={s.file.path}
              type="button"
              onClick={() => openExternal(url)}
              className="shrink-0 size-16 overflow-hidden rounded-[3px] border border-border/60 bg-muted/20 transition hover:border-primary/60"
              title={name}
            >
              <img
                src={url}
                alt={name}
                loading="lazy"
                className="h-full w-full object-cover"
              />
            </button>
          )
        })}
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-10 text-muted-foreground">
      <ImageIcon className="size-7 opacity-40" />
      <span className="text-[12px]">
        尚未生成样本图。在配置中开启 <code>sampling.everyNEpochs</code> 即可自动生成。
      </span>
    </div>
  )
}

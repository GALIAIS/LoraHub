// Minimal ANSI escape parser for the terminal-style log panel.
// Decodes SGR (Select Graphic Rendition) sequences for foreground colours and
// resets. Cursor movement, erase, mode and OSC sequences are silently dropped
// so they don't pollute rendered output. Anything we don't recognise is
// elided to keep the panel readable.

export interface AnsiChunk {
  text: string
  className: string
}

const FG_BASE: Record<number, string> = {
  30: "text-zinc-500",
  31: "text-red-400",
  32: "text-emerald-400",
  33: "text-amber-400",
  34: "text-blue-400",
  35: "text-fuchsia-400",
  36: "text-cyan-400",
  37: "text-zinc-200",
}

const FG_BRIGHT: Record<number, string> = {
  90: "text-zinc-400",
  91: "text-red-300",
  92: "text-emerald-300",
  93: "text-amber-300",
  94: "text-blue-300",
  95: "text-fuchsia-300",
  96: "text-cyan-300",
  97: "text-white",
}

function classFromSgr(params: number[]): string {
  // We only track foreground colour + bold + reset for simplicity. Background
  // colours and 256/truecolour sequences are intentionally collapsed.
  let cls = ""
  let i = 0
  while (i < params.length) {
    const code = params[i]
    if (code === 0) {
      cls = ""
    } else if (code === 1) {
      cls = cn(cls, "font-semibold")
    } else if (code === 22) {
      cls = (cls || "").replace(/\bfont-semibold\b/g, "").trim()
    } else if (code in FG_BASE) {
      cls = replaceFg(cls, FG_BASE[code])
    } else if (code in FG_BRIGHT) {
      cls = replaceFg(cls, FG_BRIGHT[code])
    } else if (code === 39) {
      cls = replaceFg(cls, "")
    } else if (code === 38 && params[i + 1] === 5 && typeof params[i + 2] === "number") {
      // 256-colour foreground — coarse-bucket to nearest base/bright tone.
      cls = replaceFg(cls, mapXterm256(params[i + 2]))
      i += 2
    } else if (code === 38 && params[i + 1] === 2) {
      // truecolour — drop colour info, keep other styling.
      i += 4
    }
    // unknown / background codes (40-49, 100-109) are intentionally ignored.
    i += 1
  }
  return cls.trim()
}

function replaceFg(cls: string, next: string): string {
  const stripped = (cls || "")
    .split(/\s+/)
    .filter((c) => c && !c.startsWith("text-"))
    .join(" ")
  return next ? cn(stripped, next) : stripped
}

function mapXterm256(n: number): string {
  // Standard ANSI block (0-15) maps directly onto our base/bright palette;
  // anything else falls back to a neutral tone so the log stays legible.
  const direct: Record<number, string> = {
    1: FG_BASE[31],
    2: FG_BASE[32],
    3: FG_BASE[33],
    4: FG_BASE[34],
    5: FG_BASE[35],
    6: FG_BASE[36],
    7: FG_BASE[37],
    9: FG_BRIGHT[91],
    10: FG_BRIGHT[92],
    11: FG_BRIGHT[93],
    12: FG_BRIGHT[94],
    13: FG_BRIGHT[95],
    14: FG_BRIGHT[96],
    15: FG_BRIGHT[97],
  }
  if (n in direct) return direct[n]
  return "text-zinc-300"
}

function cn(...parts: (string | undefined | null | false)[]): string {
  return parts.filter(Boolean).join(" ").replace(/\s+/g, " ").trim()
}

export function parseAnsi(input: string): AnsiChunk[] {
  if (!input) return []
  const out: AnsiChunk[] = []
  // ESC = 0x1B, then `[` for CSI, plus param/intermediate bytes ending on a final byte.
  // We also drop OSC (`]…ST`/BEL) and any other escape sequence as plain elisions.
  const ESC = "\x1b"
  let buf = ""
  let cls = ""
  let i = 0
  const flush = () => {
    if (buf) {
      out.push({ text: buf, className: cls })
      buf = ""
    }
  }
  while (i < input.length) {
    const ch = input[i]
    if (ch !== ESC) {
      buf += ch
      i += 1
      continue
    }
    flush()
    // Strip the escape itself.
    i += 1
    if (i >= input.length) break
    const next = input[i]
    if (next === "[") {
      // CSI sequence: collect params/intermediates until a final byte 0x40-0x7E.
      i += 1
      let params = ""
      while (i < input.length) {
        const c = input[i]
        const code = c.charCodeAt(0)
        if (code >= 0x40 && code <= 0x7e) {
          // Final byte — only honour SGR (`m`); discard cursor/erase/etc.
          if (c === "m") {
            const nums = params
              .split(";")
              .map((p) => (p === "" ? 0 : Number(p)))
              .filter((n) => Number.isFinite(n))
            cls = classFromSgr(nums)
          }
          i += 1
          break
        }
        params += c
        i += 1
      }
    } else if (next === "]") {
      // OSC sequence: scan to BEL (0x07) or ST (ESC \).
      i += 1
      while (i < input.length) {
        const c = input[i]
        if (c === "\x07") {
          i += 1
          break
        }
        if (c === ESC && input[i + 1] === "\\") {
          i += 2
          break
        }
        i += 1
      }
    } else {
      // Two-char escape (e.g. ESC c, ESC =) — drop both bytes.
      i += 1
    }
  }
  flush()
  return out
}

/**
 * Strip ANSI sequences entirely — useful for the search filter where we want
 * to match against the rendered text rather than raw escape bytes.
 */
export function stripAnsi(input: string): string {
  return parseAnsi(input)
    .map((c) => c.text)
    .join("")
}

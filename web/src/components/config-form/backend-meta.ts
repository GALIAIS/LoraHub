/**
 * Single source of truth for backend metadata: which arches each backend
 * supports, what badge to render, default arch when switching, etc.
 *
 * Mirrors the per-backend `supported_archs` set in
 * `lorahub/core/backends/<id>/backend.py`. The frontend uses this map to:
 *   - filter the arch select on the config editor (only show arches the
 *     selected backend can actually train);
 *   - render a colored backend badge on every config row + template card
 *     so users can scan a list and tell what each recipe targets;
 *   - drive the global "Settings → default backend" filter on the
 *     /configs list (hide configs from other backends by default);
 *   - choose a sensible initial arch when the user creates a fresh
 *     config or flips the backend type on an existing one.
 */
import type { BackendId } from "@/lib/api"

/** All arches the LoraHub schema knows about, mirroring `ModelArch`. */
export type Arch =
  | "sd15"
  | "sd2"
  | "sdxl"
  | "sd3"
  | "flux"
  | "flux2"
  | "lumina"
  | "hunyuan_image"
  | "anima"
  | "chroma"
  | "hidream"
  | "omnigen2"
  | "auraflow"
  | "qwen_image"
  | "cosmos"
  | "cosmos_predict2"
  | "hunyuan_video"
  | "hunyuan_video_15"
  | "ltx_video"
  | "ltx2"
  | "wan"
  | "z_image"
  | "ernie_image"

/**
 * Per-backend supported arch sets. Mirrors:
 *   - kohya: `lorahub/core/backends/kohya/backend.py` `_SUPPORTED`
 *   - dp:    `_DP_MODEL_TYPE_MAP` keys
 *   - anima_lora: only `anima` (purpose-built)
 *
 * If a future backend lands here, update both this map and the Python
 * side of the corresponding backend.py.
 */
export const SUPPORTED_ARCHS_BY_BACKEND: Record<BackendId, ReadonlySet<Arch>> = {
  kohya: new Set<Arch>([
    "sd15",
    "sd2",
    "sdxl",
    "sd3",
    "flux",
    "lumina",
    "hunyuan_image",
    "anima",
  ]),
  "diffusion-pipe": new Set<Arch>([
    "sdxl",
    "sd3",
    "flux",
    "flux2",
    "lumina",
    "hunyuan_image",
    "anima",
    "chroma",
    "hidream",
    "omnigen2",
    "auraflow",
    "qwen_image",
    "cosmos",
    "cosmos_predict2",
    "hunyuan_video",
    "hunyuan_video_15",
    "ltx_video",
    "ltx2",
    "wan",
    "z_image",
    "ernie_image",
  ]),
  anima_lora: new Set<Arch>(["anima"]),
}

/** Default arch when the user picks a backend on a fresh config. */
export const DEFAULT_ARCH_BY_BACKEND: Record<BackendId, Arch> = {
  kohya: "sdxl",
  "diffusion-pipe": "flux",
  anima_lora: "anima",
}

export interface BackendBadge {
  /** Short label rendered inside the chip. */
  label: string
  /** Tailwind class fragment for the chip background + border + text. */
  toneClass: string
  /** Long-form description used on tooltips / detail panes. */
  description: string
}

/**
 * Per-backend badge styling. Tones picked from the existing CSS variables
 * (`--primary` = blue, `--accent` = grey, `chart-3` = teal-ish) so the
 * badges reuse the workbench's palette.
 */
export const BACKEND_BADGE: Record<BackendId, BackendBadge> = {
  kohya: {
    label: "kohya",
    toneClass: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    description: "kohya-ss/sd-scripts — 经典 LoRA / DreamBooth 训练",
  },
  "diffusion-pipe": {
    label: "diffusion-pipe",
    toneClass: "border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-300",
    description: "tdrussell/diffusion-pipe — DeepSpeed pipeline,涵盖图像/视频",
  },
  anima_lora: {
    label: "anima_lora",
    toneClass: "border-cyan-500/40 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300",
    description: "sorryhyun/anima_lora — Anima DiT 专用,带 OrthoLoRA / T-LoRA / DMD turbo",
  },
}

/** Helper: is this arch supported by this backend? */
export function isArchSupported(backend: BackendId, arch: string | undefined): boolean {
  if (!arch) return false
  const set = SUPPORTED_ARCHS_BY_BACKEND[backend]
  return set ? set.has(arch as Arch) : false
}

/** Helper: pick a sensible default arch for a backend (used on type swap). */
export function defaultArchFor(backend: BackendId): Arch {
  return DEFAULT_ARCH_BY_BACKEND[backend] ?? "sdxl"
}

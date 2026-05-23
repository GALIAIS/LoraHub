/**
 * Aggregates the frontend's build-time version (`__APP_VERSION__`,
 * injected by vite.config.ts from `git describe --tags --dirty --always`)
 * with the backend's runtime version (`/api/health.version`, derived
 * from `lorahub.__version__` via hatch-vcs).
 *
 * Why this matters: the install/update flow rebuilds the Python venv
 * but skips the SPA build when `web/dist` looks fresh enough
 * (see `scripts/remote_setup.sh:build_frontend`). On a `git pull` that
 * bumped both Python and TS files, the user can end up with a new
 * backend + stale `web/dist`, and there's no obvious signal in the UI
 * — until something silently breaks because a new endpoint returns
 * a shape the old SPA doesn't understand.
 *
 * Surfacing the two version strings side-by-side lets the user spot
 * the drift at a glance without having to dig through `/api/health`
 * or check git status.
 */
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"

const FIVE_MINUTES = 5 * 60 * 1000

export interface VersionInfo {
  /** Raw frontend version baked into the bundle at build time. */
  frontend: string
  /** Raw backend version reported by `/api/health`. */
  backend: string | null
  /** True until /api/health responds for the first time. */
  loading: boolean
  /**
   * True only when frontend and backend are demonstrably on different
   * commits. We compare git short shas first (the canonical answer);
   * if either side doesn't expose one we fall back to comparing
   * cleaned tag strings. Loading / unknown literals never match.
   */
  mismatch: boolean
  /** Compact display string for the frontend, e.g. ``1.0.3-g7ddfe78``. */
  frontendDisplay: string
  /** Compact display string for the backend. */
  backendDisplay: string
}

/**
 * Strip noise to get a clean tag-style version for display purposes.
 *
 * Two formats fly around the project:
 *
 *   git describe (frontend):
 *     `1.0.2-85-g7ddfe7852[-dirty]` → drop the `-N-g<sha>` and `-dirty`
 *     suffixes → `1.0.2`. The base is the **last** released tag.
 *
 *   hatch-vcs (backend):
 *     `1.0.3.dev85+g7ddfe7852.d20260523` → drop the `.devN+g<sha>(.d<date>)?`
 *     suffix → `1.0.3`. The base is the **next** unreleased tag (PEP 440
 *     guarantees `1.0.3.devN < 1.0.3` so the resolver never confuses
 *     a dev build for a release).
 *
 * The two bases will inevitably disagree on every untagged commit
 * (frontend reads "last tag", backend reads "next tag"). For a real
 * mismatch detector we compare commit shas instead — see
 * ``extractCommitSha`` below. ``baseVersion`` is purely for display.
 */
export function baseVersion(raw: string | null | undefined): string {
  if (!raw) return ""
  return raw
    .replace(/-dirty$/, "")
    .replace(/-\d+-g[0-9a-f]+$/, "")            // git describe between tags
    .replace(/\.dev\d+\+g[0-9a-f]+(\.d\d+)?$/, "") // hatch-vcs
    .replace(/^v/, "")
}

/**
 * Pull the git short sha out of either format. ``g`` prefix is the
 * universal marker — git describe writes it (`-g7ddfe78`) and hatch-vcs
 * writes it (`+g7ddfe78`). When both sides expose a sha, comparing
 * shas is the only correct "are we on the same code?" check.
 */
export function extractCommitSha(raw: string | null | undefined): string | null {
  if (!raw) return null
  const match = raw.match(/g([0-9a-f]{7,40})/i)
  return match ? match[1].toLowerCase() : null
}

const UNKNOWN_LITERALS = new Set(["", "dev", "0.0.0+unknown", "0.0.0"])

export function useVersionInfo(): VersionInfo {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    staleTime: FIVE_MINUTES,
    refetchOnWindowFocus: false,
  })
  const frontend = __APP_VERSION__
  const backend = health.data?.version ?? null

  const fSha = extractCommitSha(frontend)
  const bSha = extractCommitSha(backend)
  const fBase = baseVersion(frontend)
  const bBase = baseVersion(backend)

  // Sha-first comparison: if both sides expose a commit hash we trust
  // it absolutely. Otherwise (released builds, no `+g<sha>` segment on
  // either side) we fall back to comparing the cleaned tag strings.
  let mismatch: boolean
  if (health.isLoading) {
    mismatch = false
  } else if (fSha && bSha) {
    mismatch = fSha !== bSha
  } else {
    mismatch =
      !!fBase &&
      !!bBase &&
      !UNKNOWN_LITERALS.has(fBase) &&
      !UNKNOWN_LITERALS.has(bBase) &&
      fBase !== bBase
  }

  // Display strategy. The two version strings come from different
  // tools that have *different opinions* about what "base version"
  // means on an untagged commit:
  //
  //   * git describe (frontend) → last released tag (e.g. 1.0.2)
  //   * hatch-vcs (backend)     → next unreleased tag (e.g. 1.0.3.dev85)
  //
  // Showing both raw forms confuses users into thinking the two
  // halves are out of sync when they're actually on the same commit.
  // When shas match we display the same canonical string on both
  // sides — anchored on the more user-recognisable git-describe base
  // (the last tag they actually cut) plus the short sha. When shas
  // diverge we surface each side's *own* base so the user can see
  // the drift.
  let frontendDisplay: string
  let backendDisplay: string
  if (fSha && bSha && fSha === bSha) {
    // Same commit — anchor display on the released tag (frontend's
    // base) so users see a number tied to their last release.
    const canonical = formatDisplay(fBase || bBase, fSha)
    frontendDisplay = canonical
    backendDisplay = canonical
  } else {
    frontendDisplay = formatDisplay(fBase, fSha)
    backendDisplay = formatDisplay(bBase, bSha)
  }

  return {
    frontend,
    backend,
    loading: health.isLoading,
    mismatch,
    frontendDisplay,
    backendDisplay,
  }
}

/**
 * Compose the user-facing short string. Released tags collapse to
 * just the version (`1.0.3`); dev builds show ``base-g<sha7>`` so the
 * user can correlate with `git log` without parsing PEP 440.
 */
function formatDisplay(base: string, sha: string | null): string {
  if (!base) return "?"
  if (!sha) return base
  return `${base}-g${sha.slice(0, 7)}`
}

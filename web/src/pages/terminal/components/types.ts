/**
 * Shared types for the terminal page subcomponents.
 *
 * One scrollback entry per `TerminalLine`. The `kind` discriminates how
 * the line is rendered (colour + prefix). `prompt` lines also store the
 * synthetic `(backend) cwd$` prefix so the page can render it even
 * after the active backend changes underneath.
 */

export type TerminalLineKind =
  | "prompt"
  | "stdout"
  | "stderr"
  | "info"
  | "error"

export interface TerminalLine {
  kind: TerminalLineKind
  text: string
  /** Set when kind === "prompt". The cwd-style prefix line. */
  prompt?: string
}

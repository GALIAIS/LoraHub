// Mirror presets + probe results (network panel in settings).

export interface MirrorPreset {
  label: string
  value: string
  probe: string
}

export interface ProbeResult {
  label: string
  value: string
  probe: string
  ok: boolean
  status: number | null
  latency_ms: number | null
  error: string | null
}

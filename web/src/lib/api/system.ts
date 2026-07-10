import { useEventStream } from "../use-event-stream"

// ============================================ system telemetry =============

export interface SystemHost {
  hostname: string
  system: string
  release: string
  python: string
}

export interface SystemCpu {
  cores_logical: number
  cores_physical: number | null
  usage_percent: number | null
  per_core_percent: number[]
  load_average: number[] | null
  arch: string
  frequency_mhz: number | null
  cpu_temperature_c: number | null
  // Newer fields - all optional so older snapshots still type-check.
  model?: string
  frequency_min_mhz?: number | null
  frequency_max_mhz?: number | null
  frequency_per_core_mhz?: number[]
}

export interface SystemMemory {
  total_bytes: number
  used_bytes: number
  available_bytes: number
  percent: number
  swap_total_bytes: number | null
  swap_used_bytes: number | null
}

export interface SystemDisk {
  path: string
  label: string
  total_bytes: number
  used_bytes: number
  free_bytes: number
  percent: number
}

export interface SystemGpu {
  index: number
  name: string
  driver: string | null
  memory_total_bytes: number | null
  memory_used_bytes: number | null
  memory_free_bytes: number | null
  utilization_percent: number | null
  temperature_c: number | null
  power_w: number | null
  power_limit_w: number | null
  fan_percent: number | null
  vendor: "nvidia" | "amd" | "intel" | "apple" | "qemu" | "unknown" | string
  // PCIe link state (optional - older backends don't emit these).
  pcie_gen_current?: number | null
  pcie_width_current?: number | null
  pcie_gen_max?: number | null
  pcie_width_max?: number | null
  // Clocks (MHz).
  sm_clock_mhz?: number | null
  mem_clock_mhz?: number | null
  sm_clock_max_mhz?: number | null
  mem_clock_max_mhz?: number | null
}

export interface ProcessInfo {
  pid: number
  name: string
  cpu_percent: number
  memory_rss_bytes: number
  memory_percent: number
}

export interface InterfaceAddress {
  family: string
  address: string
  netmask?: string | null
  broadcast?: string | null
}

export type NetworkInterfaceKind = "physical" | "loopback" | "virtual" | "wireless"

export interface NetworkInterfaceStats {
  name: string
  is_up: boolean
  speed_mbps: number | null
  mtu: number | null
  addresses: InterfaceAddress[]
  bytes_sent_total: number
  bytes_recv_total: number
  bytes_sent_per_sec: number
  bytes_recv_per_sec: number
  packets_sent_total: number
  packets_recv_total: number
  errors_in: number
  errors_out: number
  drops_in: number
  drops_out: number
  kind: NetworkInterfaceKind
}

export interface TcpConnectionStats {
  total: number
  established: number
  listen: number
  time_wait: number
  close_wait: number
  other: number
}

export interface PublicIpInfo {
  ip: string | null
  fetched_at: number
  source: "ip.sb" | "ipinfo.io" | "cached" | "unreachable" | string
}

export interface DiskIoDevice {
  device: string
  read_bytes_per_sec: number
  write_bytes_per_sec: number
  read_ops_per_sec: number
  write_ops_per_sec: number
}

export interface DiskIoStats {
  read_bytes_total: number
  write_bytes_total: number
  read_bytes_per_sec: number
  write_bytes_per_sec: number
  read_ops_per_sec: number
  write_ops_per_sec: number
  per_device: DiskIoDevice[]
}

export interface GpuProcessInfo {
  gpu_index: number
  pid: number
  process_name: string
  used_memory_mib: number
  type: "C" | "G" | "C+G" | string
}

export interface SystemBattery {
  percent: number
  plugged: boolean | null
  secs_left: number | null
}

export interface SystemSnapshot {
  timestamp: number
  has_psutil: boolean
  has_nvidia_smi: boolean
  host: SystemHost
  cpu: SystemCpu
  memory: SystemMemory
  disks: SystemDisk[]
  gpus: SystemGpu[]
  battery: SystemBattery | null
  network: {
    bytes_sent_total: number
    bytes_recv_total: number
    bytes_sent_per_sec: number
    bytes_recv_per_sec: number
    // Newer fields - all optional so older snapshots still parse.
    interfaces?: NetworkInterfaceStats[]
    tcp_connections?: TcpConnectionStats | null
    public_ip?: PublicIpInfo | null
  } | null
  // Newer top-level fields - all optional.
  processes?: ProcessInfo[]
  disk_io?: DiskIoStats | null
  gpu_processes?: GpuProcessInfo[]
}

// --- Self-update -----------------------------------------------------

export interface UpdateInfo {
  channel: "dev" | "tag"
  current: string
  latest: string | null
  update_available: boolean
  release_url: string
  release_notes: string
  checked_at: string
  is_dirty: boolean
  error: string | null
  tag_name: string | null
  published_at: string | null
  current_commit?: string | null
  latest_commit?: string | null
  /**
   * Where the `current` version string was sourced from. Anything
   * other than `hatch-vcs` means the install can't read its own git
   * tags (typical for ZIP-extracted trees) — the UI surfaces a
   * tooltip so users understand the version may lag a commit.
   */
  version_source: "env" | "git-describe" | "hatch-vcs" | "dist-metadata" | "changelog" | "fallback"
  /**
   * `false` iff this install is not a real git checkout. The
   * updater can't function on archive extracts or Docker image
   * installs, so the UI greys out the apply button when this flag is
   * false.
   */
  git_checkout: boolean
  install_kind?: "git" | "archive" | "docker" | string
}

export interface UpdateEvent {
  phase: "git" | "deps" | "build" | "done" | "restart" | "error"
  level: "info" | "warn" | "error"
  message: string
}

export interface ReleaseVersion {
  tag_name: string
  commit: string | null
}

/**
 * Subscribe to /api/system/sse for hardware telemetry. Falls back to the
 * legacy WS endpoint when EventSource isn't available.
 *
 * SSE has the proxy-friendly story we want: no upgrade handshake, no
 * AutoDL idle-kill (we send `: ping` comments), and the browser handles
 * reconnection on its own with the `retry: <ms>` directive the server
 * emits on connect.
 */
export function useSystemStream(enabled = true) {
  const { state, status } = useEventStream<
    SystemSnapshot | null,
    SystemSnapshot
  >({
    ssePath: enabled ? "/api/system/sse" : null,
    wsPath: enabled ? "/api/system/stream" : null,
    initialState: null,
    // Snapshot semantics: every frame replaces the previous state.
    // The history view is built from polled REST plus the live tail,
    // not a buffer here.
    reduce: (_prev, parsed) => parsed,
    reconnectOnVisibility: true,
  })
  return { snapshot: state, status }
}

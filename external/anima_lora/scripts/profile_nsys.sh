#!/usr/bin/env bash
# Nsight Systems timeline pass for training-loop bottleneck inspection.
#
# What this answers (in order of usefulness):
#   1. Where does wall-clock time go per step? — forward vs backward vs
#      optimizer NVTX ranges are emitted by library/training/loop.py.
#   2. Is the GPU busy or starved? — look for gaps between kernels on the
#      CUDA HW row; gaps == CPU-bound (dataloader, Python, host syncs).
#   3. Which kernels/categories dominate? — `nsys stats` summary printed
#      after the run; full per-kernel ranking lives in the .nsys-rep.
#   4. Are graphs actually capturing? — cudaGraphLaunch rows should appear
#      under production compile (reduce-overhead). Their absence means
#      cudagraph_trees fell back to eager.
#
# For per-kernel comp-vs-mem (SOL %, memory workload) you want ncu, not this.
# Memory `project_attention_compute_bound` already pins attention at 86-89%
# SM SOL on this box — start with the timeline; only drop to ncu if you've
# identified a specific kernel worth drilling into.
#
# CUDA Graphs: KEPT ON (unlike the old ncu script). nsys traces straight
# through cudaGraphLaunch so the production reduce-overhead path is the
# one you actually want to see.

set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${NSYS_OUT:-output/nsys/profile}"
mkdir -p "$(dirname "$OUT")"

# Profile window. 5 steps gives a clean repeating pattern (avoids step-0
# warmup noise and lets you eyeball variance step-to-step). Bump if you're
# chasing tail-step phenomena like a saver firing on step N.
PROFILE_START="${PROFILE_START:-3}"
PROFILE_END="${PROFILE_END:-7}"

# Tracing categories. Defaults cover what you almost always want:
#   cuda    — kernels, memcpy, cudaGraphLaunch, runtime API
#   nvtx    — forward/backward/optimizer ranges from loop.py
#   osrt    — pthread/syscall waits (catches Python GIL stalls + dataloader
#             blocking on file I/O)
#   cudnn   — conv/norm dispatch boundaries
#   cublas  — GEMM dispatch boundaries (matches kernels back to their call)
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,}"

# Sampling. CPU IP/backtrace sampling adds overhead and clutter for a
# GPU-bound trainer; leave off unless you suspect Python is the culprit.
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"

# Whether to also dump cuda-memory-usage events. Off by default — they
# fatten the .nsys-rep substantially and only matter when you're chasing
# an alloc/free pattern.
NSYS_CUDA_MEM="${NSYS_CUDA_MEM:-false}"

# Symbol resolution. Off by default — resolving DWARF / kernel-symbol info
# at trace-end stalls for a long time fetching symbol files (the live
# pain point of this script). Kernel mangled names alone are enough to
# rank kernels in `nsys stats`; turn on only if you need readable Python
# frames or libstdc++ syscall names. Pair with --cudabacktrace=none so no
# backtraces are captured in the first place (nothing to resolve).
NSYS_RESOLVE_SYMBOLS="${NSYS_RESOLVE_SYMBOLS:-false}"
NSYS_CUDABACKTRACE="${NSYS_CUDABACKTRACE:-none}"

METHOD="${METHOD:-chimera}"
PRESET="${PRESET:-default}"

echo "[nsys] export -> ${OUT}.nsys-rep"
echo "[nsys] step ${PROFILE_START}-${PROFILE_END}, trace=${NSYS_TRACE}, sample=${NSYS_SAMPLE}, cuda-mem=${NSYS_CUDA_MEM}"
echo "[nsys] resolve-symbols=${NSYS_RESOLVE_SYMBOLS}, cudabacktrace=${NSYS_CUDABACKTRACE}"
echo "[nsys] method=${METHOD} preset=${PRESET}"

# --capture-range=cudaProfilerApi + --capture-range-end=stop pairs with
# loop.py's torch.cuda.profiler.start()/stop() so the .nsys-rep only
# contains the profile window — not the cold-start text-encoder caching,
# VAE caching, compile warmup, etc. The profiler.stop() at PROFILE_END
# also ends the capture (capture-range-end=stop), which is why we don't
# need a separate --duration.
nsys profile \
    --output "$OUT" \
    --force-overwrite true \
    --trace "$NSYS_TRACE" \
    --sample "$NSYS_SAMPLE" \
    --cuda-memory-usage "$NSYS_CUDA_MEM" \
    --cudabacktrace "$NSYS_CUDABACKTRACE" \
    --resolve-symbols "$NSYS_RESOLVE_SYMBOLS" \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    python -m accelerate.commands.accelerate_cli launch \
        --num_cpu_threads_per_process 3 \
        --mixed_precision bf16 \
        train.py \
        --method "$METHOD" --preset "$PRESET" \
        --profile_steps "${PROFILE_START}-${PROFILE_END}" \
        --max_train_steps "$((PROFILE_END + 2))"

echo
echo "[nsys] === summary (nsys stats) ==="
echo "[nsys] open ${OUT}.nsys-rep in the Nsight Systems GUI for the full timeline."
echo

# Terminal-friendly rankings: which kernels and which NVTX ranges
# dominate. Skips the per-call detail you'd get in the GUI — that's the
# point: if a kernel doesn't appear in the top of these, it's not worth
# drilling into.
nsys stats \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report nvtx_sum \
    --format column \
    "${OUT}.nsys-rep" || echo "[nsys] stats failed (open the .nsys-rep in the GUI instead)"

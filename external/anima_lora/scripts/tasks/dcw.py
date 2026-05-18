"""DCW v4 calibrator: collect calibration data + train fusion head.

`make dcw` runs `scripts/dcw/measure_bias.py --dump_per_sample_gaps` once
per aspect bucket (top-5 by sample count: 832×1248, 896×1152, 768×1344,
1152×896, 1248×832) at the production env (CFG=4, mod_w=3.0), then
chains `scripts/dcw/train_fusion_head.py`. End artifact is
`output/dcw/<timestamp>-v4-fusion-head-make-dcw/fusion_head.safetensors`,
which `make test-dcw-v4` auto-resolves.

The trainer is bucket-agnostic (single population μ_g, aspect_emb pinned
to zero) — see `project_dcw_bucket_prior_cosmetic` memory. Per-bucket
sampling is kept only to balance the prompt pool across aspect buckets.

Reverse trajectories are collected with `--baseline_lambda 0.01`
(one_minus_sigma) baked in, matching `make test-dcw`'s scalar default.
The head therefore learns the residual α̂ on top of that scalar; at
inference the calibrator applies `0.01·(1−σ)` on every step plus
`α̂·gain·(1−σ)` once the head has fired. No dead-zone mismatch — g_obs
is observed on the same trajectory inference will use, and warmup steps
get the same correction as the rest. Override with `--baseline_lambda 0`
to fall back to the legacy no-DCW baseline.

Incremental gathers: each measure_bias run drops a `manifest.json` listing
the (stem, seed) pairs it collected. On every subsequent `make dcw`,
`_scan_used_stems` walks `output/dcw/*/manifest.json` matching the current
(bucket, baseline_lambda) and writes the union to
`output/dcw/.exclude/<HxW>.txt`, which is passed to measure_bias.py via
`--exclude_stems`. Re-running `make dcw` therefore grows the calibration
pool with fresh prompts instead of re-sampling the same stems. Pass
`--allow_repeats` to opt out (e.g. when you want a deliberate seed
re-roll on the same prompts).

`make dcw-train` skips the sampling phase and trains on the existing pool.

See `docs/proposal/dcw-learnable-calibrator-v4.md` §I1.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

from library.datasets.buckets import DCW_ASPECT_BUCKETS

from ._common import run


def _pop_kv(extra: list[str], key: str, default: str) -> tuple[str, list[str]]:
    """Extract ``--key value`` from extra. Returns (value, remaining_extra)."""
    if key in extra:
        i = extra.index(key)
        if i + 1 >= len(extra):
            sys.exit(f"missing value after {key}")
        value = extra[i + 1]
        return value, extra[:i] + extra[i + 2 :]
    return default, list(extra)


def _pop_flag(extra: list[str], key: str) -> tuple[bool, list[str]]:
    """Extract a boolean ``--flag`` from extra. Returns (present, remaining)."""
    if key in extra:
        i = extra.index(key)
        return True, extra[:i] + extra[i + 1 :]
    return False, list(extra)


def _latest_bucket_dir(out_root: Path, bucket_label: str) -> Path | None:
    """Most recently modified ``<ts>-<bucket_label>/`` dir under ``out_root``."""
    if not out_root.exists():
        return None
    matches = sorted(
        (
            p
            for p in out_root.iterdir()
            if p.is_dir() and p.name.endswith(f"-{bucket_label}")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _emit_aggregate_plot(
    out_root: Path, bucket_dirs: list[Path], label: str
) -> None:
    """Pool baseline trajectories across bucket runs and write a single plot.

    Replaces the per-bucket gap_curves.png that we suppress via
    ``--no-save_plot`` — one plot summarizing the whole data-gen phase,
    weighted by each bucket's (n_samples × n_seeds).
    """
    if not bucket_dirs:
        print("\n=== aggregate plot: no bucket dirs found, skipping ===")
        return
    from scripts.dcw.output import aggregate_run_dirs, make_aggregate_plot

    accum, per_bucket, n_steps = aggregate_run_dirs(bucket_dirs)
    if not accum:
        print("\n=== aggregate plot: no usable bucket data, skipping ===")
        return
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    agg_dir = out_root / f"{ts}-{label}-aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    written = make_aggregate_plot(agg_dir, accum, per_bucket, n_steps)
    if written:
        n = accum["baseline"]["n"]
        print(
            f"\n=== aggregate plot ({len(per_bucket)} buckets, n={n} trajectories) "
            f"→ {agg_dir / 'gap_curves.png'} ==="
        )


def _scan_used_stems(
    out_root: Path,
    H: int,
    W: int,
    baseline_lambda: float,
) -> tuple[set[str], int]:
    """Walk ``out_root`` for prior runs' manifest.json matching this bucket
    and baseline_lambda. Returns (used_stems, n_runs_matched).

    Filtering on baseline_lambda keeps λ=0 and λ=0.01 pools separate — they
    train different heads (residual vs no-residual), so reusing one for the
    other would break the trainer's contract. Manifests with mismatched
    baseline_lambda are skipped silently.

    Runs without a manifest.json are treated as failed/incomplete and
    correctly remain re-eligible.
    """
    used: set[str] = set()
    n_runs = 0
    if not out_root.exists():
        return used, n_runs
    for manifest_path in sorted(out_root.glob("*/manifest.json")):
        try:
            data = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        bucket = data.get("bucket")
        if bucket is None or bucket[0] != H or bucket[1] != W:
            continue
        prior_lam = data.get("baseline_lambda")
        if prior_lam is None or not math.isclose(
            float(prior_lam), baseline_lambda, abs_tol=1e-9
        ):
            continue
        for pair in data.get("pairs", []):
            stem = pair.get("stem")
            if stem:
                used.add(stem)
        n_runs += 1
    return used, n_runs


def cmd_dcw(extra):
    """Sample baseline trajectories per bucket, then train fusion head.

    Generates ``--n_images`` × ``--n_seeds`` baseline trajectories per
    bucket (default 130×1) into ``output/dcw/``, then trains the head on
    the pooled rows. Buckets only stratify the prompt pool — the trainer
    aggregates them.

    Defaults reflect the 2026-05-05 findings: prompt breadth dominates
    seed multiplicity for r_α (single-seed labels carry only ~13% noise
    floor at production target_window 7:; seed-mean averaging is
    net-harmful at this data scale — see project_dcw_seed_variance_dominates).
    n_images=130 is capped at the rarest top-5 bucket so all buckets stay
    aspect-balanced (132 stems available for 1248×832). ``--shuffle_seed=0``
    deterministically randomizes selection across the cache's 14×
    headroom (2477 stems vs 175 previously sampled).

    Other extra args pass through to every measure_bias invocation
    (--dit, --lora_weight, --pooled_text_proj '', --guidance_scale, etc.).
    """
    n_images, extra = _pop_kv(extra, "--n_images", "8")
    n_seeds, extra = _pop_kv(extra, "--n_seeds", "2")
    shuffle_seed, extra = _pop_kv(extra, "--shuffle_seed", "0")
    label, extra = _pop_kv(extra, "--label", "make-dcw")
    # Match make-test-dcw's default scalar so the trained head learns
    # the residual α̂ on top — kills the v4 dead-zone mismatch (head
    # observes / acts on the same trajectory inference will).
    baseline_lambda, extra = _pop_kv(extra, "--baseline_lambda", "0.0")

    allow_repeats, extra = _pop_flag(extra, "--allow_repeats")

    out_root = "output/dcw"
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)
    exclude_dir = out_root_path / ".exclude"
    if not allow_repeats:
        exclude_dir.mkdir(parents=True, exist_ok=True)

    bucket_run_dirs: list[Path] = []
    for H, W in DCW_ASPECT_BUCKETS:
        bucket_label = f"{label}-{H}x{W}"
        exclude_args: list[str] = []
        if not allow_repeats:
            used, n_runs = _scan_used_stems(
                out_root_path, H, W, float(baseline_lambda)
            )
            exclude_path = exclude_dir / f"{H}x{W}.txt"
            exclude_path.write_text(
                "\n".join(
                    [
                        f"# auto-generated by make dcw at bucket {H}x{W}, "
                        f"baseline_lambda={baseline_lambda}",
                        f"# {len(used)} prior stems across {n_runs} run(s) "
                        f"under {out_root}/*/manifest.json",
                        *sorted(used),
                    ]
                )
                + "\n"
            )
            exclude_args = ["--exclude_stems", str(exclude_path)]
            print(
                f"\n=== DCW sample: bucket {H}x{W} ({n_images} imgs × {n_seeds} seeds, "
                f"shuffle_seed={shuffle_seed}, baseline_lambda={baseline_lambda}) ===\n"
                f"    excluding {len(used)} stems from {n_runs} prior matching run(s); "
                f"pass --allow_repeats to disable."
            )
        else:
            print(
                f"\n=== DCW sample: bucket {H}x{W} ({n_images} imgs × {n_seeds} seeds, "
                f"shuffle_seed={shuffle_seed}, baseline_lambda={baseline_lambda}) ===\n"
                f"    --allow_repeats: not deduping against prior runs."
            )
        run(
            [
                sys.executable,
                "scripts/dcw/measure_bias.py",
                "--image_h",
                str(H),
                "--image_w",
                str(W),
                "--n_images",
                n_images,
                "--n_seeds",
                n_seeds,
                "--shuffle_seed",
                shuffle_seed,
                "--baseline_lambda",
                baseline_lambda,
                "--dump_per_sample_gaps",
                "--no-save_plot",
                "--label",
                bucket_label,
                "--out_root",
                out_root,
                *exclude_args,
                *extra,
            ]
        )
        bucket_dir = _latest_bucket_dir(out_root_path, bucket_label)
        if bucket_dir is not None:
            bucket_run_dirs.append(bucket_dir)

    _emit_aggregate_plot(out_root_path, bucket_run_dirs, label)

    print("\n=== DCW: training fusion head on pooled trajectories ===")
    run(
        [
            sys.executable,
            "scripts/dcw/train_fusion_head.py",
            "--label",
            label,
        ]
    )
    print(
        "\nDone. Run `make test-dcw-v4` to inference with the fresh artifact "
        "(auto-resolves the latest fusion_head.safetensors)."
    )


def cmd_dcw_train(extra):
    """Train-only on existing pool (no sampling, ~30s)."""
    run([sys.executable, "scripts/dcw/train_fusion_head.py", *extra])

"""Output helpers: CSV writers, plot, summary printer, accumulator."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from scripts.dcw.haar import BANDS

log = logging.getLogger("dcw-bench")


def _accumulate_row(
    accum: dict,
    name: str,
    v_fwd: np.ndarray,
    fwd_bands: dict[str, np.ndarray],
    rev_norms: np.ndarray,
    rev_bands: dict[str, np.ndarray],
    per_sample_bands: dict[str, np.ndarray] | None,
    per_sample_v_rev_bands: dict[str, np.ndarray] | None,
    per_sample_stems: list[str] | None,
    img_idx: int,
    seed_idx: int,
    n_seeds: int,
    stem: str,
    fei_low: np.ndarray | None = None,
    per_sample_fei_low: np.ndarray | None = None,
) -> None:
    """Fold one (img, seed, config) trajectory into the accumulator.

    ``fei_low`` (shape ``(n_steps,)``, ∈[0,1] simplex) is the 2-band FEI
    low-band energy captured per-step on the latent entering the step.
    Mirrors the per-sample band slot — only stashed for the ``baseline``
    config and only when ``--dump_per_sample_gaps`` allocated
    ``per_sample_fei_low``. Optional so legacy callers stay valid.
    """
    gap = rev_norms - v_fwd
    accum[name]["v_fwd"] += v_fwd
    accum[name]["v_rev"] += rev_norms
    accum[name]["gap"] += gap
    accum[name]["gap_sq"] += gap**2
    for b in BANDS:
        gap_b = rev_bands[b] - fwd_bands[b]
        accum[name]["v_fwd_bands"][b] += fwd_bands[b]
        accum[name]["v_rev_bands"][b] += rev_bands[b]
        accum[name]["gap_bands"][b] += gap_b
        if name == "baseline" and per_sample_bands is not None:
            row = img_idx * n_seeds + seed_idx
            per_sample_bands[b][row] = gap_b
            per_sample_v_rev_bands[b][row] = rev_bands[b]
            per_sample_stems[row] = stem
    if (
        name == "baseline"
        and per_sample_fei_low is not None
        and fei_low is not None
    ):
        row = img_idx * n_seeds + seed_idx
        per_sample_fei_low[row] = fei_low
    accum[name]["n"] += 1


def write_per_step_csv(
    out_dir: Path, accum: dict, sigmas: torch.Tensor, n_steps: int
) -> Path:
    csv_path = out_dir / "per_step.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        headers = ["step", "sigma_i"]
        for name in accum:
            headers += [
                f"{name}_v_fwd",
                f"{name}_v_rev",
                f"{name}_gap",
                f"{name}_gap_std",
            ]
        w.writerow(headers)
        for i in range(n_steps):
            row: list = [i, float(sigmas[i])]
            for name in accum:
                row += [
                    accum[name]["v_fwd"][i],
                    accum[name]["v_rev"][i],
                    accum[name]["gap"][i],
                    accum[name]["gap_std"][i],
                ]
            w.writerow(row)
    return csv_path


def write_per_band_csv(
    out_dir: Path, accum: dict, sigmas: torch.Tensor, n_steps: int
) -> Path:
    csv_path = out_dir / "per_step_bands.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        headers = ["step", "sigma_i"]
        for name in accum:
            for b in BANDS:
                headers += [
                    f"{name}_v_fwd_{b}",
                    f"{name}_v_rev_{b}",
                    f"{name}_gap_{b}",
                ]
        w.writerow(headers)
        for i in range(n_steps):
            row: list = [i, float(sigmas[i])]
            for name in accum:
                for b in BANDS:
                    row += [
                        accum[name]["v_fwd_bands"][b][i],
                        accum[name]["v_rev_bands"][b][i],
                        accum[name]["gap_bands"][b][i],
                    ]
            w.writerow(row)
    return csv_path


def make_plot(out_dir: Path, accum: dict, n_steps: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharex=True)
    base = accum["baseline"]
    xs = range(n_steps)

    axes[0].plot(xs, base["v_fwd"], label="forward ‖v(x_t, t)‖", color="#2a9d8f")
    axes[0].plot(xs, base["v_rev"], label="reverse ‖v(x̂_t, t)‖", color="#e76f51")
    axes[0].set_title("Baseline forward vs reverse velocity (Fig 1c)")
    axes[0].set_xlabel("step i")
    axes[0].set_ylabel("‖v‖₂")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name in accum:
        axes[1].plot(xs, accum[name]["gap"], label=name, alpha=0.85)
    axes[1].fill_between(
        xs,
        base["gap"] - base["gap_std"],
        base["gap"] + base["gap_std"],
        color="#888888",
        alpha=0.20,
        label="baseline ±1σ across (img×seed)",
    )
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("gap(i) = ‖v_rev‖ − ‖v_fwd‖  (closer to 0 = better)")
    axes[1].set_xlabel("step i")
    axes[1].set_ylabel("gap")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    band_colors = {"LL": "#264653", "LH": "#2a9d8f", "HL": "#e9c46a", "HH": "#e76f51"}
    for b in BANDS:
        axes[2].plot(xs, base["gap_bands"][b], label=b, color=band_colors[b], alpha=0.9)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("Baseline gap by Haar subband")
    axes[2].set_xlabel("step i")
    axes[2].set_ylabel("gap (band)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / "gap_curves.png"
    fig.savefig(png_path, dpi=130)
    log.info(f"plot → {png_path}")
    return True


def _empty_run_acc(steps: int) -> dict:
    return dict(
        v_fwd=np.zeros(steps),
        v_rev=np.zeros(steps),
        gap=np.zeros(steps),
        gap_sq=np.zeros(steps),
        v_fwd_bands={b: np.zeros(steps) for b in BANDS},
        v_rev_bands={b: np.zeros(steps) for b in BANDS},
        gap_bands={b: np.zeros(steps) for b in BANDS},
        n=0,
    )


def _finalize_run_acc(acc: dict) -> dict:
    n = acc["n"]
    if n == 0:
        return acc
    for k in ("v_fwd", "v_rev", "gap", "gap_sq"):
        acc[k] = acc[k] / n
    acc["gap_std"] = np.sqrt(np.maximum(acc["gap_sq"] - acc["gap"] ** 2, 0.0))
    for k in ("v_fwd_bands", "v_rev_bands", "gap_bands"):
        for b in BANDS:
            acc[k][b] = acc[k][b] / n
    return acc


def _bucket_key_for(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            bucket = data.get("bucket")
            if bucket and len(bucket) == 2:
                return f"{int(bucket[0])}x{int(bucket[1])}"
        except (OSError, json.JSONDecodeError):
            pass
    # Pre-manifest runs: pull from result.json args (image_h/image_w).
    result_path = run_dir / "result.json"
    if result_path.exists():
        try:
            args = (json.loads(result_path.read_text()).get("args") or {})
            h, w = args.get("image_h"), args.get("image_w")
            if h is not None and w is not None:
                return f"{int(h)}x{int(w)}"
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
    return "unknown"


def aggregate_run_dirs(
    run_dirs: list[Path],
) -> tuple[dict, dict[str, dict], int]:
    """Pool baseline per_step + per_step_bands data across multiple runs.

    Returns ``(pooled, per_bucket, n_steps)``:

    * ``pooled``: dict shaped like the in-memory accumulator that
      ``make_plot`` consumes (single key 'baseline'), weighted by each
      run's ``n_samples × n_seeds``.
    * ``per_bucket``: ``{"HxW": accum_dict}`` — same shape as the
      'baseline' value of ``pooled``, one entry per bucket parsed from
      manifest.json. Lets ``make_aggregate_plot`` overlay one line per
      bucket on top of the pooled summary.
    * ``n_steps``: shared step count.

    Reconstructs gap_std from the per-bucket mean+std via E[g²] =
    std² + mean². Skips dirs missing per_step.csv / per_step_bands.csv /
    result.json, or whose step counts disagree with the first valid run.
    """
    n_steps = 0
    pooled: dict | None = None
    per_bucket: dict[str, dict] = {}

    for run_dir in run_dirs:
        per_step_csv = run_dir / "per_step.csv"
        per_band_csv = run_dir / "per_step_bands.csv"
        result_json = run_dir / "result.json"
        if not (per_step_csv.exists() and per_band_csv.exists() and result_json.exists()):
            log.warning(f"aggregate: missing artifacts in {run_dir.name}; skipping")
            continue
        try:
            meta = json.loads(result_json.read_text())
        except (OSError, json.JSONDecodeError):
            log.warning(f"aggregate: unreadable result.json in {run_dir.name}; skipping")
            continue
        m = meta.get("metrics", {}) or {}
        n_b = int(m.get("n_samples", 0)) * int(m.get("n_seeds", 0))
        if n_b <= 0:
            continue

        with per_step_csv.open() as f:
            step_rows = list(csv.DictReader(f))
        with per_band_csv.open() as f:
            band_rows = list(csv.DictReader(f))
        steps = len(step_rows)
        if steps != len(band_rows) or steps == 0:
            log.warning(f"aggregate: row mismatch in {run_dir.name}; skipping")
            continue
        if pooled is None:
            n_steps = steps
            pooled = _empty_run_acc(steps)
        elif steps != n_steps:
            log.warning(
                f"aggregate: step count mismatch ({steps} vs {n_steps}) "
                f"for {run_dir.name}; skipping"
            )
            continue

        bucket_key = _bucket_key_for(run_dir)
        bucket_acc = per_bucket.setdefault(bucket_key, _empty_run_acc(n_steps))

        for i, row in enumerate(step_rows):
            gap_mean = float(row["baseline_gap"])
            gap_std = float(row["baseline_gap_std"])
            v_fwd = float(row["baseline_v_fwd"])
            v_rev = float(row["baseline_v_rev"])
            gap_sq_contrib = n_b * (gap_std * gap_std + gap_mean * gap_mean)
            for acc in (pooled, bucket_acc):
                acc["v_fwd"][i] += n_b * v_fwd
                acc["v_rev"][i] += n_b * v_rev
                acc["gap"][i] += n_b * gap_mean
                acc["gap_sq"][i] += gap_sq_contrib
        for i, row in enumerate(band_rows):
            for b in BANDS:
                v_fwd_b = float(row[f"baseline_v_fwd_{b}"])
                v_rev_b = float(row[f"baseline_v_rev_{b}"])
                gap_b = float(row[f"baseline_gap_{b}"])
                for acc in (pooled, bucket_acc):
                    acc["v_fwd_bands"][b][i] += n_b * v_fwd_b
                    acc["v_rev_bands"][b][i] += n_b * v_rev_b
                    acc["gap_bands"][b][i] += n_b * gap_b
        pooled["n"] += n_b
        bucket_acc["n"] += n_b

    if pooled is None or pooled["n"] == 0:
        return {}, {}, 0

    _finalize_run_acc(pooled)
    for acc in per_bucket.values():
        _finalize_run_acc(acc)
    return {"baseline": pooled}, per_bucket, n_steps


def make_aggregate_plot(
    out_dir: Path,
    accum: dict,
    per_bucket: dict[str, dict],
    n_steps: int,
) -> bool:
    """Pooled summary with per-bucket gap overlay.

    Layout: 1×3, same as ``make_plot``. Panels 1 (fwd/rev) and 3
    (per-Haar-band) stay pooled — overlaying 5 buckets there would
    swamp the curves. Panel 2 (gap) shows one colored line per bucket
    plus the pooled mean in bold black, with a bucket legend.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping aggregate plot")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharex=True)
    base = accum["baseline"]
    xs = range(n_steps)

    axes[0].plot(xs, base["v_fwd"], label="forward ‖v(x_t, t)‖", color="#2a9d8f")
    axes[0].plot(xs, base["v_rev"], label="reverse ‖v(x̂_t, t)‖", color="#e76f51")
    axes[0].set_title("Pooled forward vs reverse velocity")
    axes[0].set_xlabel("step i")
    axes[0].set_ylabel("‖v‖₂")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    bucket_keys = sorted(per_bucket.keys())
    cmap = plt.get_cmap("viridis")
    colors = (
        [cmap(i / max(1, len(bucket_keys) - 1)) for i in range(len(bucket_keys))]
        if bucket_keys
        else []
    )
    for key, color in zip(bucket_keys, colors):
        b_acc = per_bucket[key]
        axes[1].plot(
            xs,
            b_acc["gap"],
            label=f"{key} (n={b_acc['n']})",
            color=color,
            alpha=0.85,
            lw=1.2,
        )
    axes[1].plot(
        xs,
        base["gap"],
        label=f"pooled (n={base['n']})",
        color="black",
        lw=2.0,
    )
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("gap(i) per bucket  +  pooled mean")
    axes[1].set_xlabel("step i")
    axes[1].set_ylabel("gap = ‖v_rev‖ − ‖v_fwd‖")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    band_colors = {"LL": "#264653", "LH": "#2a9d8f", "HL": "#e9c46a", "HH": "#e76f51"}
    for b in BANDS:
        axes[2].plot(xs, base["gap_bands"][b], label=b, color=band_colors[b], alpha=0.9)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("Pooled gap by Haar subband")
    axes[2].set_xlabel("step i")
    axes[2].set_ylabel("gap (band)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / "gap_curves.png"
    fig.savefig(png_path, dpi=130)
    log.info(f"aggregate plot → {png_path}")
    return True


def print_summary(accum: dict, ranked: list, dcw_sweep: bool) -> None:
    base = accum["baseline"]
    print("\n=== SNR-t bias measurement ===")
    print(
        f"baseline integrated signed gap: {base['gap'].sum():+.3f}  "
        f"(Anima flow-matching: gap is typically negative — forward > reverse — "
        f"opposite of the DDPM noise-pred sign in the paper)"
    )
    print(
        f"baseline gap std across {base['n']} (img×seed) trajectories: "
        f"mean σ_step = {base['gap_std'].mean():.3f}, "
        f"max σ_step = {base['gap_std'].max():.3f}"
    )

    print("\nbaseline integrated signed gap by Haar subband:")
    print(f"  {'band':<4s}  {'signed':>9s}  {'|gap|':>8s}")
    for b in BANDS:
        g = float(base["gap_bands"][b].sum())
        a = float(np.abs(base["gap_bands"][b]).sum())
        print(f"  {b:<4s}  {g:+9.3f}  {a:8.3f}")

    if dcw_sweep:
        print("\nconfigs ranked by integrated |gap| (smaller = closer alignment):")
        for rank, (name, a, s) in enumerate(ranked, 1):
            print(f"  {rank:>2}. {name:<24s}  |gap|={a:7.3f}  signed={s:+7.3f}")

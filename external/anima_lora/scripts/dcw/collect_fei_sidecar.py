"""Replay reverse trajectories on existing DCW bench runs to recover FEI.

Bench runs that predate scripts/dcw/trajectory.py's FEI capture don't carry
per-step ``fei_low`` in ``gaps_per_sample.npz``. Re-running ``make dcw`` from
scratch is wasteful since the forward branch (~half the wall time) is already
captured in ``per_step_bands.csv`` and the rest of the bench artifacts.

This script reads ``result.json`` + ``manifest.json`` from each existing
run dir and **replays only the reverse trajectory** at the same seeds, the
same baseline λ, and the same DiT / TE / mod-guidance config — capturing
``fei_low`` and nothing else. Output: ``<run_dir>/fei_low.npz`` aligned
row-by-row with ``<run_dir>/gaps_per_sample.npz``.

Deterministic seeds + the same encoded (x_0, embed) cache + the same σ
schedule give bit-identical ``x_hat`` trajectories ⇒ bit-identical FEI.

``scripts/dcw/fusion_data.py::load_bench_runs`` prefers the sidecar over
the main npz's ``fei_low`` key (which is absent on legacy pools anyway),
so once this script runs the trainer's ``--fei_obs != off`` modes pick up
the rev-replayed pool automatically — no extra plumbing.

Usage::

    # Replay every run under output/dcw/ that lacks fei_low.npz
    python -m scripts.dcw.collect_fei_sidecar --results_root output/dcw/

    # Target a specific run dir (skip the walk)
    python -m scripts.dcw.collect_fei_sidecar --run_dir output/dcw/20260513-2155-foo

"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.anima import weights as anima_utils  # noqa: E402
from library.inference import sampling as inference_utils  # noqa: E402
from library.inference.adapters import clear_hydra_sigma  # noqa: E402
from library.inference.text import MAX_CROSSATTN_TOKENS  # noqa: E402
from scripts.dcw.cache import load_cached, pick_cached_samples  # noqa: E402
from scripts.dcw.trajectory import (  # noqa: E402
    encode_uncond_embed,
    run_reverse_batched,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dcw-fei-sidecar")


def _result_args(run_dir: Path) -> dict | None:
    rj = run_dir / "result.json"
    if not rj.exists():
        return None
    try:
        return json.loads(rj.read_text()).get("args", {})
    except json.JSONDecodeError:
        log.warning(f"{run_dir.name}: result.json is malformed; skipping")
        return None


def _manifest(run_dir: Path) -> dict | None:
    mf = run_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text())
    except json.JSONDecodeError:
        log.warning(f"{run_dir.name}: manifest.json is malformed; skipping")
        return None


def _gather_run_dirs(
    results_roots: list[Path], explicit: list[Path] | None
) -> list[Path]:
    if explicit:
        return [p.resolve() for p in explicit]
    out: list[Path] = []
    for root in results_roots:
        if not root.exists():
            continue
        out.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return out


def _key_for_text_state(args: dict) -> tuple:
    """Identity of the (text-encoder, uncond, mod-guidance) setup.

    Two runs sharing this key can reuse the same uncond crossattn embed
    and the same mod-guidance setup — no need to reload the text encoder
    between them.
    """
    return (
        float(args.get("guidance_scale", 1.0)),
        str(args.get("negative_prompt", "")),
        bool(args.get("pooled_text_proj")),
        str(args.get("pooled_text_proj") or ""),
        float(args.get("mod_w", 0.0)),
        str(args.get("mod_pos_prompt", "")),
        str(args.get("mod_neg_prompt", "")),
        int(args.get("mod_start_layer", 0)),
        int(args.get("mod_end_layer", 0)),
        int(args.get("mod_taper", 0)),
        float(args.get("mod_taper_scale", 0.0)),
        float(args.get("mod_final_w", 0.0)),
    )


class _ArgsNS:
    """Lightweight argparse.Namespace-like wrapper for setup_mod_guidance."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _setup_text_state(
    anima,
    args: dict,
    *,
    device: torch.device,
    text_encoder_path: str,
) -> Optional[torch.Tensor]:
    """Prime tokenizer/encoder strategies, encode uncond, set up mod-guidance.

    Returns the uncond crossattn embed (or None when CFG=1 and mod is off).
    Mirrors measure_bias.py's transient TE block. Frees the TE before
    returning — replay shouldn't keep it resident.
    """
    needs_te = bool(args.get("pooled_text_proj")) or float(
        args.get("guidance_scale", 1.0)
    ) != 1.0
    if not needs_te:
        anima.reset_mod_guidance()
        return None

    from library.anima import strategy as strategy_anima, text_strategies
    from library.inference.models import load_text_encoder

    text_strategies.TokenizeStrategy.set_strategy(
        strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=text_encoder_path,
            t5_tokenizer_path=None,
            qwen3_max_length=MAX_CROSSATTN_TOKENS,
            t5_max_length=MAX_CROSSATTN_TOKENS,
        )
    )
    text_strategies.TextEncodingStrategy.set_strategy(
        strategy_anima.AnimaTextEncodingStrategy()
    )

    # load_text_encoder also reads args.lora_weight and args.lora_multiplier
    # to decide if any LoRA changes the TE-side strategy (none here — replay
    # runs on the base DiT, just like the original bench). Pass explicit
    # None / [1.0] defaults so the attribute lookups don't AttributeError.
    te_args = _ArgsNS(
        text_encoder=text_encoder_path,
        attn_mode=args.get("attn_mode", "flash"),
        lora_weight=None,
        lora_multiplier=[1.0],
    )
    text_encoder = load_text_encoder(te_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()

    embed_uncond = None
    if float(args.get("guidance_scale", 1.0)) != 1.0:
        embed_uncond = encode_uncond_embed(
            anima,
            text_encoder,
            str(args.get("negative_prompt", "")),
            device,
        )

    if args.get("pooled_text_proj"):
        from library.inference.mod_guidance import setup_mod_guidance

        mod_args = _ArgsNS(
            pooled_text_proj=args["pooled_text_proj"],
            mod_w=float(args.get("mod_w", 3.0)),
            mod_pos_prompt=args.get("mod_pos_prompt", ""),
            mod_neg_prompt=args.get("mod_neg_prompt", ""),
            mod_start_layer=int(args.get("mod_start_layer", 8)),
            mod_end_layer=int(args.get("mod_end_layer", 27)),
            mod_taper=int(args.get("mod_taper", 0)),
            mod_taper_scale=float(args.get("mod_taper_scale", 0.25)),
            mod_final_w=float(args.get("mod_final_w", 0.0)),
            text_encoder=text_encoder_path,
        )
        setup_mod_guidance(mod_args, anima, device, shared_models={"text_encoder": text_encoder})
    else:
        anima.reset_mod_guidance()

    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embed_uncond


def _replay_one_run(
    anima,
    run_dir: Path,
    *,
    args: dict,
    manifest: dict,
    embed_uncond: Optional[torch.Tensor],
    device: torch.device,
    dataset_dir: Path,
) -> bool:
    """Replay one run's reverse trajectories; write fei_low.npz. Return True on success."""
    # Required scalars
    image_h = args.get("image_h")
    image_w = args.get("image_w")
    if image_h is None or image_w is None:
        log.warning(f"{run_dir.name}: missing image_h/w in result.json; skipping")
        return False
    n_seeds = int(args.get("n_seeds", 1))
    n_steps = int(args.get("infer_steps", 28))
    flow_shift = float(args.get("flow_shift", 1.0))
    cfg_scale = float(args.get("guidance_scale", 1.0))
    baseline_lambda = float(args.get("baseline_lambda", 0.0))
    text_variant = int(args.get("text_variant", 0))

    stems_in_order: list[str] = []
    seen = set()
    for entry in manifest.get("pairs", []):
        s = entry.get("stem")
        if s and s not in seen:
            seen.add(s)
            stems_in_order.append(s)
    if not stems_in_order:
        log.warning(f"{run_dir.name}: manifest has no pairs; skipping")
        return False

    n_imgs = len(stems_in_order)
    n_traj = n_imgs * n_seeds
    fei_low_arr = np.zeros((n_traj, n_steps), dtype=np.float64)
    seeds_arr = np.zeros(n_traj, dtype=np.int64)

    _, sigmas_t = inference_utils.get_timesteps_sigmas(n_steps, flow_shift, device)
    sigmas = sigmas_t.cpu()

    seed_base = int(args.get("seed_base", manifest.get("seed_base", 0)))

    # Cache filenames are zero-padded ({H:04d}x{W:04d}_anima.npz), so we
    # can't just format from (image_h, image_w). Reuse pick_cached_samples's
    # regex-based discovery and build a stem→(npz, te) lookup over the full
    # bucket-matched candidate pool, then index by manifest stem.
    bucket_pool = pick_cached_samples(
        dataset_dir, n=10**9, image_h=image_h, image_w=image_w
    )
    cache_by_stem = {stem: (npz, te) for stem, npz, te in bucket_pool}

    pbar = tqdm(total=n_imgs, desc=f"{run_dir.name} rev-replay")
    for img_idx, stem in enumerate(stems_in_order):
        if stem not in cache_by_stem:
            log.warning(
                f"{run_dir.name}: missing cache for stem {stem!r} "
                f"(no {stem}_<H>x<W>_anima.npz at {image_h}x{image_w}); skipping run"
            )
            pbar.close()
            return False
        npz, te = cache_by_stem[stem]
        x_0, embed = load_cached(npz, te, text_variant, device)
        seeds = [seed_base + 1000 * img_idx + j for j in range(n_seeds)]
        rev_out = run_reverse_batched(
            anima,
            x_0,
            embed,
            sigmas,
            noise_seeds=seeds,
            dcw_lams=[baseline_lambda] * n_seeds,
            device=device,
            embed_uncond=embed_uncond,
            cfg_scale=cfg_scale,
            return_final=False,
        )
        # rev_out is a list of (norms, bands, fei_low) — only fei_low is needed.
        for seed_idx, (_, _, fei_low) in enumerate(rev_out):
            row = img_idx * n_seeds + seed_idx
            fei_low_arr[row] = fei_low[:n_steps]
            seeds_arr[row] = seeds[seed_idx]
        pbar.update(1)
        pbar.set_postfix_str(stem)
    pbar.close()

    clear_hydra_sigma(anima)

    out_path = run_dir / "fei_low.npz"
    np.savez(
        out_path,
        fei_low=fei_low_arr,
        seeds=seeds_arr,
        stems=np.array(
            [stems_in_order[r // n_seeds] for r in range(n_traj)], dtype=object
        ),
        sigmas=sigmas.numpy()[:n_steps],
    )
    log.info(f"{run_dir.name}: wrote {out_path.name} ({n_traj} rows × {n_steps} steps)")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_root",
        type=Path,
        nargs="*",
        default=[ROOT / "output" / "dcw"],
        help="Directories holding bench-script run dirs (walk all subdirs).",
    )
    p.add_argument(
        "--run_dir",
        type=Path,
        nargs="*",
        default=None,
        help="Target specific run dirs (overrides --results_root walk).",
    )
    p.add_argument(
        "--dit",
        type=str,
        default="models/diffusion_models/anima-base-v1.0.safetensors",
    )
    p.add_argument(
        "--text_encoder",
        type=str,
        default="models/text_encoders/qwen_3_06b_base.safetensors",
    )
    p.add_argument(
        "--dataset_dir",
        type=Path,
        default=ROOT / "post_image_dataset" / "lora",
        help="Cached *_anima.npz + *_anima_te.safetensors location.",
    )
    p.add_argument("--attn_mode", type=str, default="flash")
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the DiT once after the first run's text-state "
        "setup. Same logic as measure_bias.py — each unique latent (H, W) "
        "pays a one-time warm-up; dynamo auto-flips to dynamic shapes after "
        "the second distinct bucket. set_hydra_sigma routes through "
        "_orig_mod so router-state writes still land. Default on; pass "
        "--no-compile if a run has trouble compiling.",
    )
    p.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs that already have fei_low.npz (default on). "
        "Pass --no-skip_existing to overwrite.",
    )
    p.add_argument(
        "--skip_main_npz_fei",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs whose gaps_per_sample.npz already carries the "
        "fei_low key (post-capture runs that wouldn't gain from a sidecar). "
        "Default on.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    candidates = _gather_run_dirs(args.results_root, args.run_dir)
    if not candidates:
        sys.exit("no run dirs found under the given roots")

    # Pre-filter: drop runs missing artifacts or already done.
    eligible: list[tuple[Path, dict, dict]] = []
    for run_dir in candidates:
        if args.skip_existing and (run_dir / "fei_low.npz").exists():
            continue
        rj = _result_args(run_dir)
        mf = _manifest(run_dir)
        if rj is None or mf is None:
            continue
        # Sidecar is meaningless without an aligned gaps_per_sample.npz to
        # join against in the trainer — the sidecar's whole point is to fill
        # in the FEI column for an existing per-sample dump.
        if not (run_dir / "gaps_per_sample.npz").exists():
            log.info(f"{run_dir.name}: no gaps_per_sample.npz; skipping")
            continue
        if args.skip_main_npz_fei:
            try:
                with np.load(run_dir / "gaps_per_sample.npz", allow_pickle=True) as z:
                    if "fei_low" in z.files:
                        continue
            except Exception:
                pass
        eligible.append((run_dir, rj, mf))

    if not eligible:
        log.info("nothing to do — every candidate is already done or ineligible")
        return

    log.info(f"replaying {len(eligible)} run dir(s)")

    log.info("loading DiT…")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    # Defer .to(device, dtype) + .eval() until after the first run's
    # pooled_text_proj (if any) is loaded — matches measure_bias.py.
    anima.to(device, dtype=dtype)
    anima.eval().requires_grad_(False)

    cur_key: tuple | None = None
    embed_uncond: Optional[torch.Tensor] = None
    compiled = False

    for run_dir, rj_args, manifest in eligible:
        key = _key_for_text_state(rj_args)
        if key != cur_key:
            # Reload pooled_text_proj if this group needs mod-guidance —
            # has to happen before .to() per measure_bias's note, but the
            # model is already on-device. anima_utils.load_pooled_text_proj
            # handles the meta-tensor → real-tensor swap safely after .to().
            if rj_args.get("pooled_text_proj"):
                anima_utils.load_pooled_text_proj(
                    anima, rj_args["pooled_text_proj"], "cpu"
                )
            embed_uncond = _setup_text_state(
                anima,
                rj_args,
                device=device,
                text_encoder_path=args.text_encoder,
            )
            cur_key = key

        # Compile after the first run's setup (mod-guidance attach + uncond
        # encode) so the OptimizedModule wraps the fully-prepared graph,
        # mirroring measure_bias.py's "compile last" ordering. Skipping if
        # already compiled — torch.compile is idempotent but the wrapping
        # cost is non-trivial.
        if args.compile and not compiled:
            log.info("torch.compile(DiT)…")
            anima = torch.compile(anima)
            compiled = True

        _replay_one_run(
            anima,
            run_dir,
            args=rj_args,
            manifest=manifest,
            embed_uncond=embed_uncond,
            device=device,
            dataset_dir=args.dataset_dir,
        )


if __name__ == "__main__":
    main()

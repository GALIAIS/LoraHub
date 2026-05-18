"""Inference entry-points for shipped methods (test / test-* commands).

All variants share ``INFERENCE_BASE`` from ``_common`` and add method-specific
flags. Experimental inference commands (exp-test-postfix*, exp-test-prefix,
exp-test-ref, exp-test-ip, exp-test-easycontrol) live in
``scripts/experimental_tasks/inference.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ._common import (
    INFERENCE_BASE,
    ROOT,
    latest_hydra,
    latest_lora,
    latest_output,
    run,
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _mod_flags() -> list[str]:
    """Resolve latest distilled pooled_text_proj for ``MOD=1``."""
    return ["--pooled_text_proj", str(latest_output("pooled_text_proj"))]


def _base_test_args(*, lora_default: bool = True) -> list[str]:
    """Build the shared ``inference.py`` argv prefix used by every ``test*`` command.

    Honors three env levers so they compose uniformly across ``test``,
    ``test-smc-cfg``, ``test-dcw``, ``test-dcw-v4``:

    - ``NOLORA=1`` skips ``--lora_weight`` (bare DiT). When unset, ``lora_default``
      decides whether the caller wants a LoRA by default — ``test-dcw-v4`` opts
      out (DCW v4 is meant to ride on the bare DiT unless the user adds one).
    - ``SPECTRUM=1`` appends Spectrum flags.
    - ``MOD=1`` appends ``--pooled_text_proj <latest>``.
    """
    args = list(INFERENCE_BASE)
    nolora_env = os.environ.get("NOLORA")
    if nolora_env is None:
        include_lora = lora_default
    else:
        include_lora = not _env_truthy("NOLORA")
    if include_lora:
        args += ["--lora_weight", str(latest_lora())]
    if _env_truthy("SPECTRUM"):
        args += _spectrum_flags()
    if _env_truthy("MOD"):
        args += _mod_flags()
    return args


def _spectrum_flags(stop_caching_step: int = 27) -> list[str]:
    return [
        "--spectrum",
        "--spectrum_window_size",
        "2.0",
        "--spectrum_flex_window",
        "0.25",
        "--spectrum_warmup",
        "7",
        "--spectrum_w",
        "0.3",
        "--spectrum_m",
        "3",
        "--spectrum_lam",
        "0.1",
        "--spectrum_stop_caching_step",
        str(stop_caching_step),
        "--spectrum_calibration",
        "0.0",
    ]


def cmd_test(extra):
    """Inference with the latest LoRA. See ``_base_test_args`` for env levers."""
    run([*_base_test_args(), *extra])


def cmd_test_hydra(extra):
    # Uses the moe sibling (router-live); static-merge is auto-skipped in
    # library/inference_pipeline.py:_is_hydra_moe detection.
    run([*INFERENCE_BASE, "--lora_weight", str(latest_hydra()), *extra])


def cmd_test_merge(extra):
    """Inference with a baked (merged) DiT from MODEL_DIR (default 'output_temp').

    MODEL_DIR accepts either a directory (picks the latest
    ``*_merged.safetensors`` inside) or a direct ``.safetensors`` path. The
    merged file is a standalone DiT (LoRA folded in), so no ``--lora_weight``
    is passed. The trailing ``--dit`` overrides the base one in
    ``INFERENCE_BASE`` (argparse keeps the last value).
    """
    target = Path(os.environ.get("MODEL_DIR", "output_temp"))
    if not target.is_absolute():
        target = ROOT / target
    if target.is_file():
        chosen = target
    elif target.is_dir():
        candidates = sorted(
            target.glob("*_merged.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"No '*_merged.safetensors' files found in {target}", file=sys.stderr)
            sys.exit(1)
        chosen = candidates[0]
    else:
        print(f"MODEL_DIR path not found: {target}", file=sys.stderr)
        sys.exit(1)
    run([*INFERENCE_BASE, "--dit", str(chosen), *extra])


def cmd_test_dcw(extra):
    """Inference with latest LoRA + DCW post-step correction.

    Defaults bake in λ=0.01 + one_minus_sigma schedule (see
    bench/dcw/findings.md). Override via --dcw_lambda / --dcw_schedule in extra.
    Honors SPECTRUM / MOD / NOLORA env levers (see ``_base_test_args``).
    """
    run([*_base_test_args(), "--dcw", "--dcw_lambda", "0.01", *extra])


def cmd_test_smc_cfg(extra):
    """Inference with latest LoRA + SMC-CFG (arXiv:2603.03281).

    Production defaults (λ=5, α=0.2). Override via --smc_cfg_lambda /
    --smc_cfg_alpha in extra. Honors SPECTRUM / MOD / NOLORA env levers
    (see ``_base_test_args``); composes with --dcw via extra.
    """
    run([*_base_test_args(), "--smc_cfg", *extra])


def _latest_fusion_head() -> str:
    """Resolve the most recent fusion_head.safetensors under any DCW root.

    Scans output/dcw/ (new `make dcw` output), post_image_dataset/dcw/
    (legacy), and bench/dcw/results/ (legacy). Newest mtime wins.
    """
    from pathlib import Path

    roots = [
        Path("output/dcw"),
        Path("post_image_dataset/dcw"),
        Path("bench/dcw/results"),
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("*/fusion_head.safetensors"))
    if not candidates:
        raise SystemExit(
            "no fusion_head.safetensors found under output/dcw/, "
            "post_image_dataset/dcw/, or bench/dcw/results/ — "
            "run `make dcw-train` first"
        )
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def cmd_test_dcw_v4(extra):
    """Inference with DCW learnable calibrator (no LoRA by default).

    Auto-resolves the most recent fusion_head.safetensors. Pass
    --dcw_calibrator <path> (or legacy --dcw_v4 <path>) in extra to override.
    Honors SPECTRUM / MOD / NOLORA env levers (see ``_base_test_args``).
    Defaults to NOLORA semantics; set ``NOLORA=0`` to attach the latest LoRA,
    or pass ``--lora_weight <path>`` in extra to attach a specific one.
    """
    extra_has_calib = any(
        a == "--dcw_calibrator" or a == "--dcw_v4" for a in extra
    )
    calib_args = [] if extra_has_calib else ["--dcw_calibrator", _latest_fusion_head()]
    run([*_base_test_args(lora_default=False), *calib_args, *extra])


def cmd_test_spectrum_dcw(extra):
    """Spectrum + DCW composed. Equivalent to ``make test SPECTRUM=1 --dcw``."""
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_lora()),
            *_spectrum_flags(stop_caching_step=27),
            "--dcw",
            *extra,
        ]
    )


def cmd_test_dcw_v4_spectrum(extra):
    """Spectrum + DCW learnable calibrator composed.

    Spectrum knobs match ``cmd_test`` with stop_caching_step=27 to match
    DCW's 28-step contract, plus DCW calibrator (auto-resolves the most recent
    fusion_head.safetensors). Pass --dcw_calibrator <path> in extra to override.
    """
    extra_has_calib = any(
        a == "--dcw_calibrator" or a == "--dcw_v4" for a in extra
    )
    calib_args = [] if extra_has_calib else ["--dcw_calibrator", _latest_fusion_head()]
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_lora()),
            *_spectrum_flags(stop_caching_step=27),
            *calib_args,
            "--infer_steps",
            "28",
            *extra,
        ]
    )

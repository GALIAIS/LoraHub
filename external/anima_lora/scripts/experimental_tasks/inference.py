"""Experimental inference entry-points (exp-test-* commands).

Covers the unstable methods kept under ``make exp-*``:
postfix / postfix_exp / postfix_func, IP-Adapter, EasyControl. Reference-image
variants (exp-test-ip / exp-test-easycontrol) accept REF_IMAGE env or first
positional arg, copy the ref alongside the generated output.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

from scripts.tasks._common import (
    INFERENCE_BASE,
    ROOT,
    latest_output,
    run,
)

_REF_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _random_ref_image(directory: Path) -> str | None:
    if not directory.is_dir():
        return None
    pool = [p for p in directory.iterdir() if p.suffix.lower() in _REF_IMAGE_EXTS]
    if not pool:
        return None
    pick = random.choice(pool)
    print(f"  > Random ref: {pick}")
    return str(pick)


def cmd_test_postfix(extra):
    # exclude both _exp and _func so the vanilla postfix target doesn't grab them
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("anima_postfix*.safetensors")
            if "_exp" not in f.name and "_func" not in f.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        print(
            "No 'anima_postfix*.safetensors' files found in output/ckpt/",
            file=sys.stderr,
        )
        sys.exit(1)
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(outputs[0]),
            *extra,
        ]
    )


def cmd_test_postfix_exp(extra):
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(latest_output("anima_postfix_exp")),
            *extra,
        ]
    )


def cmd_test_postfix_func(extra):
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(latest_output("anima_postfix_func")),
            *extra,
        ]
    )


def cmd_test_turbo(extra):
    """Inference with the latest turbo student LoRA at 4 steps, cfg=1.0.

    CFG is baked into the student during distillation, so production inference
    runs cfg=1.0 (no double-CFG). Step count defaults to 4 — the value the
    student was distilled at — but extra args can override.
    """
    weight = latest_output("anima_turbo")
    base = list(INFERENCE_BASE)
    # Replace defaults so `--infer_steps`/`--guidance_scale` reflect the turbo
    # contract (4 steps, cfg=1.0). User extra args still win since they come last.
    base = _override_arg(base, "--sampler", "lcm")
    base = _override_arg(base, "--infer_steps", "4")
    base = _override_arg(base, "--guidance_scale", "1.0")
    run(
        [
            *base,
            "--lora_weight",
            str(weight),
            *extra,
        ]
    )


def _override_arg(argv: list[str], flag: str, value: str) -> list[str]:
    """Replace a ``--flag VALUE`` (or ``--flag V1 V2``) pair in argv with a
    fresh ``--flag value`` pair. Used to retarget INFERENCE_BASE defaults
    for the turbo contract (4 steps, cfg=1.0) without rewriting the whole list.
    """
    if flag not in argv:
        return argv + [flag, value]
    i = argv.index(flag)
    # Drop the flag and its single value; INFERENCE_BASE doesn't use multi-arg
    # flags for these two keys.
    return argv[:i] + [flag, value] + argv[i + 2 :]


def cmd_test_ip(extra):
    """Inference with latest IP-Adapter weight.

    Reference image is taken from REF_IMAGE env or the first positional arg.
    Falls back to a random image from ``post_image_dataset/resized/`` (the
    IP-Adapter source layout) when neither is supplied.
    PROMPT, NEG, IP_SCALE env vars override defaults. Saves to output/tests/ip/
    and copies the ref image alongside the generated output as ``<name>_ref.png``.

    Examples:
      python tasks.py exp-test-ip ref.png --prompt "a girl in a coffee shop"
      REF_IMAGE=ref.png IP_SCALE=0.8 python tasks.py exp-test-ip
      python tasks.py exp-test-ip                 # random ref from post_image_dataset/resized/
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-ip <ref_image> [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-ip [extra...]\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dir = ROOT / "output" / "tests" / "ip"
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        *INFERENCE_BASE,
        "--save_path",
        str(save_dir),
        "--ip_adapter_weight",
        str(latest_output("anima_ip_adapter")),
        "--ip_image",
        ref_image,
        "--ip_image_match_size",
    ]
    if scale := os.environ.get("IP_SCALE"):
        args += ["--ip_scale", scale]
    args += ["--prompt", os.environ.get("PROMPT") or "double peace, v v,"]
    if neg := os.environ.get("NEG"):
        args += ["--negative_prompt", neg]
    args += list(extra)
    run(args)

    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_ref.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        ref_dst = pngs[0].with_name(pngs[0].stem + "_ref.png")
        shutil.copy(ref_image, ref_dst)
        print(f"  > Ref pasted: {ref_dst}")


def cmd_test_directedit(extra):
    """DirectEdit on a random source image, seeded by wd-swinv2-tagger-v3.

    Pipeline:
      1. Pick source image (REF_IMAGE env, first positional arg, or random
         from ``post_image_dataset/resized/``).
      2. Run wd-swinv2-tagger-v3 on the source -> ``src_tags`` caption
         (downloaded on first use to ``models/captioners/wd-swinv2-tagger-v3/``).
      3. Build edit prompts:
            prompt_src = src_tags
            prompt_tar = src_tags + ", " + PROMPT
         (PROMPT env or ``--prompt`` extra arg supplies the edit instruction.
         Defaults to ``"double peace"``.)
      4. Call ``scripts/edit.py`` (DirectEdit invert + edit) using the same
         DiT/VAE/TE trio as the other inference targets.
      5. Save under ``output/tests/directedit/`` and copy the source image
         alongside as ``<name>_src.png``.

    Examples:
      make exp-test-directedit PROMPT='double peace'
      REF_IMAGE=foo.png make exp-test-directedit PROMPT='glasses'
      python tasks.py exp-test-directedit foo.png --prompt 'smile'
    """
    # 1. Resolve source image — same logic as cmd_test_ip / cmd_test_easycontrol.
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-directedit [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-directedit\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Pull the user-supplied edit instruction. PROMPT env wins; fall back
    #    to a ``--prompt`` flag in extra; final default = "double peace".
    edit_prompt = os.environ.get("PROMPT", "").strip()
    cleaned_extra: list[str] = []
    skip_next = False
    for j, tok in enumerate(extra):
        if skip_next:
            skip_next = False
            continue
        if tok == "--prompt" and j + 1 < len(extra):
            if not edit_prompt:
                edit_prompt = extra[j + 1]
            skip_next = True
            continue
        cleaned_extra.append(tok)
    extra = cleaned_extra
    if not edit_prompt:
        edit_prompt = "double peace, v v. She is showing double peace"

    # 3. Run Anima Tagger on the source.
    sys.path.insert(0, str(ROOT))
    from PIL import Image  # noqa: PLC0415

    anima_ckpt = ROOT / "models" / "captioners" / "anima-tagger-v1" / "model.safetensors"
    if not anima_ckpt.exists():
        raise SystemExit(
            f"Anima Tagger checkpoint missing at {anima_ckpt} — "
            "train via `python -m scripts.anima_tagger.cli`."
        )

    print(f"  > tagging source: {ref_image}")
    from library.captioning.anima_tagger import AnimaTagger  # noqa: PLC0415

    tagger = AnimaTagger(ckpt_dir=anima_ckpt.parent)

    src_caption = tagger.predict_caption(Image.open(ref_image))
    if not src_caption:
        print(
            "  ! tagger produced no tags above threshold; using empty source "
            "prompt — DirectEdit reconstruction will be weaker than usual.",
            file=sys.stderr,
        )
    print(f"  > src caption: {src_caption[:120]}{'...' if len(src_caption) > 120 else ''}")

    # 4. Save dir + edit.py invocation. Reuse INFERENCE_BASE for the model
    #    path trio (--dit / --text_encoder / --vae) so this stays in sync with
    #    the other test commands automatically.
    #
    #    Hand the edit instruction to edit.py via --edit_instruction so the
    #    dispatcher (Qwen3 last-token cosine + threshold/gap gate; see
    #    library/inference/edit_dispatcher.py and plan.md) runs in-process —
    #    REPLACE on confident matches, REMOVE on explicit `-X` / `no X`,
    #    APPEND otherwise. Running the dispatcher in this wrapper would load
    #    Qwen3 a second time; we'd rather edit.py do it once.
    save_dir = ROOT / "output" / "tests" / "directedit"
    save_dir.mkdir(parents=True, exist_ok=True)

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop "inference.py"
    leftover_base = list(base_iter)
    args = [py, "scripts/edit.py", *_filter_inference_base_for_edit(leftover_base)]
    args += [
        "--image", str(ref_image),
        "--prompt_src", src_caption,
        "--edit_instruction", edit_prompt,
        "--save_path", str(save_dir),
    ]
    args += list(extra)
    run(args)

    # 5. Copy the source alongside the edited output for side-by-side review.
    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_src.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
        shutil.copy(ref_image, src_dst)
        print(f"  > Source pasted: {src_dst}")


def cmd_test_directedit_dry(extra):
    """DirectEdit functional sanity check using preprocessed cross-emb variants.

    Bypasses the tagger and the text encoder. Auto-resolves the source image's
    `_anima_te.safetensors` cache (the file `cache_text_embeddings.py` writes
    — same format the trainer consumes) and runs one invert + edit pass per
    stored variant with ψ_tar == ψ_src. With `--caption_shuffle_variants N`
    caches, this sweeps v0 (pristine) + v1..v{N-1} (tag-shuffled). Each pass
    should reconstruct the source; divergence flags numeric drift in
    invert/edit_forward against that variant's cross-emb representation.

    Examples:
      make exp-test-directedit-dry
      REF_IMAGE=foo.png make exp-test-directedit-dry
      python tasks.py exp-test-directedit-dry foo.png --seed 7
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-directedit-dry [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-directedit-dry\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-resolve the matching TE cache file. Try the standard cache_dir
    # location first (post_image_dataset/lora/ — what configs/base.toml's
    # subset cache_dir points at), then the legacy sidecar location next to
    # the source image.
    stem = Path(ref_image).stem
    suffix = "_anima_te.safetensors"
    candidates = [
        ROOT / "post_image_dataset" / "lora" / f"{stem}{suffix}",
        Path(ref_image).parent / f"{stem}{suffix}",
    ]
    cache_path = next((p for p in candidates if p.is_file()), None)
    if cache_path is None:
        print(
            f"  ! No TE cache found for {ref_image}.\n"
            f"    Looked in: {candidates[0]}\n"
            f"           and: {candidates[1]}\n"
            "    Run `make preprocess-te` first (with --caption_shuffle_variants N "
            "to get a multi-variant cache).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  > TE cache: {cache_path}")

    save_dir = ROOT / "output" / "tests" / "directedit_dry"
    save_dir.mkdir(parents=True, exist_ok=True)

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop "inference.py"
    leftover_base = list(base_iter)
    args = [py, "scripts/edit.py", *_filter_inference_base_for_edit(leftover_base)]
    args += [
        "--image", str(ref_image),
        "--cached_embed", str(cache_path),
        "--save_path", str(save_dir),
    ]
    args += list(extra)
    run(args)

    # Copy the source alongside the reconstruction for side-by-side review.
    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_src.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
        shutil.copy(ref_image, src_dst)
        print(f"  > Source pasted: {src_dst}")


def _resolve_inference_base_flag(name: str) -> str | None:
    """Read a ``--name <value>`` pair out of ``INFERENCE_BASE``.

    Lets ``cmd_invert_directedit`` reuse the same ``--dit`` path as the
    other test commands without duplicating the model-path string.
    """
    base = list(INFERENCE_BASE)
    for i, tok in enumerate(base):
        if tok == name and i + 1 < len(base):
            return base[i + 1]
    return None


def _resolve_ref_image_pool(directory: Path, n: int) -> list[str]:
    """Pick ``n`` distinct random images from ``directory``.

    Same convention as ``_random_ref_image`` but returns a list (no
    replacement) for the N_IMAGES > 1 path.
    """
    if not directory.is_dir():
        return []
    pool = [p for p in directory.iterdir() if p.suffix.lower() in _REF_IMAGE_EXTS]
    if not pool:
        return []
    if n >= len(pool):
        return [str(p) for p in pool]
    return [str(p) for p in random.sample(pool, n)]


def cmd_invert_directedit(extra):
    """Probe: does an inverted per-image postfix tail make DirectEdit dry-mode
    reconstruction (ψ_tar == ψ_src) more robust than the bare T5(tags) prefix?

    For each of ``N_IMAGES`` source images:

      1. Invert the orthogonal postfix tail (scripts/inversion/invert_postfix_tail.py)
         — K trainable scales over a frozen Q@diag(s) basis, optimized against
         flow-matching loss through the frozen DiT.
      2. Run ``scripts/edit.py --cached_embed`` in dry mode against the
         **baseline** v0 prefix (T5(tags)).
      3. Splice ``Q @ diag(s)`` into the last K positions of that prefix and
         run dry mode again. This is the **postfix-augmented** ψ_src.

    Both runs write to ``output/tests/invert_directedit/<stem>/{baseline,postfix}/``
    with the source pasted alongside, so the postfix's reconstruction lift can
    be eyeballed without going through the full bench analyzer.

    Env vars (all optional, all match invert_postfix_tail.py defaults):
      N_IMAGES (default 1)   — number of images to process
      REF_IMAGE              — single explicit image path (sets N_IMAGES=1)
      K (default 48)         — tail length
      INVERT_STEPS (100)     — inversion optimization steps
      INVERT_LR (0.01)       — AdamW lr
      LAMBDA_ZERO (0.0)      — ‖s‖² regularization
      SIGMA_MIN (0.0)        — P-GRAFT low-σ skip (lower σ bound)
      SIGMA_MAX (1.0)        — upper σ bound; <1.0 restricts to low-σ supervision
      BASIS (svd_te)         — basis kind (svd_te|random)
      SEED (0)               — per-image inversion seed
      TIMESTEPS_PER_STEP (1) — batched sigmas per forward (× GRAD_ACCUM = total)
      GRAD_ACCUM (1)         — grad accum steps

    Examples:
      make exp-invert-directedit
      REF_IMAGE=post_image_dataset/resized/12345.png make exp-invert-directedit
      N_IMAGES=3 make exp-invert-directedit
      K=8 INVERT_STEPS=50 make exp-invert-directedit
    """
    sys.path.insert(0, str(ROOT))

    # 1. Resolve image pool.
    ref_image_override = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image_override and extra and not extra[0].startswith("-"):
        ref_image_override = extra[0]
        extra = extra[1:]

    n_images_env = os.environ.get("N_IMAGES", "").strip()
    try:
        n_images = max(1, int(n_images_env)) if n_images_env else 1
    except ValueError:
        print(f"  ! N_IMAGES={n_images_env!r} is not an int, using 1", file=sys.stderr)
        n_images = 1
    if ref_image_override:
        n_images = 1

    if ref_image_override:
        images = [ref_image_override]
    else:
        images = _resolve_ref_image_pool(
            ROOT / "post_image_dataset" / "resized", n_images
        )
    if not images:
        print(
            "Usage: python tasks.py exp-invert-directedit [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-invert-directedit\n"
            "   or: N_IMAGES=3 python tasks.py exp-invert-directedit\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Inversion knobs — env overrides for the common dials, defaults match
    #    the proposal (and the invert_postfix_tail.py CLI defaults).
    K = int(os.environ.get("K", "32"))
    invert_steps = int(os.environ.get("INVERT_STEPS", "30"))
    invert_lr = float(os.environ.get("INVERT_LR", "1e-2"))
    lambda_zero = float(os.environ.get("LAMBDA_ZERO", "0.0"))
    sigma_min = float(os.environ.get("SIGMA_MIN", "0"))
    sigma_max = float(os.environ.get("SIGMA_MAX", "0.5"))
    basis_kind = os.environ.get("BASIS", "svd_te").strip()
    seed = int(os.environ.get("SEED", "0"))
    timesteps_per_step = int(os.environ.get("TIMESTEPS_PER_STEP", "2"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "6"))

    run_root = ROOT / "output" / "tests" / "invert_directedit"
    run_root.mkdir(parents=True, exist_ok=True)
    basis_path = run_root / f"svd_basis_K{K}.pt"

    # 3. Resolve the shared --dit / etc. flags from INFERENCE_BASE so this
    #    target follows the same model trio as the rest of the test family.
    dit_path = _resolve_inference_base_flag("--dit")
    if dit_path is None:
        print("  ! INFERENCE_BASE has no --dit value", file=sys.stderr)
        sys.exit(1)
    attn_mode = _resolve_inference_base_flag("--attn_mode") or "flash"

    # Lazy import — keep the task module light when this command isn't run.
    from library.inference.postfix_inversion import (  # noqa: PLC0415
        load_or_build_basis,
        load_tail_s,
        splice_tail_into_te_cache,
    )

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop 'inference.py'
    leftover_base = list(base_iter)
    edit_base_args = [
        py,
        "scripts/edit.py",
        *_filter_inference_base_for_edit(leftover_base),
    ]

    for i, ref_image in enumerate(images):
        stem = Path(ref_image).stem
        print(f"\n=== [{i + 1}/{len(images)}] {stem} ===")

        # 4. Find the cached TE for this image (the baseline v0 prefix).
        suffix = "_anima_te.safetensors"
        te_candidates = [
            ROOT / "post_image_dataset" / "lora" / f"{stem}{suffix}",
            Path(ref_image).parent / f"{stem}{suffix}",
        ]
        te_path = next((p for p in te_candidates if p.is_file()), None)
        if te_path is None:
            print(
                f"  ! No TE cache found for {stem}. Looked in:\n"
                f"    {te_candidates[0]}\n    {te_candidates[1]}\n"
                "    Run `make preprocess-te` first.",
                file=sys.stderr,
            )
            continue

        # 5. Invert the postfix tail via the standalone CLI.
        img_dir = run_root / stem
        invert_out = img_dir / "inversion"
        s_path = invert_out / "s" / f"{stem}_s.safetensors"
        if s_path.exists():
            print(f"  > Reusing existing inversion: {s_path}")
        else:
            invert_cmd = [
                py,
                "scripts/inversion/invert_postfix_tail.py",
                "--dit", str(dit_path),
                "--attn_mode", str(attn_mode),
                "--image_dir", str(te_path.parent),
                "--image_stem", stem,
                "--K", str(K),
                "--basis", basis_kind,
                "--basis_path", str(basis_path),
                "--steps", str(invert_steps),
                "--lr", str(invert_lr),
                "--lambda_zero", str(lambda_zero),
                "--sigma_min", str(sigma_min),
                "--sigma_max", str(sigma_max),
                "--seed", str(seed),
                "--timesteps_per_step", str(timesteps_per_step),
                "--grad_accum", str(grad_accum),
                "--output_dir", str(invert_out),
                "--vr"
            ]
            run(invert_cmd)
            if not s_path.exists():
                print(
                    f"  ! Inversion did not produce {s_path}; skipping {stem}",
                    file=sys.stderr,
                )
                continue

        # 6. Build the spliced TE cache — load s + Q on CPU, splice, write.
        #    D=1024 matches Qwen3's hidden size (the only one Anima ships
        #    against); load_or_build_basis verifies cached shape and fails
        #    loud if the on-disk basis doesn't match.
        s, _ = load_tail_s(str(s_path))
        Q = load_or_build_basis(
            K=K,
            D=1024,
            kind=basis_kind,
            te_cache_dir=str(te_path.parent),
            basis_path=str(basis_path),
            seed=seed,
        )

        spliced_te = img_dir / "te_postfix.safetensors"
        splice_tail_into_te_cache(
            str(te_path),
            str(spliced_te),
            s=s,
            Q=Q,
            variant_index=0,
        )
        print(f"  > Spliced TE cache: {spliced_te}")

        # 7. Two dry-mode edit.py invocations: baseline + postfix.
        for label, cache_for_run in (
            ("baseline", te_path),
            ("postfix", spliced_te),
        ):
            save_dir = img_dir / label
            save_dir.mkdir(parents=True, exist_ok=True)
            edit_cmd = [
                *edit_base_args,
                "--image", str(ref_image),
                "--cached_embed", str(cache_for_run),
                "--cached_embed_variants", "0",
                "--save_path", str(save_dir),
                *list(extra),
            ]
            run(edit_cmd)

            # Paste the source for side-by-side eyeballing.
            pngs = sorted(
                (
                    p
                    for p in save_dir.glob("*.png")
                    if not p.name.endswith("_src.png")
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if pngs:
                src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
                shutil.copy(ref_image, src_dst)
                print(f"  > [{label}] source pasted: {src_dst}")


def _filter_inference_base_for_edit(args: list[str]) -> list[str]:
    """Drop INFERENCE_BASE flags that ``scripts/edit.py`` doesn't accept.

    INFERENCE_BASE bundles plenty of generation-only flags (--prompt, --seed,
    --image_size, --infer_steps, --sampler, etc.) that overlap with or
    conflict with edit.py's own. Keep only the model/path flags we actually
    want to forward; let edit.py supply its own defaults for the rest.
    """
    keep_flags = {
        "--dit",
        "--text_encoder",
        "--vae",
        "--vae_chunk_size",
        "--attn_mode",
    }
    boolean_flags = {"--vae_disable_cache"}
    out: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in keep_flags and i + 1 < len(args):
            out.extend([tok, args[i + 1]])
            i += 2
        elif tok in boolean_flags:
            out.append(tok)
            i += 1
        else:
            i += 1
    return out


def cmd_test_easycontrol(extra):
    """Inference with latest EasyControl weight.

    Reference image is taken from REF_IMAGE env or the first positional arg.
    Falls back to a random image from ``easycontrol-dataset/`` (the EasyControl
    source layout) when neither is supplied.
    PROMPT, NEG, EC_SCALE env vars override defaults. Saves to
    output/tests/easycontrol/ and copies the ref image alongside the generated
    output as ``<name>_ref.png``.

    Examples:
      python tasks.py exp-test-easycontrol ref.png --prompt "a girl in a coffee shop"
      REF_IMAGE=ref.png EC_SCALE=0.8 python tasks.py exp-test-easycontrol
      python tasks.py exp-test-easycontrol         # random ref from easycontrol-dataset/
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "easycontrol-dataset") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-easycontrol <ref_image> [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-easycontrol [extra...]\n"
            "   (no ref given and easycontrol-dataset/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dir = ROOT / "output" / "tests" / "easycontrol"
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        *INFERENCE_BASE,
        "--save_path",
        str(save_dir),
        "--easycontrol_weight",
        str(latest_output("anima_easycontrol")),
        "--easycontrol_image",
        ref_image,
        "--easycontrol_image_match_size",
    ]
    if scale := os.environ.get("EC_SCALE"):
        args += ["--easycontrol_scale", scale]
    if prompt := os.environ.get("PROMPT"):
        args += ["--prompt", prompt]
    if neg := os.environ.get("NEG"):
        args += ["--negative_prompt", neg]
    args += list(extra)
    run(args)

    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_ref.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        ref_dst = pngs[0].with_name(pngs[0].stem + "_ref.png")
        shutil.copy(ref_image, ref_dst)
        print(f"  > Ref pasted: {ref_dst}")

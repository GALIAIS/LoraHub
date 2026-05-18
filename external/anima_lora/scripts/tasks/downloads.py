"""Model download entry-points (Anima base, SAM3, MIT, PE-Core).

All targets shell out to ``hf download`` (rather than the SDK) so the user's
``hf auth login`` cache is honored.
"""

from __future__ import annotations

import shutil

from ._common import ROOT, run


def cmd_download_sam3(_extra):
    (ROOT / "models" / "sam3").mkdir(parents=True, exist_ok=True)
    run(["hf", "download", "facebook/sam3", "--local-dir", "models/sam3"])


def cmd_download_pe(_extra):
    # PE-Core-L14-336 — only the .pt checkpoint is needed; vision tower is
    # vendored at library/models/pe.py (no perception_models clone required).
    (ROOT / "models" / "pe").mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "facebook/PE-Core-L14-336",
            "PE-Core-L14-336.pt",
            "--local-dir",
            "models/pe",
        ]
    )


def cmd_download_pe_spatial(_extra):
    # PE-Spatial-B16-512 — auxiliary encoder for the Anima Tagger's
    # dual-encoder configuration. Same vendored vision tower (different
    # config entry); only the .pt is fetched here.
    (ROOT / "models" / "pe").mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "facebook/PE-Spatial-B16-512",
            "PE-Spatial-B16-512.pt",
            "--local-dir",
            "models/pe",
        ]
    )


def cmd_download_mit(_extra):
    (ROOT / "models" / "mit").mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "a-b-c-x-y-z/Manga-Text-Segmentation-2025",
            "model.pth",
            "--local-dir",
            "models/mit",
        ]
    )


def cmd_download_anima(_extra):
    for d in ["diffusion_models", "text_encoders", "vae"]:
        (ROOT / "models" / d).mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "circlestone-labs/Anima",
            "split_files/diffusion_models/anima-base-v1.0.safetensors",
            "split_files/text_encoders/qwen_3_06b_base.safetensors",
            "split_files/vae/qwen_image_vae.safetensors",
            "--local-dir",
            "models",
            "--include",
            "split_files/*",
        ]
    )
    split = ROOT / "models" / "split_files"
    for subdir in ["diffusion_models", "text_encoders", "vae"]:
        src = split / subdir
        dst = ROOT / "models" / subdir
        if src.exists():
            for f in src.iterdir():
                shutil.move(str(f), str(dst / f.name))
    if split.exists():
        shutil.rmtree(split)


def cmd_download_models(_extra):
    cmd_download_anima(_extra)
    cmd_download_sam3(_extra)
    cmd_download_mit(_extra)
    cmd_download_pe(_extra)
    cmd_download_pe_spatial(_extra)

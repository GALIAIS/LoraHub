#!/usr/bin/env python3
"""Download the Qwen3 TE for Anima via huggingface_hub.

Why python instead of wget: hf-mirror's xet redirect lands on
cas-bridge.xethub.hf.co with a TLS handshake that the system OpenSSL
on this VPS terminates with 'unexpected eof while reading'. The
Python `requests` stack used by huggingface_hub uses a different SSL
backend and downloads the same file fine.
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Force the Chinese mirror so the download stays in-region.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path("/root/autodl-tmp/LoraHub/models/circlestone-labs__Anima/split_files")
TARGETS = [
    ("split_files/text_encoders/qwen_3_06b_base.safetensors",
     ROOT / "text_encoders" / "qwen_3_06b_base.safetensors"),
    # transformer + vae are already downloaded but we keep them here in
    # case anyone re-runs this script after wiping the dir.
    ("split_files/diffusion_models/anima-base-v1.0.safetensors",
     ROOT / "diffusion_models" / "anima-base-v1.0.safetensors"),
    ("split_files/vae/qwen_image_vae.safetensors",
     ROOT / "vae" / "qwen_image_vae.safetensors"),
]

REPO = "circlestone-labs/Anima"

for hf_rel, dest in TARGETS:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size:,} bytes)")
        continue
    print(f"[fetch] {hf_rel} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=REPO,
        filename=hf_rel,
        local_dir=str(ROOT.parent),  # file ends up under .../circlestone-labs__Anima/split_files/...
        local_dir_use_symlinks=False,
    )
    print(f"[done]  {cached}")

print("[ok] all files present")
for hf_rel, dest in TARGETS:
    if dest.exists():
        print(f"  {dest.relative_to(ROOT)}: {dest.stat().st_size:,} bytes")

"""Tests for the Anima inference backend (subprocess wrapper + LoRA conversion)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from lorahub.core.inference import PromptSpec
from lorahub.core.inference.anima import (
    AnimaInferenceBackend,
    AnimaInferenceConfig,
    InferenceFailed,
    InferenceSkipped,
    convert_dp_lora_to_kohya,
)


# --------------------------------------------------------------------------- #
# convert_dp_lora_to_kohya
# --------------------------------------------------------------------------- #


def _write_dp_lora(
    path: Path,
    *,
    rank: int = 4,
    in_dim: int = 8,
    out_dim: int = 16,
    write_config: bool = True,
    alpha: int = 4,
) -> None:
    """Synthesize a tiny dp-style adapter_model.safetensors with a
    couple of representative modules (a DiT block and an LLM-adapter
    block) so we can verify both prefix branches survive conversion."""
    state = {
        "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, in_dim),
        "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight": torch.randn(out_dim, rank),
        "diffusion_model.blocks.0.mlp.layer1.lora_A.weight": torch.randn(rank, in_dim),
        "diffusion_model.blocks.0.mlp.layer1.lora_B.weight": torch.randn(out_dim, rank),
        # LLM-adapter modules use o_proj rather than output_proj.
        "diffusion_model.llm_adapter.blocks.0.cross_attn.o_proj.lora_A.weight": torch.randn(rank, in_dim),
        "diffusion_model.llm_adapter.blocks.0.cross_attn.o_proj.lora_B.weight": torch.randn(out_dim, rank),
    }
    save_file(state, str(path), metadata={"format": "pt"})
    if write_config:
        (path.parent / "adapter_config.json").write_text(
            json.dumps({"r": rank, "lora_alpha": alpha, "target_modules": ["Block"]}),
            encoding="utf-8",
        )


def test_convert_emits_kohya_keys(tmp_path: Path) -> None:
    src = tmp_path / "step100" / "adapter_model.safetensors"
    src.parent.mkdir()
    _write_dp_lora(src, rank=4, alpha=4)
    out = tmp_path / "step100" / "lorahub_converted.safetensors"

    convert_dp_lora_to_kohya(src, out)

    assert out.is_file()
    with safe_open(str(out), framework="pt") as f:
        keys = set(f.keys())
        meta = f.metadata() or {}

    # Each module produces three artefacts: lora_down / lora_up / alpha.
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight" in keys
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight" in keys
    assert "lora_unet_blocks_0_self_attn_q_proj.alpha" in keys
    assert "lora_unet_blocks_0_mlp_layer1.lora_down.weight" in keys
    # LLM-adapter modules end up under the same lora_unet_ prefix.
    assert "lora_unet_llm_adapter_blocks_0_cross_attn_o_proj.lora_down.weight" in keys

    # No lingering peft-style keys.
    assert not any(".lora_A." in k for k in keys)
    assert not any(".lora_B." in k for k in keys)
    assert not any(k.startswith("diffusion_model.") for k in keys)

    # Metadata round-trips with our custom marker.
    assert meta.get("format") == "pt"
    assert meta.get("lorahub_source") == "diffusion-pipe-peft"


def test_convert_alpha_value_matches_config(tmp_path: Path) -> None:
    src = tmp_path / "step100" / "adapter_model.safetensors"
    src.parent.mkdir()
    _write_dp_lora(src, rank=8, alpha=8)
    out = tmp_path / "out.safetensors"

    convert_dp_lora_to_kohya(src, out)
    with safe_open(str(out), framework="pt") as f:
        alpha = f.get_tensor("lora_unet_blocks_0_self_attn_q_proj.alpha")
    assert alpha.item() == 8.0


def test_convert_uses_fallbacks_when_config_missing(tmp_path: Path) -> None:
    src = tmp_path / "step100" / "adapter_model.safetensors"
    src.parent.mkdir()
    _write_dp_lora(src, rank=4, alpha=4, write_config=False)
    out = tmp_path / "out.safetensors"

    convert_dp_lora_to_kohya(src, out, rank_fallback=99, alpha_fallback=99.0)
    with safe_open(str(out), framework="pt") as f:
        alpha = f.get_tensor("lora_unet_blocks_0_self_attn_q_proj.alpha")
    assert alpha.item() == 99.0


def test_convert_skips_partial_pairs(tmp_path: Path) -> None:
    """If only lora_A or lora_B is present (corrupted save), drop that
    pair entirely rather than crash."""
    src = tmp_path / "step100" / "adapter_model.safetensors"
    src.parent.mkdir()
    state = {
        "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 8),
        # missing matching lora_B
        "diffusion_model.blocks.1.self_attn.q_proj.lora_A.weight": torch.randn(4, 8),
        "diffusion_model.blocks.1.self_attn.q_proj.lora_B.weight": torch.randn(16, 4),
    }
    save_file(state, str(src), metadata={"format": "pt"})
    out = tmp_path / "out.safetensors"

    convert_dp_lora_to_kohya(src, out, rank_fallback=4, alpha_fallback=4.0)
    with safe_open(str(out), framework="pt") as f:
        keys = set(f.keys())
    # block 1 survives, block 0 dropped.
    assert "lora_unet_blocks_1_self_attn_q_proj.lora_down.weight" in keys
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight" not in keys


# --------------------------------------------------------------------------- #
# AnimaInferenceBackend.render — subprocess interactions
# --------------------------------------------------------------------------- #


def _build_cfg(tmp_path: Path) -> AnimaInferenceConfig:
    """All paths point to existing tmp files so _sanity_check passes;
    individual tests then patch subprocess behaviour."""
    py = tmp_path / "venv-python.exe"
    repo = tmp_path / "sd-scripts"
    transformer = tmp_path / "transformer.safetensors"
    vae = tmp_path / "vae.safetensors"
    te = tmp_path / "te.safetensors"
    repo.mkdir()
    (repo / "anima_minimal_inference.py").write_text("# stub")
    py.write_text("# stub")
    transformer.write_text("base")
    vae.write_text("vae")
    te.write_text("te")
    return AnimaInferenceConfig(
        sd_scripts_python=py,
        sd_scripts_repo=repo,
        transformer_path=transformer,
        vae_path=vae,
        text_encoder_path=te,
        timeout_per_image_s=10.0,
        min_free_vram_mib=0,  # disable VRAM gate by default
    )


def _make_lora_input(tmp_path: Path) -> Path:
    src = tmp_path / "step100" / "adapter_model.safetensors"
    src.parent.mkdir()
    _write_dp_lora(src)
    return src


class _FakeProc:
    def __init__(self, returncode: int, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_render_writes_png_and_converts_lora(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_cfg(tmp_path)
    backend = AnimaInferenceBackend(cfg)
    lora = _make_lora_input(tmp_path)
    out = tmp_path / "samples" / "step100_00.png"
    spec = PromptSpec(prompt="hello", index=0, width=128, height=128)

    def _fake_run(cmd, **kwargs):
        # Simulate the subprocess writing a PNG into --save_path.
        save_idx = cmd.index("--save_path")
        scratch = Path(cmd[save_idx + 1])
        (scratch / "result.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        return _FakeProc(returncode=0)

    monkeypatch.setattr("subprocess.run", _fake_run)
    backend.render(
        lora_path=lora,
        spec=spec,
        out_path=out,
        default_steps=8,
        default_cfg=4.0,
    )

    assert out.is_file()
    # Side-effect: converted LoRA cached next to the original.
    converted = lora.parent / "lorahub_converted.safetensors"
    assert converted.is_file()


def test_render_raises_inference_failed_on_nonzero_exit(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _build_cfg(tmp_path)
    backend = AnimaInferenceBackend(cfg)
    lora = _make_lora_input(tmp_path)
    out = tmp_path / "samples" / "x.png"
    spec = PromptSpec(prompt="hi", index=0)

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: _FakeProc(returncode=2, stderr="boom"),
    )
    with pytest.raises(InferenceFailed) as excinfo:
        backend.render(
            lora_path=lora, spec=spec, out_path=out,
            default_steps=8, default_cfg=4.0,
        )
    assert "exited 2" in str(excinfo.value)


def test_render_raises_inference_failed_when_no_png_produced(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _build_cfg(tmp_path)
    backend = AnimaInferenceBackend(cfg)
    lora = _make_lora_input(tmp_path)
    out = tmp_path / "samples" / "x.png"
    spec = PromptSpec(prompt="hi", index=0)

    # Subprocess succeeds but writes nothing.
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: _FakeProc(returncode=0),
    )
    with pytest.raises(InferenceFailed) as excinfo:
        backend.render(
            lora_path=lora, spec=spec, out_path=out,
            default_steps=8, default_cfg=4.0,
        )
    assert "produced no PNG" in str(excinfo.value)


def test_render_raises_inference_skipped_on_low_vram(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.min_free_vram_mib = 999_999  # make the gate impossible to pass
    backend = AnimaInferenceBackend(cfg)
    lora = _make_lora_input(tmp_path)
    out = tmp_path / "samples" / "x.png"
    spec = PromptSpec(prompt="hi", index=0)

    # Force the VRAM helper to report a small amount of free memory.
    monkeypatch.setattr(
        "lorahub.core.inference.anima._has_enough_vram", lambda _min: False
    )
    with pytest.raises(InferenceSkipped):
        backend.render(
            lora_path=lora, spec=spec, out_path=out,
            default_steps=8, default_cfg=4.0,
        )


def test_sanity_check_lists_missing_paths(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.transformer_path = tmp_path / "does_not_exist.safetensors"
    backend = AnimaInferenceBackend(cfg)
    lora = _make_lora_input(tmp_path)
    out = tmp_path / "x.png"

    with pytest.raises(InferenceFailed) as excinfo:
        backend.render(
            lora_path=lora,
            spec=PromptSpec(prompt="hi"),
            out_path=out,
            default_steps=8,
            default_cfg=4.0,
        )
    assert "transformer=" in str(excinfo.value)

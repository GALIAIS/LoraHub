"""anima_lora preview backend tests — registration + factory + render argv.

We don't run inference.py for real (would need 30+ GB of weights and
torch 2.11). Tests cover the contract layer: registration order
(anima_lora wins ahead of in-process anima), factory gating
(missing dit/vae/qwen3 paths → fall through), is_available behaviour,
and render argv frame.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lorahub.core.inference import PromptSpec
from lorahub.core.inference.anima_lora_backend import (
    AnimaLoraInferenceBackend,
    _anima_lora_factory,
)

# --------------------------------------------------------------------------- #
# Registration order — anima_lora wins ahead of anima
# --------------------------------------------------------------------------- #


def test_anima_lora_registered_before_anima() -> None:
    """anima_lora's factory must run first so it can claim arch=anima.

    Importing the backend modules side-effect-registers them. We import
    them in the same order jobs_helpers.py does (anima → anima_lora →
    diffusers) and assert the chain has anima_lora at index 0.
    """
    from lorahub.core.inference import anima as _anima_mod  # noqa: F401
    from lorahub.core.inference import (  # noqa: F401
        anima_lora_backend as _anima_lora_mod,
    )
    from lorahub.core.inference.registry import registered_backend_names

    names = registered_backend_names()
    assert "anima_lora" in names
    assert "anima" in names
    # anima_lora must come before anima so resolve_backend() reaches it
    # first when both factories could serve the arch.
    assert names.index("anima_lora") < names.index("anima")


# --------------------------------------------------------------------------- #
# Factory — gating + path extraction
# --------------------------------------------------------------------------- #


def _fake_recipe(
    *,
    dit: Path | None,
    vae: Path | None,
    qwen3: Path | None,
) -> SimpleNamespace:
    """Build a SimpleNamespace shaped like TrainingConfig for the factory.

    We deliberately don't go through Pydantic — the factory only reads
    a handful of fields by attribute, so a SimpleNamespace keeps the
    test independent of TrainingConfig's many other required fields.
    """
    return SimpleNamespace(
        backend=SimpleNamespace(
            python_executable=None, sd_scripts_path=None
        ),
        base_model=SimpleNamespace(
            checkpoint=dit,
            arch_paths=SimpleNamespace(ae=vae, qwen3=qwen3),
        ),
    )


def test_factory_returns_none_for_non_anima_arch(tmp_path: Path) -> None:
    """Factory must skip arches it can't serve so the registry falls through."""
    p = tmp_path / "f"
    p.write_bytes(b"")
    recipe = _fake_recipe(dit=p, vae=p, qwen3=p)
    assert _anima_lora_factory(arch="sdxl", recipe=recipe, workspace=tmp_path) is None
    assert _anima_lora_factory(arch="flux", recipe=recipe, workspace=tmp_path) is None


def test_factory_returns_none_when_missing_dit_path(tmp_path: Path) -> None:
    """Missing dit/vae/qwen3 path → factory bows out, registry tries next."""
    p = tmp_path / "f"
    p.write_bytes(b"")
    recipe = _fake_recipe(dit=None, vae=p, qwen3=p)
    assert _anima_lora_factory(arch="anima", recipe=recipe, workspace=tmp_path) is None


def test_factory_returns_none_when_missing_vae_path(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_bytes(b"")
    recipe = _fake_recipe(dit=p, vae=None, qwen3=p)
    assert _anima_lora_factory(arch="anima", recipe=recipe, workspace=tmp_path) is None


def test_factory_returns_none_when_missing_qwen3_path(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_bytes(b"")
    recipe = _fake_recipe(dit=p, vae=p, qwen3=None)
    assert _anima_lora_factory(arch="anima", recipe=recipe, workspace=tmp_path) is None


def test_factory_returns_backend_when_all_paths_present(tmp_path: Path) -> None:
    """All three artefacts on disk + vendored copy → backend constructed."""
    dit = tmp_path / "dit.safetensors"
    vae = tmp_path / "vae"
    vae.mkdir()
    qwen3 = tmp_path / "qwen3"
    qwen3.mkdir()
    for p in (dit,):
        p.write_bytes(b"")

    recipe = _fake_recipe(dit=dit, vae=vae, qwen3=qwen3)
    backend = _anima_lora_factory(arch="anima", recipe=recipe, workspace=tmp_path)
    assert backend is not None
    assert isinstance(backend, AnimaLoraInferenceBackend)
    assert backend.name == "anima_lora"
    assert backend.is_available(arch="anima")
    # Other arches still skipped by the constructed backend.
    assert not backend.is_available(arch="sdxl")


def test_factory_returns_none_when_recipe_is_none() -> None:
    """No recipe → factory bows out (used by registry resolve_backend)."""
    assert _anima_lora_factory(arch="anima", recipe=None, workspace=None) is None


# --------------------------------------------------------------------------- #
# is_available — gates on artefact presence
# --------------------------------------------------------------------------- #


def test_is_available_false_when_dit_missing(tmp_path: Path) -> None:
    """Backend constructed but DiT vanishes between probe + render → False."""
    from lorahub.core.backends.anima_lora.bootstrap import resolve

    env = resolve()
    dit = tmp_path / "missing.safetensors"
    vae = tmp_path / "vae"
    vae.mkdir()
    qwen3 = tmp_path / "qwen3"
    qwen3.mkdir()
    backend = AnimaLoraInferenceBackend(
        env=env, dit_path=dit, vae_path=vae, text_encoder_path=qwen3
    )
    assert backend.is_available(arch="anima") is False


# --------------------------------------------------------------------------- #
# render — argv frame
# --------------------------------------------------------------------------- #


def test_render_builds_inference_argv(tmp_path: Path) -> None:
    """`render` invokes inference.py with the expected flag set.

    We patch subprocess.run so the test doesn't need torch / diffusers
    on the host, plus a touch on the output path so the post-run check
    that it exists doesn't fail.
    """
    from lorahub.core.backends.anima_lora.bootstrap import resolve

    env = resolve()
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    vae = tmp_path / "vae"
    vae.mkdir()
    qwen3 = tmp_path / "qwen3"
    qwen3.mkdir()
    out = tmp_path / "preview.png"
    backend = AnimaLoraInferenceBackend(
        env=env, dit_path=dit, vae_path=vae, text_encoder_path=qwen3
    )

    captured: list[list[str]] = []

    class _FakeProc:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(list(argv))
        out.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal png header
        return _FakeProc()

    spec = PromptSpec(
        prompt="cat in space",
        width=1024,
        height=1024,
        seed=42,
        steps=20,
        cfg=4.5,
        negative="blurry",
    )
    with patch(
        "lorahub.core.inference.anima_lora_backend.subprocess.run", fake_run
    ):
        backend.render(
            lora_path=tmp_path / "lora.safetensors",
            spec=spec,
            out_path=out,
            default_steps=50,
            default_cfg=3.5,
        )

    assert len(captured) == 1
    argv = captured[0]
    assert argv[1].endswith("inference.py")
    # All required upstream flags must be present.
    for flag in (
        "--dit", "--vae", "--text_encoder",
        "--lora_weight", "--prompt", "--image_size",
        "--infer_steps", "--guidance_scale", "--save_path",
        "--seed", "--negative_prompt",
    ):
        assert flag in argv, f"render argv missing {flag}: {argv}"
    # Spec overrides must win over defaults.
    seed_idx = argv.index("--seed")
    assert argv[seed_idx + 1] == "42"
    steps_idx = argv.index("--infer_steps")
    assert argv[steps_idx + 1] == "20"


def test_render_falls_back_to_default_steps_and_cfg(tmp_path: Path) -> None:
    """When spec leaves steps/cfg None, the worker defaults flow through."""
    from lorahub.core.backends.anima_lora.bootstrap import resolve

    env = resolve()
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    vae = tmp_path / "vae"
    vae.mkdir()
    qwen3 = tmp_path / "qwen3"
    qwen3.mkdir()
    out = tmp_path / "preview.png"
    backend = AnimaLoraInferenceBackend(
        env=env, dit_path=dit, vae_path=vae, text_encoder_path=qwen3
    )
    captured: list[list[str]] = []

    class _FakeProc:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(list(argv))
        out.write_bytes(b"\x89PNG\r\n\x1a\n")
        return _FakeProc()

    spec = PromptSpec(prompt="hi", width=512, height=512)
    with patch(
        "lorahub.core.inference.anima_lora_backend.subprocess.run", fake_run
    ):
        backend.render(
            lora_path=tmp_path / "lora.safetensors",
            spec=spec,
            out_path=out,
            default_steps=50,
            default_cfg=3.5,
        )

    argv = captured[0]
    steps_idx = argv.index("--infer_steps")
    assert argv[steps_idx + 1] == "50"
    cfg_idx = argv.index("--guidance_scale")
    # repr(float(3.5)) — keep the format stable.
    assert argv[cfg_idx + 1] == repr(3.5)
    # No --seed when spec.seed is None.
    assert "--seed" not in argv
    # No --negative_prompt when spec.negative is None.
    assert "--negative_prompt" not in argv


def test_render_raises_when_subprocess_fails(tmp_path: Path) -> None:
    """A non-zero return code surfaces as RuntimeError with the tail."""
    from lorahub.core.backends.anima_lora.bootstrap import resolve

    env = resolve()
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    vae = tmp_path / "vae"
    vae.mkdir()
    qwen3 = tmp_path / "qwen3"
    qwen3.mkdir()
    backend = AnimaLoraInferenceBackend(
        env=env, dit_path=dit, vae_path=vae, text_encoder_path=qwen3
    )

    class _FakeProc:
        returncode = 1
        stderr = "RuntimeError: CUDA out of memory"
        stdout = ""

    with patch(
        "lorahub.core.inference.anima_lora_backend.subprocess.run",
        lambda *_a, **_k: _FakeProc(),
    ):
        with pytest.raises(RuntimeError, match="exited 1"):
            backend.render(
                lora_path=tmp_path / "lora.safetensors",
                spec=PromptSpec(prompt="x"),
                out_path=tmp_path / "out.png",
                default_steps=50,
                default_cfg=3.5,
            )

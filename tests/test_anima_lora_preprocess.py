"""anima_lora auto-preprocess tests — cache detection + spawn orchestration.

The real preprocess steps spawn ``python preprocess/resize_images.py`` and
friends, which need torch + Anima models on disk. We don't run any of that
in CI: instead we patch the SubprocessRunner factory so each invocation
records what would have been spawned, and assert the orchestration logic
itself.

Three scenarios:

- Cache complete  → no subprocess invoked, function returns immediately.
- Cache missing   → resize / latent / TE steps spawn in order with the
                    expected argv shape.
- No images       → :class:`PreprocessError` raised before any spawn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lorahub.core.backends.anima_lora.bootstrap import AnimaLoraEnv
from lorahub.core.backends.anima_lora.preprocess import (
    PreprocessError,
    ensure_cache,
)
from lorahub.core.config.schema import BaseModelConfig


def _env(tmp_path: Path) -> AnimaLoraEnv:
    repo = tmp_path / "anima_lora"
    repo.mkdir()
    (repo / "preprocess").mkdir()
    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    return AnimaLoraEnv(repo_path=repo.resolve(), python_executable=py.resolve())


def _base_model(tmp_path: Path) -> BaseModelConfig:
    """Construct a BaseModelConfig with all three required path fields set."""
    ckpt = tmp_path / "dit.safetensors"
    ckpt.write_bytes(b"")
    qwen = tmp_path / "qwen3.safetensors"
    qwen.write_bytes(b"")
    ae = tmp_path / "vae.safetensors"
    ae.write_bytes(b"")
    return BaseModelConfig.model_validate(
        {
            "arch": "anima",
            "checkpoint": str(ckpt),
            "archPaths": {"qwen3": str(qwen), "ae": str(ae)},
        }
    )


def _stub_runner_factory(spawned: list[list[str]]) -> Any:
    """Build a runner_factory that stores argv + returns a stub runner.

    The stub mimics the real :class:`SubprocessRunner` surface
    ``ensure_cache`` consumes: ``start()`` (no-op), ``wait()``
    (returns ``RunResult(returncode=0, ...)``).
    """
    from lorahub.core.backends._common.runner import RunResult

    def factory(**kwargs: Any) -> Any:  # noqa: ANN401
        spawned.append(list(kwargs["argv"]))
        runner = MagicMock()
        runner.start = MagicMock(return_value=None)
        runner.wait = MagicMock(return_value=RunResult(returncode=0, duration_s=0.0))
        return runner

    return factory


def test_ensure_cache_skips_when_complete(tmp_path: Path) -> None:
    """All TE caches present → no preprocess subprocess spawned."""
    env = _env(tmp_path)
    bm = _base_model(tmp_path)
    image_dir = tmp_path / "raw"
    image_dir.mkdir()
    workspace = tmp_path / "ws"
    cache_dir = workspace / "post_image_dataset" / "lora"
    cache_dir.mkdir(parents=True)
    # Two images, both already cached (TE sidecar present).
    for stem in ("a", "b"):
        (image_dir / f"{stem}.jpg").write_bytes(b"")
        (image_dir / f"{stem}.txt").write_text("tag", encoding="utf-8")
        (cache_dir / f"{stem}_anima_te.safetensors").write_bytes(b"")

    spawned: list[list[str]] = []
    factory = _stub_runner_factory(spawned)
    ensure_cache(
        image_dir=image_dir,
        workspace=workspace,
        base_model=bm,
        env=env,
        runner_factory=factory,
    )
    assert spawned == []


def test_ensure_cache_spawns_when_missing(tmp_path: Path) -> None:
    """Missing TE cache → resize + cache_latents + cache_text_embeddings spawn."""
    env = _env(tmp_path)
    bm = _base_model(tmp_path)
    image_dir = tmp_path / "raw"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"")
    (image_dir / "a.txt").write_text("tag", encoding="utf-8")
    workspace = tmp_path / "ws"

    spawned: list[list[str]] = []
    factory = _stub_runner_factory(spawned)
    ensure_cache(
        image_dir=image_dir,
        workspace=workspace,
        base_model=bm,
        env=env,
        runner_factory=factory,
    )
    # Three steps — same order resize → latents → TE.
    assert len(spawned) == 3
    assert spawned[0][1].endswith("resize_images.py")
    assert spawned[1][1].endswith("cache_latents.py")
    assert spawned[2][1].endswith("cache_text_embeddings.py")
    # The cache flag must point at the workspace post_image_dataset/lora.
    expected_cache = (
        workspace / "post_image_dataset" / "lora"
    ).resolve()
    assert str(expected_cache) in spawned[1]
    assert str(expected_cache) in spawned[2]


def test_ensure_cache_fails_when_no_images(tmp_path: Path) -> None:
    """Empty raw image dir → PreprocessError raised before any spawn."""
    env = _env(tmp_path)
    bm = _base_model(tmp_path)
    image_dir = tmp_path / "raw"
    image_dir.mkdir()
    workspace = tmp_path / "ws"

    spawned: list[list[str]] = []
    factory = _stub_runner_factory(spawned)
    with pytest.raises(PreprocessError, match="no images found"):
        ensure_cache(
            image_dir=image_dir,
            workspace=workspace,
            base_model=bm,
            env=env,
            runner_factory=factory,
        )
    assert spawned == []


def test_ensure_cache_propagates_subprocess_failure(tmp_path: Path) -> None:
    """A non-zero exit from any preprocess step surfaces as PreprocessError."""
    from lorahub.core.backends._common.runner import RunResult

    env = _env(tmp_path)
    bm = _base_model(tmp_path)
    image_dir = tmp_path / "raw"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"")
    workspace = tmp_path / "ws"

    def failing_factory(**kwargs: Any) -> Any:  # noqa: ANN401
        runner = MagicMock()
        runner.start = MagicMock(return_value=None)
        runner.wait = MagicMock(
            return_value=RunResult(returncode=2, duration_s=0.1)
        )
        return runner

    with pytest.raises(PreprocessError, match="returncode=2"):
        ensure_cache(
            image_dir=image_dir,
            workspace=workspace,
            base_model=bm,
            env=env,
            runner_factory=failing_factory,
        )

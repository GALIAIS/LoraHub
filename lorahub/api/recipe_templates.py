"""Built-in recipe templates served by ``GET /api/recipes/templates``.

The web UI lets the user spawn a new recipe from one of these starting points
instead of having to fill the whole form from scratch. We keep the templates
inline (no disk I/O) and validate each one through :class:`RecipeConfig` at
import time — that way a typo in a template surfaces as an ImportError on
server start rather than as a 500 the first time someone clicks "New".

Path-bearing fields (``checkpoint`` / ``dataset.source``) are intentionally
left blank: ``pathlib.Path("")`` is a valid Path, so the schema accepts it,
and the user is expected to fill the real path in the form before saving.
"""

from __future__ import annotations

from typing import Any

from lorahub.core.config.schema import RecipeConfig


def _sdxl_character() -> dict[str, Any]:
    return {
        "base_model": {"arch": "sdxl", "checkpoint": ""},
        "dataset": {
            "source": "",
            "resolution": [1024, 1024],
            "num_repeats": 10,
        },
        "network": {
            "type": "lora",
            "rank": 32,
            "alpha": 16,
            "target_unet": True,
            "target_text_encoder": False,
        },
        "optimizer": {
            "type": "adamw8bit",
            "lr": {"unet": 1.0e-4, "text_encoder": 5.0e-5},
            "schedule": "cosine_with_restarts",
            "warmup_steps": 100,
        },
        "schedule": {"epochs": 10, "batch_size": 1, "grad_accum": 2},
        "precision": "bf16",
        "gradient_checkpointing": True,
        "cache_latents": True,
        "sampling": {"enabled": False},
        "output": {"save_dtype": "fp16"},
    }


def _sdxl_style() -> dict[str, Any]:
    return {
        "base_model": {"arch": "sdxl", "checkpoint": ""},
        "dataset": {
            "source": "",
            "resolution": [1024, 1024],
            "num_repeats": 4,
        },
        "network": {
            "type": "lora",
            "rank": 16,
            "alpha": 8,
            "target_unet": True,
            "target_text_encoder": True,
        },
        "optimizer": {
            "type": "adamw8bit",
            "lr": {"unet": 1.0e-4, "text_encoder": 5.0e-5},
            "schedule": "cosine_with_restarts",
            "warmup_steps": 100,
        },
        "schedule": {"epochs": 20, "batch_size": 1, "grad_accum": 2},
        "precision": "bf16",
        "gradient_checkpointing": True,
        "cache_latents": True,
        "sampling": {"enabled": False},
        "output": {"save_dtype": "fp16"},
    }


def _sd15_character() -> dict[str, Any]:
    return {
        "base_model": {"arch": "sd15", "checkpoint": ""},
        "dataset": {
            "source": "",
            "resolution": [768, 768],
            "num_repeats": 10,
        },
        "network": {
            "type": "lora",
            "rank": 16,
            "alpha": 8,
            "target_unet": True,
            "target_text_encoder": False,
        },
        "optimizer": {
            "type": "adamw8bit",
            "lr": {"unet": 5.0e-5, "text_encoder": 5.0e-5},
            "schedule": "cosine_with_restarts",
            "warmup_steps": 100,
        },
        "schedule": {"epochs": 10, "batch_size": 1, "grad_accum": 2},
        "precision": "fp16",
        "gradient_checkpointing": True,
        "cache_latents": True,
        "sampling": {"enabled": False},
        "output": {"save_dtype": "fp16"},
    }


def _blank() -> dict[str, Any]:
    # Smallest skeleton that still validates: arch + checkpoint + dataset source.
    # Everything else falls back to RecipeConfig defaults.
    return {
        "base_model": {"arch": "sdxl", "checkpoint": ""},
        "dataset": {"source": ""},
    }


_TEMPLATE_DEFS: list[dict[str, Any]] = [
    {
        "id": "sdxl_character",
        "name": "SDXL Character",
        "description": "8GB VRAM friendly SDXL character LoRA (rank 32 / 1024px).",
        "arch": "sdxl",
        "recipe": _sdxl_character(),
    },
    {
        "id": "sdxl_style",
        "name": "SDXL Style",
        "description": "Lower-rank SDXL style LoRA, trains the text encoder for 20 epochs.",
        "arch": "sdxl",
        "recipe": _sdxl_style(),
    },
    {
        "id": "sd15_character",
        "name": "SD 1.5 Character",
        "description": "SD 1.5 character LoRA at 768px, fp16 for older GPUs.",
        "arch": "sd15",
        "recipe": _sd15_character(),
    },
    {
        "id": "blank",
        "name": "Blank",
        "description": "Minimal skeleton — fill every field manually.",
        "arch": "sdxl",
        "recipe": _blank(),
    },
]


def _validate_templates() -> list[dict[str, Any]]:
    """Run ``RecipeConfig.model_validate`` on every template at import time.

    Returns the validated template list. A bad template raises ImportError
    instead of crashing the first user that clicks "New from template".
    """
    for tpl in _TEMPLATE_DEFS:
        try:
            RecipeConfig.model_validate(tpl["recipe"])
        except Exception as exc:  # noqa: BLE001
            msg = f"built-in recipe template {tpl['id']!r} failed validation: {exc}"
            raise ImportError(msg) from exc
    return _TEMPLATE_DEFS


TEMPLATES: list[dict[str, Any]] = _validate_templates()


__all__ = ["TEMPLATES"]

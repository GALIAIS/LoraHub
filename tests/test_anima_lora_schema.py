"""Cut0 tests — `AnimaLoraOptions` schema + dispatch hook.

Locks the public shape of the third backend's options so future cuts
(compiler, runner, preview backend) don't accidentally rename or drop
fields the UI / tests / serialised configs depend on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pydantic
import pytest

from lorahub.api.jobs_helpers import _select_backend
from lorahub.core.config.schema import (
    AnimaLoraMethodChimeraConfig,
    AnimaLoraMethodEasyControlConfig,
    AnimaLoraMethodIPAdapterConfig,
    AnimaLoraMethodPostfixConfig,
    AnimaLoraOptions,
    BackendConfig,
    TrainingConfig,
)


def _minimal_recipe(tmp_path: Path) -> dict[str, Any]:
    ckpt = tmp_path / "m.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    return {
        "base_model": {"checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "optimizer": {"lr": {"unet": 1.0e-4, "text_encoder": 5.0e-5}},
        "network": {"rank": 16, "alpha": 8},
        "output": {"name": "demo"},
    }


def test_default_anima_lora_options_constructs_clean() -> None:
    """Empty-args ``AnimaLoraOptions`` mirrors anima_lora's lora.toml defaults."""
    opts = AnimaLoraOptions()
    assert opts.method == "lora"
    assert opts.preset == "default"
    assert opts.network_module == "networks.lora_anima"
    assert opts.network_dim == 16
    assert opts.optimizer_type == "AdamW"
    assert opts.attn_mode == "flash"
    # method=lora's default stack: OrthoLoRA + T-LoRA both on.
    assert opts.lora.use_ortho is True
    assert opts.lora.use_timestep_mask is True
    assert opts.lora.min_rank == 8
    # Other method sub-configs are None until the user opts in.
    assert opts.postfix is None
    assert opts.chimera is None
    assert opts.easycontrol is None
    assert opts.ip_adapter is None


def test_method_postfix_requires_subconfig() -> None:
    """Picking method='postfix' without filling `postfix` is a user error."""
    with pytest.raises(pydantic.ValidationError, match="postfix"):
        AnimaLoraOptions(method="postfix")


def test_method_chimera_requires_subconfig() -> None:
    with pytest.raises(pydantic.ValidationError, match="chimera"):
        AnimaLoraOptions(method="chimera")


def test_method_easycontrol_requires_subconfig() -> None:
    with pytest.raises(pydantic.ValidationError, match="easycontrol"):
        AnimaLoraOptions(method="easycontrol")


def test_method_ip_adapter_requires_subconfig() -> None:
    with pytest.raises(pydantic.ValidationError, match="ip"):
        AnimaLoraOptions(method="ip_adapter")


def test_each_method_with_subconfig_validates_clean() -> None:
    """Each non-default method validates when its sub-config is supplied."""
    AnimaLoraOptions(method="postfix", postfix=AnimaLoraMethodPostfixConfig())
    AnimaLoraOptions(method="chimera", chimera=AnimaLoraMethodChimeraConfig())
    AnimaLoraOptions(
        method="easycontrol", easycontrol=AnimaLoraMethodEasyControlConfig()
    )
    AnimaLoraOptions(
        method="ip_adapter", ip_adapter=AnimaLoraMethodIPAdapterConfig()
    )


def test_subconfig_defaults_match_upstream_anima_lora() -> None:
    """Sub-config defaults track the values from anima_lora's methods/*.toml."""
    pf = AnimaLoraMethodPostfixConfig()
    assert pf.mode == "cond"
    assert pf.cond_hidden_dim == 1024
    assert pf.ortho_basis == "svd_te"
    assert pf.lambda_init == 0.3

    chi = AnimaLoraMethodChimeraConfig()
    assert chi.balance_w_content == pytest.approx(2e-7)
    assert chi.balance_w_freq == pytest.approx(5e-7)

    ec = AnimaLoraMethodEasyControlConfig()
    assert ec.b_cond_init == -10.0
    assert ec.cond_token_count == 4096
    # gate_lr deliberately not on EasyControl — it's IP-Adapter's knob.

    ipa = AnimaLoraMethodIPAdapterConfig()
    assert ipa.encoder == "PE-Core-L14-336"
    assert ipa.gate_lr == pytest.approx(1e-3)


def test_backend_config_accepts_anima_lora_type() -> None:
    """`BackendConfig.type` Literal includes the new value."""
    bc = BackendConfig(type="anima_lora", anima_lora=AnimaLoraOptions())
    assert bc.type == "anima_lora"
    assert bc.anima_lora is not None
    # Other two backends keep working unchanged.
    assert BackendConfig(type="kohya").anima_lora is None
    assert BackendConfig(type="diffusion-pipe").anima_lora is None


def test_backend_config_anima_lora_is_optional() -> None:
    """A fresh kohya / dp config doesn't have to set anima_lora=None explicitly."""
    bc = BackendConfig()  # default kohya
    assert bc.type == "kohya"
    assert bc.anima_lora is None


def test_select_backend_raises_not_implemented_for_anima_lora(
    tmp_path: Path,
) -> None:
    """Cut0 dispatch: explicit, actionable error before cut1/2 land.

    Without this, a user who flips backend.type to anima_lora before the
    runner is in place would just get a generic "unsupported backend".
    """
    cfg = TrainingConfig.model_validate(
        _minimal_recipe(tmp_path)
        | {"backend": {"type": "anima_lora", "animaLora": {}}}
    )
    with pytest.raises(NotImplementedError) as exc_info:
        _select_backend(cfg)
    detail = str(exc_info.value)
    assert "anima_lora" in detail
    # Tell the user where they are in the cut sequence.
    assert "cut" in detail.lower()


def test_select_backend_unchanged_for_kohya_and_dp(tmp_path: Path) -> None:
    """Adding the anima_lora branch must not regress the other two."""
    cfg_k = TrainingConfig.model_validate(_minimal_recipe(tmp_path))
    backend = _select_backend(cfg_k)
    assert backend.__class__.__name__ == "KohyaBackend"

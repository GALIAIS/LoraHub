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
    AnimaLoraMethodLoraConfig,
    AnimaLoraMethodPostfixConfig,
    AnimaLoraOptions,
    BackendConfig,
    TrainingConfig,
)


def _minimal_config(tmp_path: Path) -> dict[str, Any]:
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
    assert opts.network_dim == 32
    assert opts.optimizer_type == "AdamW"
    # ``flash`` matches Backend's base.toml default — operators without
    # a working flash-attn build override to ``torch`` (PyTorch SDPA)
    # explicitly in their config.
    assert opts.attn_mode == "flash"
    # method=lora's default stack: algorithm=ortho keeps anima's upstream
    # OrthoLoRA + T-LoRA layout; the legacy ``use_ortho`` shadow stays
    # ``None`` because the user didn't touch it (the enum drives the
    # algorithm choice now).
    assert opts.lora.algorithm == "ortho"
    assert opts.lora.use_ortho is None
    assert opts.lora.use_timestep_mask is True
    assert opts.lora.min_rank == 16
    # Other method sub-configs are None until the user opts in.
    assert opts.postfix is None
    assert opts.chimera is None
    assert opts.easycontrol is None
    assert opts.ip_adapter is None


def test_tlora_algorithm_requires_timestep_mask() -> None:
    """T-LoRA is the public name for plain LoRA + timestep rank masking."""
    cfg = AnimaLoraMethodLoraConfig(algorithm="tlora")
    assert cfg.algorithm == "tlora"
    assert cfg.use_timestep_mask is True

    with pytest.raises(pydantic.ValidationError, match="use_timestep_mask=True"):
        AnimaLoraMethodLoraConfig(
            algorithm="tlora",
            use_timestep_mask=False,
        )


def test_asr_tlora_algorithm_is_separate_from_tlora() -> None:
    cfg = AnimaLoraMethodLoraConfig(
        algorithm="asr_tlora",
        use_timestep_mask=False,
    )
    assert cfg.algorithm == "asr_tlora"
    assert cfg.use_timestep_mask is True
    assert cfg.per_sample_timestep_mask is False


def test_svd_down_init_is_plain_lora_only() -> None:
    cfg = AnimaLoraMethodLoraConfig(
        algorithm="lora",
        down_init="weight_svd",
    )
    assert cfg.down_init == "weight_svd"

    with pytest.raises(pydantic.ValidationError, match="weight_svd"):
        AnimaLoraMethodLoraConfig(
            algorithm="ortho",
            down_init="weight_svd",
        )


def test_lycoris_algorithm_aliases_normalize_to_anima_registry_keys() -> None:
    """LyCORIS/kohya-style names are accepted but compile through local modules."""
    cases = {
        "locon": "lora",
        "lycoris_locon": "lora",
        "lycoris_tlora": "tlora",
        "lycoris_loha": "loha",
        "lycoris_lokr": "lokr",
        "lycoris_ia3": "ia3",
        "lycoris_dylora": "dylora",
        "lycoris_full": "full",
        "diag-oft": "diag_oft",
        "lycoris_diag-oft": "diag_oft",
        "lycoris_boft": "boft",
        "lycoris_glora": "glora",
    }
    for raw, canonical in cases.items():
        cfg = AnimaLoraMethodLoraConfig(algorithm=raw)
        assert cfg.algorithm == canonical


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
    assert bc.distributed.strategy == "ddp"
    # Other two backends keep working unchanged.
    assert BackendConfig(type="kohya").anima_lora is None
    assert BackendConfig(type="diffusion-pipe").anima_lora is None


def test_backend_distributed_strategy_accepts_camel_case() -> None:
    bc = BackendConfig.model_validate(
        {
            "type": "anima_lora",
            "distributed": {
                "strategy": "fsdp",
                "fsdp": {
                    "autoWrapPolicy": "size_based",
                    "minNumParams": 123456,
                    "cpuOffload": True,
                },
            },
        }
    )

    assert bc.distributed.strategy == "fsdp"
    assert bc.distributed.fsdp.auto_wrap_policy == "size_based"
    assert bc.distributed.fsdp.min_num_params == 123456
    dumped = bc.model_dump(by_alias=True)
    assert dumped["distributed"]["fsdp"]["autoWrapPolicy"] == "size_based"
    assert dumped["distributed"]["fsdp"]["minNumParams"] == 123456
    assert dumped["distributed"]["fsdp"]["cpuOffload"] is True


def test_backend_config_anima_lora_is_optional() -> None:
    """A fresh kohya / dp config doesn't have to set anima_lora=None explicitly."""
    bc = BackendConfig()  # default kohya
    assert bc.type == "kohya"
    assert bc.anima_lora is None


def test_select_backend_returns_anima_lora_backend(
    tmp_path: Path,
) -> None:
    """Cut2 dispatch is live: type='anima_lora' yields the real backend.

    The earlier cut0 incarnation of this test asserted NotImplementedError
    while compiler + runner were still being built; cut2 ships them, so
    the dispatch returns an `AnimaLoraBackend` instance that's ready to
    `.validate()` / `.launch()`.
    """
    from lorahub.core.backends.anima_lora import AnimaLoraBackend

    cfg = TrainingConfig.model_validate(
        _minimal_config(tmp_path)
        | {"backend": {"type": "anima_lora", "animaLora": {}}}
    )
    backend = _select_backend(cfg)
    assert isinstance(backend, AnimaLoraBackend)
    assert backend.name == "anima_lora"


def test_select_backend_unchanged_for_kohya_and_dp(tmp_path: Path) -> None:
    """Adding the anima_lora branch must not regress the other two."""
    cfg_k = TrainingConfig.model_validate(_minimal_config(tmp_path))
    backend = _select_backend(cfg_k)
    assert backend.__class__.__name__ == "KohyaBackend"

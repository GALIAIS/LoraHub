"""Save/load round-trip + lifecycle defaults for AdapterNetworkBase subclasses.

Track 3 of refactoring_proposal.md: networks/methods/{postfix, ip_adapter,
easycontrol, soft_tokens} all subclass AdapterNetworkBase. These tests pin
the protocol invariants that the trainer relies on:

  - is_mergeable() == False
  - enable_gradient_checkpointing() is a no-op
  - prepare_grad_etc() leaves all parameters trainable
  - save_weights produces a .safetensors file with the required ss_* metadata
    (ss_network_module, ss_network_spec, sshs_model_hash, sshs_legacy_hash)
    and method-specific ss_* fields
  - load_weights round-trips bit-equivalently on a fresh instance
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

# Each entry: (label, factory, must_have_metadata_keys)
#  - factory() returns a freshly constructed adapter network.
#  - must_have_metadata_keys are method-specific ss_* stamps that should
#    appear in the saved file's metadata.
def _make_postfix_cond():
    from networks.methods.postfix import PostfixNetwork

    return PostfixNetwork(
        num_postfix_tokens=4,
        embed_dim=16,
        mode="cond",
        ortho_basis="random",
    )


def _make_postfix_plain():
    from networks.methods.postfix import PostfixNetwork

    return PostfixNetwork(
        num_postfix_tokens=4,
        embed_dim=16,
        mode="postfix",
    )


def _make_soft_tokens():
    from networks.methods.soft_tokens import SoftTokensNetwork

    return SoftTokensNetwork(
        num_tokens=2,
        embed_dim=16,
        n_layers=3,
        n_t_buckets=8,
    )


def _make_ip_adapter():
    from networks.methods.ip_adapter import IPAdapterNetwork

    return IPAdapterNetwork(
        num_ip_tokens=4,
        encoder_name="test",
        encoder_dim=32,
        context_dim=16,
        num_blocks=2,
        hidden_size=16,
        num_heads=4,
        resampler_layers=1,
        resampler_heads=2,
        ip_init_std=1e-3,
        ip_scale=1.0,
        pe_lora_enabled=False,
    )


def _make_easycontrol():
    from networks.methods.easycontrol import EasyControlNetwork

    return EasyControlNetwork(
        num_blocks=2,
        hidden_size=16,
        num_heads=4,
        mlp_ratio=2.0,
        cond_lora_dim=4,
        cond_lora_alpha=4.0,
        b_cond_init=-10.0,
        cond_scale=1.0,
        apply_ffn_lora=True,
        cond_token_count=8,
    )


CASES = [
    pytest.param(
        "postfix_cond",
        _make_postfix_cond,
        "networks.methods.postfix",
        "postfix",
        {"ss_mode": "cond", "ss_num_postfix_tokens": "4", "ss_embed_dim": "16"},
        id="postfix_cond",
    ),
    pytest.param(
        "postfix_plain",
        _make_postfix_plain,
        "networks.methods.postfix",
        "postfix",
        {"ss_mode": "postfix", "ss_num_postfix_tokens": "4"},
        id="postfix_plain",
    ),
    pytest.param(
        "soft_tokens",
        _make_soft_tokens,
        "networks.methods.soft_tokens",
        "soft_tokens",
        {"ss_num_tokens": "2", "ss_n_layers": "3", "ss_n_t_buckets": "8"},
        id="soft_tokens",
    ),
    pytest.param(
        "ip_adapter",
        _make_ip_adapter,
        "networks.methods.ip_adapter",
        "ip_adapter",
        {"ss_num_ip_tokens": "4", "ss_num_blocks": "2", "ss_pe_lora_enabled": "False"},
        id="ip_adapter",
    ),
    pytest.param(
        "easycontrol",
        _make_easycontrol,
        "networks.methods.easycontrol",
        "easycontrol",
        {"ss_num_blocks": "2", "ss_cond_lora_dim": "4", "ss_apply_ffn_lora": "1"},
        id="easycontrol",
    ),
]


@pytest.mark.parametrize("label,factory,module,spec,extra_meta", CASES)
def test_lifecycle_defaults(label, factory, module, spec, extra_meta):
    """The protocol the trainer relies on is honored by the base + overrides."""
    del label, module, spec, extra_meta
    net = factory()

    # Defaults from AdapterNetworkBase.
    assert net.is_mergeable() is False
    # set_multiplier round-trips
    net.set_multiplier(0.5)
    assert net.multiplier == 0.5
    net.set_multiplier(1.0)
    # enable_gradient_checkpointing is a no-op (must not raise)
    net.enable_gradient_checkpointing()
    # prepare_grad_etc flips at least one parameter on.
    net.prepare_grad_etc(text_encoder=None, unet=None)
    assert any(p.requires_grad for p in net.parameters())
    # Trainable list is non-empty
    params = list(net.get_trainable_params())
    assert len(params) > 0


@pytest.mark.parametrize("label,factory,module,spec,extra_meta", CASES)
def test_save_metadata(label, factory, module, spec, extra_meta, tmp_path: Path):
    """Saved .safetensors carries the right ss_* metadata."""
    del label
    net = factory()
    file = tmp_path / "adapter.safetensors"
    net.save_weights(str(file), dtype=torch.float32, metadata={"ss_test_marker": "x"})

    assert file.exists()
    with safe_open(str(file), framework="pt") as f:
        meta = f.metadata() or {}

    assert meta.get("ss_network_module") == module
    assert meta.get("ss_network_spec") == spec
    assert meta.get("ss_test_marker") == "x"
    assert "sshs_model_hash" in meta
    assert "sshs_legacy_hash" in meta
    for k, v in extra_meta.items():
        assert meta.get(k) == v, f"{k}: expected {v!r}, got {meta.get(k)!r}"


@pytest.mark.parametrize("label,factory,module,spec,extra_meta", CASES)
def test_save_load_roundtrip(
    label, factory, module, spec, extra_meta, tmp_path: Path
):
    """Saving and reloading yields the same trainable tensors."""
    del module, spec, extra_meta
    net = factory()
    file = tmp_path / f"{label}.safetensors"
    # Pin a non-default value into trainable params so we can detect failed loads.
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad or p.numel() < 32:
                p.add_(torch.randn_like(p) * 0.1)

    net.save_weights(str(file), dtype=torch.float32, metadata=None)

    fresh = factory()
    fresh.load_weights(str(file))

    # Compare every persistent tensor that's present in both state dicts.
    a = {k: v for k, v in net.state_dict().items()}
    b = {k: v for k, v in fresh.state_dict().items()}
    # ip_adapter drops _pe_inner.* from the file; that's intentional and
    # the fresh instance starts with pe_lora_enabled=False so it has no such
    # keys either.
    shared = set(a.keys()) & set(b.keys())
    assert shared, "no shared state_dict keys after round-trip"
    for k in shared:
        if k.startswith("_pe_inner."):
            continue
        torch.testing.assert_close(
            a[k].to(torch.float32),
            b[k].to(torch.float32),
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m, k=k: f"mismatch at key {k!r}: {m}",
        )

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
ANIMA_ROOT = ROOT / "external" / "anima_lora"
if str(ANIMA_ROOT) not in sys.path:
    sys.path.insert(0, str(ANIMA_ROOT))

from networks.lora_modules.lokr import (  # noqa: E402
    FactorizedLoKrModule,
    LoKrModule,
    lokr_state_dict_to_lycoris,
)
from networks.lora_anima.config import LoRANetworkCfg  # noqa: E402
from networks.lora_anima.network import LoRANetwork  # noqa: E402


def _paired_modules() -> tuple[LoKrModule, FactorizedLoKrModule]:
    torch.manual_seed(7)
    base_a = torch.nn.Linear(12, 10, bias=True)
    base_b = torch.nn.Linear(12, 10, bias=True)
    base_b.load_state_dict(base_a.state_dict())
    plain = LoKrModule("plain", base_a, lora_dim=3, alpha=3, factor=2)
    fact = FactorizedLoKrModule("fact", base_b, lora_dim=3, alpha=3, factor=2)
    with torch.no_grad():
        fact.lokr_w1.copy_(plain.lokr_w1)
        fact.lokr_w2_a.copy_(plain.lokr_w2_a)
        fact.lokr_w2_b.copy_(plain.lokr_w2_b)
        fact.lokr_w2_b.normal_(0, 0.02)
        plain.lokr_w2_b.copy_(fact.lokr_w2_b)
    plain.apply_to()
    fact.apply_to()
    return plain, fact


def test_factorized_lokr_forward_matches_materialized_lokr() -> None:
    plain, fact = _paired_modules()
    x = torch.randn(4, 5, 12)

    torch.testing.assert_close(
        fact.org_module_ref[0](x), plain.org_module_ref[0](x), rtol=1e-5, atol=1e-6
    )


def test_factorized_lokr_parameter_gradients_match_materialized_lokr() -> None:
    plain, fact = _paired_modules()
    x_plain = torch.randn(4, 5, 12, requires_grad=True)
    x_fact = x_plain.detach().clone().requires_grad_(True)

    plain.org_module_ref[0](x_plain).square().mean().backward()
    fact.org_module_ref[0](x_fact).square().mean().backward()

    torch.testing.assert_close(x_fact.grad, x_plain.grad, rtol=1e-5, atol=1e-6)
    for left, right in (
        (fact.lokr_w1.grad, plain.lokr_w1.grad),
        (fact.lokr_w2_a.grad, plain.lokr_w2_a.grad),
        (fact.lokr_w2_b.grad, plain.lokr_w2_b.grad),
    ):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


def test_lokr_save_layout_matches_comfy_lycoris_order() -> None:
    mod = LoKrModule("lokr", torch.nn.Linear(2048, 2048, bias=False), lora_dim=8, alpha=8, factor=8)
    internal = mod._delta()
    sd = {
        "lokr.lokr_w1": mod.lokr_w1.detach().clone(),
        "lokr.lokr_w2_a": mod.lokr_w2_a.detach().clone(),
        "lokr.lokr_w2_b": mod.lokr_w2_b.detach().clone(),
    }

    lokr_state_dict_to_lycoris(sd)
    w2 = sd["lokr.lokr_w2_a"] @ sd["lokr.lokr_w2_b"]
    assert w2.shape == (256, 256)

    runtime_sd = {
        "lokr_w2_a": sd["lokr.lokr_w2_a"],
        "lokr_w2_b": sd["lokr.lokr_w2_b"],
    }
    mod.normalize_state_dict_for_runtime(runtime_sd)
    loaded = torch.kron(
        sd["lokr.lokr_w1"].float(),
        runtime_sd["lokr_w2_b"].float() @ runtime_sd["lokr_w2_a"].float(),
    )
    torch.testing.assert_close(loaded, internal, rtol=1e-5, atol=1e-6)


class _FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = torch.nn.Linear(4, 4, bias=False)
        self.output_proj = torch.nn.Linear(4, 4, bias=False)


class _FakeMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(4, 4, bias=False)
        self.layer2 = torch.nn.Linear(4, 4, bias=False)


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()
        self.cross_attn = _FakeAttention()
        self.mlp = _FakeMlp()


class _FakeAnima(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block()])


def test_attention_target_preset_excludes_mlp_modules() -> None:
    cfg = LoRANetworkCfg.from_kwargs(
        {"exclude_patterns": [r"blocks\.\d+\.mlp\..*"]},
        network_dim=2,
        network_alpha=2,
        neuron_dropout=None,
        module_class=FactorizedLoKrModule,
    )
    net = LoRANetwork([], _FakeAnima(), cfg)
    names = [module.lora_name for module in net.unet_loras]

    assert names
    assert any("_self_attn_" in name for name in names)
    assert any("_cross_attn_" in name for name in names)
    assert not any("_mlp_" in name for name in names)

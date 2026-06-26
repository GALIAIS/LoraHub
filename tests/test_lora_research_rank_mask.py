from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANIMA_ROOT = ROOT / "external" / "anima_lora"
if str(ANIMA_ROOT) not in sys.path:
    sys.path.insert(0, str(ANIMA_ROOT))

_LAYER_RANK_PATH = ANIMA_ROOT / "networks" / "lora_research" / "layer_rank.py"
_LAYER_RANK_SPEC = importlib.util.spec_from_file_location(
    "lora_research_layer_rank", _LAYER_RANK_PATH
)
assert _LAYER_RANK_SPEC and _LAYER_RANK_SPEC.loader
_LAYER_RANK = importlib.util.module_from_spec(_LAYER_RANK_SPEC)
_LAYER_RANK_SPEC.loader.exec_module(_LAYER_RANK)
layer_rank_budget = _LAYER_RANK.layer_rank_budget
layer_rank_multiplier = _LAYER_RANK.layer_rank_multiplier

_STYLE_FIDELITY_PATH = ANIMA_ROOT / "networks" / "lora_research" / "style_fidelity.py"
_STYLE_FIDELITY_SPEC = importlib.util.spec_from_file_location(
    "lora_research_style_fidelity", _STYLE_FIDELITY_PATH
)
assert _STYLE_FIDELITY_SPEC and _STYLE_FIDELITY_SPEC.loader
_STYLE_FIDELITY = importlib.util.module_from_spec(_STYLE_FIDELITY_SPEC)
_STYLE_FIDELITY_SPEC.loader.exec_module(_STYLE_FIDELITY)
style_data_regime = _STYLE_FIDELITY.style_data_regime
style_rank_budget = _STYLE_FIDELITY.style_rank_budget
style_recipe = _STYLE_FIDELITY.style_recipe


def test_per_sample_rank_mask_preserves_sample_timestep():
    torch = pytest.importorskip("torch")
    from networks.lora_research.rank_mask import per_sample_rank_mask, rank_budget

    timesteps = torch.tensor([0.0, 0.5, 1.0])

    budget = rank_budget(timesteps, rank=8, min_rank=2)
    mask = per_sample_rank_mask(timesteps, rank=8, min_rank=2, target_ndim=3)

    assert budget.tolist() == [8.0, 5.0, 2.0]
    assert mask.shape == (3, 1, 8)
    assert [int(row.sum().item()) for row in mask[:, 0, :]] == [8, 5, 2]


def test_per_sample_timestep_mask_kwarg_is_hidden_and_default_off():
    pytest.importorskip("torch")
    from networks.lora_anima.config import LoRANetworkCfg
    from networks.lora_modules.lora import LoRAModule

    default_cfg = LoRANetworkCfg.from_kwargs(
        8,
        8,
        None,
        None,
        {},
        LoRAModule,
    )
    enabled_cfg = LoRANetworkCfg.from_kwargs(
        8,
        8,
        None,
        None,
        {"use_timestep_mask": "true", "per_sample_timestep_mask": "true"},
        LoRAModule,
    )

    assert default_cfg.per_sample_timestep_mask is False
    assert enabled_cfg.use_timestep_mask is True
    assert enabled_cfg.per_sample_timestep_mask is True


def test_per_sample_timestep_mask_broadcasts_over_linear_and_conv_activations():
    torch = pytest.importorskip("torch")
    from networks.lora_modules.lora import LoRAModule

    module = LoRAModule(
        "test_lora",
        torch.nn.Linear(6, 6, bias=False),
        lora_dim=4,
        alpha=4,
    )
    module._timestep_mask = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]
    )

    assert module._rank_mask_for(torch.zeros(2, 4)).shape == (2, 4)
    assert module._rank_mask_for(torch.zeros(2, 9, 4)).shape == (2, 1, 4)
    assert module._rank_mask_for(torch.zeros(2, 1, 8, 8, 4)).shape == (
        2,
        1,
        1,
        1,
        4,
    )

    conv_mask = module._rank_mask_for(torch.zeros(2, 4, 8, 8))
    assert conv_mask.shape == (2, 4, 1, 1)
    assert conv_mask[:, :, 0, 0].tolist() == [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]


def test_layer_rank_budget_keeps_attention_capacity_before_mlp():
    assert layer_rank_multiplier("net.blocks.0.cross_attn.q_proj") == 1.0
    assert layer_rank_multiplier("net.blocks.0.self_attn.q_proj") == 0.75
    assert layer_rank_multiplier("net.blocks.0.mlp.layer1") == 0.5

    assert layer_rank_budget(16, "net.blocks.0.cross_attn.q_proj", rank=16) == 16
    assert layer_rank_budget(16, "net.blocks.0.self_attn.q_proj", rank=16) == 12
    assert layer_rank_budget(16, "net.blocks.0.mlp.layer1", rank=16) == 8
    assert layer_rank_budget(1, "net.blocks.0.mlp.layer1", rank=16, min_rank=2) == 2


def test_style_fidelity_recipe_scales_capacity_by_dataset_size():
    assert style_data_regime(4) == "few"
    assert style_data_regime(32) == "standard"
    assert style_data_regime(200) == "many"

    assert style_recipe(4)["caption_dropout_rate"] > style_recipe(200)["caption_dropout_rate"]
    assert style_recipe(4)["alpha_rank_scale"] > style_recipe(200)["alpha_rank_scale"]

    assert style_rank_budget(16, "net.blocks.0.cross_attn.q_proj", rank=16, image_count=4) == 16
    assert style_rank_budget(16, "net.blocks.0.self_attn.q_proj", rank=16, image_count=4) == 10
    assert style_rank_budget(16, "net.blocks.0.mlp.layer1", rank=16, image_count=4) == 6
    assert style_rank_budget(16, "net.blocks.0.mlp.layer1", rank=16, image_count=200) == 10

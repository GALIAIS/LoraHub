from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
ANIMA_ROOT = ROOT / "external" / "anima_lora"
if str(ANIMA_ROOT) not in sys.path:
    sys.path.insert(0, str(ANIMA_ROOT))

from networks.lora_research.rank_mask import per_sample_rank_mask, rank_budget  # noqa: E402
from networks.lora_anima.config import LoRANetworkCfg  # noqa: E402
from networks.lora_modules.lora import LoRAModule  # noqa: E402


def test_per_sample_rank_mask_preserves_sample_timestep():
    timesteps = torch.tensor([0.0, 0.5, 1.0])

    budget = rank_budget(timesteps, rank=8, min_rank=2)
    mask = per_sample_rank_mask(timesteps, rank=8, min_rank=2, target_ndim=3)

    assert budget.tolist() == [8.0, 5.0, 2.0]
    assert mask.shape == (3, 1, 8)
    assert [int(row.sum().item()) for row in mask[:, 0, :]] == [8, 5, 2]


def test_per_sample_timestep_mask_kwarg_is_hidden_and_default_off():
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

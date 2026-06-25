from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy  # noqa: E402

S, D = 4, 8


def _strategy(**kw):
    return AnimaTextEncoderOutputsCachingStrategy(
        cache_to_disk=True,
        batch_size=1,
        skip_disk_cache_validity_check=True,
        **kw,
    )


def _write_pruned_variant_cache(path: Path, num_variants: int = 3) -> None:
    data = {
        "num_variants": torch.tensor(num_variants, dtype=torch.int64),
        "v0_intact": torch.tensor(1, dtype=torch.int8),
        "caption_dropout_rate": torch.tensor(0.1),
    }
    for i in range(num_variants):
        data[f"crossattn_emb_v{i}"] = torch.randn(S, D)
        data[f"t5_attn_mask_v{i}"] = torch.ones(S, dtype=torch.int32)
    save_file(data, str(path))


@pytest.mark.parametrize("flag", [False, True])
def test_pruned_crossattn_cache_loads_regardless_of_runtime_flag(tmp_path, flag):
    cache = tmp_path / "x_anima_te.safetensors"
    _write_pruned_variant_cache(cache)
    strat = _strategy(
        cache_llm_adapter_outputs=flag,
        use_shuffled_caption_variants=True,
    )

    for seed in range(10):
        random.seed(seed)
        prompt_embeds, attn_mask, t5_input_ids, _t5_mask, crossattn, _drop = (
            strat.load_outputs_npz(str(cache))
        )
        assert crossattn.shape == (S, D)
        assert prompt_embeds is None
        assert attn_mask is None
        assert t5_input_ids is None


def test_plain_cache_still_loads_when_runtime_flag_requests_adapter(tmp_path):
    cache = tmp_path / "y_anima_te.safetensors"
    save_file(
        {
            "prompt_embeds": torch.randn(S, D),
            "attn_mask": torch.ones(S, dtype=torch.int32),
            "t5_input_ids": torch.ones(S, dtype=torch.int64),
            "t5_attn_mask": torch.ones(S, dtype=torch.int32),
            "caption_dropout_rate": torch.tensor(0.1),
        },
        str(cache),
    )

    out = _strategy(cache_llm_adapter_outputs=True).load_outputs_npz(str(cache))
    assert len(out) == 5
    assert out[0].shape == (S, D)


def test_incompatible_cache_raises_actionable_error(tmp_path):
    cache = tmp_path / "z_anima_te.safetensors"
    save_file(
        {
            "t5_attn_mask": torch.ones(S, dtype=torch.int32),
            "caption_dropout_rate": torch.tensor(0.1),
        },
        str(cache),
    )

    with pytest.raises(RuntimeError, match="re-run preprocess-te"):
        _strategy().load_outputs_npz(str(cache))

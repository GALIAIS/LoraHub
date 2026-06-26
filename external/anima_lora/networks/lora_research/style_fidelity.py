"""Research-only recipe for style-focused LoRA training.

The rule is intentionally small: dataset size chooses capacity/dropout knobs,
then layer names choose where rank budget is spent.
"""

from __future__ import annotations


def style_data_regime(image_count: int) -> str:
    if image_count <= 0:
        raise ValueError(f"image_count must be > 0, got {image_count}")
    if image_count <= 8:
        return "few"
    if image_count <= 80:
        return "standard"
    return "many"


def style_recipe(image_count: int) -> dict[str, float | str]:
    regime = style_data_regime(image_count)
    if regime == "few":
        return {
            "regime": regime,
            "alpha_rank_scale": 1.5,
            "caption_dropout_rate": 0.28,
            "min_rank_ratio": 0.25,
            "spectral_weight": 0.02,
        }
    if regime == "standard":
        return {
            "regime": regime,
            "alpha_rank_scale": 1.15,
            "caption_dropout_rate": 0.18,
            "min_rank_ratio": 0.33,
            "spectral_weight": 0.01,
        }
    return {
        "regime": regime,
        "alpha_rank_scale": 0.9,
        "caption_dropout_rate": 0.1,
        "min_rank_ratio": 0.5,
        "spectral_weight": 0.005,
    }


def style_layer_multiplier(module_name: str, image_count: int) -> float:
    regime = style_data_regime(image_count)
    name = module_name.lower()

    if "cross_attn" in name:
        return 1.0
    if "self_attn" in name:
        return {"few": 0.625, "standard": 0.75, "many": 0.875}[regime]
    if ".mlp." in name or "_mlp" in name:
        return {"few": 0.375, "standard": 0.5, "many": 0.625}[regime]
    return 1.0


def style_rank_budget(
    timestep_budget: float,
    module_name: str,
    *,
    rank: int,
    image_count: int,
    min_rank: int = 1,
) -> int:
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if not 1 <= min_rank <= rank:
        raise ValueError(f"min_rank must be in [1, rank], got {min_rank}")
    scaled = round(float(timestep_budget) * style_layer_multiplier(module_name, image_count))
    return max(min_rank, min(rank, int(scaled)))


def _demo() -> None:
    assert style_data_regime(1) == "few"
    assert style_data_regime(32) == "standard"
    assert style_data_regime(200) == "many"
    assert style_rank_budget(16, "net.blocks.0.cross_attn.q_proj", rank=16, image_count=4) == 16
    assert style_rank_budget(16, "net.blocks.0.self_attn.q_proj", rank=16, image_count=4) == 10
    assert style_rank_budget(16, "net.blocks.0.mlp.layer1", rank=16, image_count=4) == 6


if __name__ == "__main__":
    _demo()

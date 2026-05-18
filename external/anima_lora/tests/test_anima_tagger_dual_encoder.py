"""Smoke tests for the Anima Tagger dual-encoder path (PE-Core + PE-Spatial).

Doesn't touch real PE checkpoints — exercises only the config / head /
encoder-registry / bucket-spec wiring with synthetic tensors. The
encoder-loader path (auto-fetch + checkpoint load) is left to manual
verification since it costs disk + network.
"""

from __future__ import annotations

import torch

from library.captioning.anima_tagger_model import (
    AnimaTaggerConfig,
    AnimaTaggerHead,
)


def test_config_legacy_single_encoder_keys_unchanged():
    """v1 config.json layout must be byte-identical for non-aux configs.

    Old anima-tagger-v1 checkpoints don't carry the aux fields; their
    config.json mustn't grow phantom keys after this change.
    """
    cfg = AnimaTaggerConfig(d_in=1024, n_tags=100, pool_kind="map")
    d = cfg.to_dict()
    # No aux-prefixed fields when has_aux is False.
    assert all("_aux" not in k for k in d.keys()), sorted(d.keys())
    # Round-trips back to the same trunk_in_dim.
    cfg2 = AnimaTaggerConfig.from_dict(d)
    assert cfg2.trunk_in_dim == cfg.trunk_in_dim
    assert not cfg2.has_aux


def test_config_dual_encoder_roundtrip():
    cfg = AnimaTaggerConfig(
        d_in=1024,
        n_tags=100,
        pool_kind="map",
        d_in_aux=768,
        n_people_counts=8,
    )
    d = cfg.to_dict()
    assert "d_in_aux" in d and d["d_in_aux"] == 768
    cfg2 = AnimaTaggerConfig.from_dict(d)
    assert cfg2.has_aux
    # 1024*(4+1+1) + 768*(4+1+1) = 6144 + 4608 = 10752
    assert cfg2.trunk_in_dim == 10752


def test_head_single_encoder_forward_shapes():
    cfg = AnimaTaggerConfig(d_in=1024, n_tags=50, n_people_counts=8, pool_kind="map")
    head = AnimaTaggerHead(cfg)
    tag, rate, people = head(torch.randn(2, 577, 1024))
    assert tag.shape == (2, 50)
    assert rate.shape == (2, 3)
    assert people.shape == (2, 8)


def test_head_dual_encoder_forward_shapes():
    cfg = AnimaTaggerConfig(
        d_in=1024,
        n_tags=50,
        n_people_counts=8,
        pool_kind="map",
        d_in_aux=768,
    )
    head = AnimaTaggerHead(cfg)
    tag, rate, people = head(
        torch.randn(2, 577, 1024),
        torch.randn(2, 1025, 768),
    )
    assert tag.shape == (2, 50)
    assert rate.shape == (2, 3)
    assert people.shape == (2, 8)


def test_head_dual_requires_aux_tensor():
    cfg = AnimaTaggerConfig(d_in=1024, n_tags=10, pool_kind="map", d_in_aux=768)
    head = AnimaTaggerHead(cfg)
    try:
        head(torch.randn(1, 577, 1024))
    except ValueError as e:
        assert "feat_aux" in str(e)
    else:
        raise AssertionError("expected ValueError for missing feat_aux")


def test_head_single_refuses_aux_tensor():
    cfg = AnimaTaggerConfig(d_in=1024, n_tags=10, pool_kind="map")
    head = AnimaTaggerHead(cfg)
    try:
        head(torch.randn(1, 577, 1024), torch.randn(1, 1025, 768))
    except ValueError as e:
        assert "no aux encoder" in str(e)
    else:
        raise AssertionError("expected ValueError for unexpected feat_aux")


def test_pe_spatial_config_present_in_registry():
    """PE-Spatial-B16-512 must be in PE_CONFIGS and produce 1025 tokens at 512px."""
    from library.models.pe import PE_CONFIGS, build_pe_vision

    assert "PE-Spatial-B16-512" in PE_CONFIGS
    cfg = PE_CONFIGS["PE-Spatial-B16-512"]
    assert (cfg.image_size, cfg.patch_size, cfg.width) == (512, 16, 768)
    assert cfg.layers == 12 and cfg.heads == 12
    assert cfg.use_cls_token is True
    assert cfg.pool_type == "none"
    assert cfg.use_ln_post is False
    assert cfg.output_dim is None

    # Shape check on an uninitialized model (skip checkpoint load — too
    # heavy for a smoke test).
    m = build_pe_vision("PE-Spatial-B16-512").eval()
    with torch.no_grad():
        feats, _pooled = m.encode(torch.randn(1, 3, 512, 512))
    assert feats.shape == (1, 1025, 768)


def test_pe_spatial_bucket_spec_aspect_aligned():
    """PE-Spatial buckets mirror PE-Core aspects so dual-cache batching is
    1:1 across encoders. Verify that pick_bucket on a sweep of source
    aspects produces matching bucket ranks for both specs."""
    import math

    from library.vision.buckets import get_bucket_spec, pick_bucket

    spec_core = get_bucket_spec("pe")
    spec_spatial = get_bucket_spec("pe_spatial")
    assert len(spec_core.buckets) == len(spec_spatial.buckets)

    # Sort bucket aspects for both — they should align at equal indices.
    aspects_core = sorted(h / w for h, w in spec_core.buckets)
    aspects_spatial = sorted(h / w for h, w in spec_spatial.buckets)
    for ac, asp in zip(aspects_core, aspects_spatial):
        assert abs(math.log(ac) - math.log(asp)) < 0.05, (ac, asp)

    # Spot-check: same source aspect → same bucket *rank* in both specs.
    for src_h, src_w in [(1024, 1024), (1024, 768), (768, 1024), (1024, 512), (512, 1024)]:
        b_core = pick_bucket(src_h, src_w, spec_core)
        b_spat = pick_bucket(src_h, src_w, spec_spatial)
        rank_core = sorted(spec_core.buckets, key=lambda hw: hw[0] / hw[1]).index(b_core)
        rank_spat = sorted(
            spec_spatial.buckets, key=lambda hw: hw[0] / hw[1]
        ).index(b_spat)
        assert rank_core == rank_spat, (src_h, src_w, b_core, b_spat)


def test_pe_spatial_encoder_registry_entry():
    from library.vision.encoders import get_encoder_info

    info = get_encoder_info("pe_spatial")
    assert info.d_enc == 768
    assert info.bucket_spec.patch == 16
    # Native bucket count = 32*32 + 1 CLS.
    assert info.t_max_tokens() >= 1024


def test_state_dict_aux_keys_present_only_when_dual():
    """Sanity: dual-encoder head's state_dict has pool_aux.* keys; single doesn't."""
    cfg_single = AnimaTaggerConfig(d_in=1024, n_tags=10, pool_kind="map")
    cfg_dual = AnimaTaggerConfig(d_in=1024, n_tags=10, pool_kind="map", d_in_aux=768)
    sd_single = AnimaTaggerHead(cfg_single).state_dict()
    sd_dual = AnimaTaggerHead(cfg_dual).state_dict()
    assert not any(k.startswith("pool_aux.") for k in sd_single.keys())
    assert any(k.startswith("pool_aux.") for k in sd_dual.keys())


# ── Mixed pool kinds (per-encoder) ────────────────────────────────────────


def test_mixed_main_mean_aux_map_forward_shapes():
    """Production target: PE-Core mean + PE-Spatial MAP."""
    cfg = AnimaTaggerConfig(
        d_in=1024,
        n_tags=50,
        n_people_counts=8,
        pool_kind="mean",
        d_in_aux=768,
        pool_kind_aux="map",
    )
    # 1024 (main mean) + 768 * (4 + 1 + 1) = 1024 + 4608 = 5632
    assert cfg.trunk_in_dim == 5632
    head = AnimaTaggerHead(cfg)
    tag, rate, people = head(
        torch.randn(2, 1024),                  # main: pre-pooled [B, D]
        torch.randn(2, 1025, 768),             # aux: [B, T_a, D_a]
    )
    assert tag.shape == (2, 50)
    assert rate.shape == (2, 3)
    assert people.shape == (2, 8)


def test_mixed_main_mean_aux_map_state_dict_layout():
    """Mixed config: only pool_aux exists; no main pool buffers."""
    cfg = AnimaTaggerConfig(
        d_in=1024, n_tags=10,
        pool_kind="mean",
        d_in_aux=768, pool_kind_aux="map",
    )
    sd = AnimaTaggerHead(cfg).state_dict()
    assert not any(k.startswith("pool.") for k in sd.keys()), \
        "main side is mean — no main MAPHead expected"
    assert any(k.startswith("pool_aux.") for k in sd.keys()), \
        "aux side is map — pool_aux MAPHead expected"


def test_mixed_main_map_aux_mean_forward_shapes():
    """Symmetric inverse — map main, mean aux. Mostly for completeness."""
    cfg = AnimaTaggerConfig(
        d_in=1024, n_tags=10,
        pool_kind="map",
        d_in_aux=768, pool_kind_aux="mean",
    )
    # 1024*6 (main map) + 768 (aux mean) = 6144 + 768 = 6912
    assert cfg.trunk_in_dim == 6912
    head = AnimaTaggerHead(cfg)
    tag, _, _ = head(
        torch.randn(2, 577, 1024),
        torch.randn(2, 768),                   # aux: pre-pooled [B, D_aux]
    )
    assert tag.shape == (2, 10)


def test_mixed_pool_kind_aux_omitted_when_inheriting():
    """Round-trip: pool_kind_aux only appears in to_dict() when it differs from main.

    Dual-MAP from default pool_kind=map should produce a config.json that's
    byte-identical to what the prior dual-MAP-only code emitted (no spurious
    pool_kind_aux key).
    """
    cfg = AnimaTaggerConfig(d_in=1024, n_tags=10, pool_kind="map", d_in_aux=768)
    d = cfg.to_dict()
    assert "pool_kind_aux" not in d, (
        "pool_kind_aux should be omitted when inheriting pool_kind"
    )
    # Round-trip preserves effective_pool_kind_aux="map".
    cfg2 = AnimaTaggerConfig.from_dict(d)
    assert cfg2.effective_pool_kind_aux == "map"


def test_mixed_pool_kind_aux_emitted_when_differs():
    cfg = AnimaTaggerConfig(
        d_in=1024, n_tags=10,
        pool_kind="mean",
        d_in_aux=768, pool_kind_aux="map",
    )
    d = cfg.to_dict()
    assert d.get("pool_kind_aux") == "map"
    cfg2 = AnimaTaggerConfig.from_dict(d)
    assert cfg2.pool_kind == "mean"
    assert cfg2.pool_kind_aux == "map"
    assert cfg2.effective_pool_kind_aux == "map"
    assert cfg2.trunk_in_dim == cfg.trunk_in_dim


def test_mixed_main_mean_rejects_map_input_to_main():
    """Helpful error when caller passes [B, T, D] to a mean-pool side."""
    cfg = AnimaTaggerConfig(
        d_in=1024, n_tags=10,
        pool_kind="mean",
        d_in_aux=768, pool_kind_aux="map",
    )
    head = AnimaTaggerHead(cfg)
    try:
        head(torch.randn(2, 577, 1024), torch.randn(2, 1025, 768))
    except ValueError as e:
        assert "main side" in str(e) and "pre-pooled" in str(e)
    else:
        raise AssertionError("expected ValueError on rank-3 main input with mean pool")


def test_dual_dataset_class_supports_per_side_pool_kind():
    """Quick API check on CachedDualDataset — verifies it accepts the per-side
    pool_kind args and rejects bad combinations without needing a real cache."""
    from library.captioning.anima_tagger_data import CachedDualDataset

    # Bad pool_kind value should raise immediately (before any disk access).
    try:
        CachedDualDataset.__init__.__annotations__  # touch to ensure import works
    except Exception as e:
        raise AssertionError(f"import failed: {e}")
    # Manually invoke the validation by constructing with empty-ish args.
    # We don't have a real manifest so this would error on disk if validation
    # didn't fire first — exact error type is what we're checking.
    from library.captioning.anima_tagger_data import TaggerManifest
    fake_manifest = TaggerManifest(
        stems=[], image_paths=[], tag_indices=[], rating_indices=[],
        people_count_indices=[], train_stems=[], val_stems=[],
        n_tags=0, n_ratings=0, n_people_counts=0,
    )
    from pathlib import Path
    try:
        CachedDualDataset(
            fake_manifest,
            Path("/nonexistent/main"), "weird", None,
            Path("/nonexistent/aux"), "map", None,
        )
    except ValueError as e:
        assert "pool_kind" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown pool_kind")

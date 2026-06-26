from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


ROOT = Path(__file__).resolve().parents[1]
ANIMA_ROOT = ROOT / "external" / "anima_lora"
if str(ANIMA_ROOT) not in sys.path:
    sys.path.insert(0, str(ANIMA_ROOT))

from library.training.ema import _named_trainables  # noqa: E402
from networks.methods.full_finetune import FullFinetuneNetwork  # noqa: E402


def test_full_finetune_tracks_external_dit_params_and_saves_model(monkeypatch, tmp_path):
    dit = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    te = nn.Linear(2, 2)
    network = FullFinetuneNetwork(text_encoders=[te], unet=dit)

    network.apply_to(te, dit, apply_text_encoder=False, apply_unet=True)
    params, descriptions = network.prepare_optimizer_params_with_multiple_te_lrs(
        text_encoder_lr=None,
        unet_lr=None,
        default_lr=1e-6,
    )

    assert descriptions == ["anima_dit"]
    assert params[0]["lr"] == 1e-6
    assert [id(p) for p in params[0]["params"]] == [id(p) for p in dit.parameters()]
    assert all(not p.requires_grad for p in te.parameters())

    named = _named_trainables(network)
    assert len(named) == len(list(dit.parameters()))
    assert named[0][0] == "trainable_0"

    saved = {}

    def fake_save(path, state_dict, metadata, dtype):
        saved["path"] = path
        saved["keys"] = sorted(state_dict.keys())
        saved["metadata"] = dict(metadata)
        saved["dtype"] = dtype

    monkeypatch.setattr(
        "networks.methods.full_finetune.anima_weights.save_anima_model",
        fake_save,
    )
    network.save_weights(tmp_path / "full.safetensors", torch.float32, {"base": "x"})

    assert saved["path"].endswith("full.safetensors")
    assert saved["metadata"]["ss_network_module"] == "networks.methods.full_finetune"
    assert saved["metadata"]["ss_full_finetune"] == "true"
    assert saved["dtype"] is torch.float32
    assert saved["keys"] == sorted(dit.state_dict().keys())

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lorahub.api.jobs_helpers import _select_backend
from lorahub.api.jobs_helpers.lifecycle import _apply_settings_gpu_dispatch_default
from lorahub.api.settings import Settings, probe_ai_toolkit_backend
from lorahub.api.terminal_runner import resolve_backend_session
from lorahub.core.backends._common import bootstrap as common_bootstrap
from lorahub.core.backends.ai_toolkit import backend as ai_backend
from lorahub.core.backends.ai_toolkit import bootstrap as ai_bootstrap
from lorahub.core.backends.ai_toolkit import installer
from lorahub.core.backends.ai_toolkit.backend import AIToolkitBackend
from lorahub.core.backends.ai_toolkit.compiler import CompilationError, compile_config
from lorahub.core.backends.ai_toolkit.parser import parse_line
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType


def _cfg() -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {
                "type": "ai_toolkit",
                "extraArgs": {
                    "model.name_or_path": "krea/Krea-2-Turbo",
                    "model.assistant_lora_path": "adapter.safetensors",
                },
            },
        }
    )


def test_ai_toolkit_compiler_emits_krea2_yaml(tmp_path: Path) -> None:
    argv, files = compile_config(_cfg(), tmp_path)
    assert argv == [str(tmp_path / "_lorahub_ai_toolkit.yaml")]
    data = yaml.safe_load(next(iter(files.values())))
    process = data["config"]["process"][0]
    assert process["model"]["arch"] == "krea2"
    assert process["model"]["name_or_path"] == "krea/Krea-2-Turbo"
    assert process["model"]["assistant_lora_path"] == "adapter.safetensors"


def test_ai_toolkit_compiler_rejects_blank_dataset_source(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg.dataset.source = None

    with pytest.raises(CompilationError, match="requires dataset.source"):
        compile_config(cfg, tmp_path)


def test_ai_toolkit_compiler_emits_supported_krea2_network_types(tmp_path: Path) -> None:
    for network_type in ("lora", "dora", "loha", "lokr", "lorm"):
        cfg = TrainingConfig.model_validate(
            {
                "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
                "dataset": {"source": ".", "resolution": [1024, 1024]},
                "network": {"type": network_type, "rank": 12, "alpha": 12},
                "backend": {"type": "ai_toolkit"},
            }
        )

        _, files = compile_config(cfg, tmp_path)
        data = yaml.safe_load(next(iter(files.values())))
        network = data["config"]["process"][0]["network"]

        assert network["type"] == network_type
        assert network["linear"] == 12
        assert network["linear_alpha"] == 12
        if network_type == "lorm":
            assert network["lorm"]["extract_mode"] == "fixed"
            assert network["lorm"]["extract_mode_param"] == 12


def test_ai_toolkit_compiler_rejects_unsupported_krea2_network_type(
    tmp_path: Path,
) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "network": {"type": "locon"},
            "backend": {"type": "ai_toolkit"},
        }
    )

    with pytest.raises(ValueError, match="ai_toolkit krea2 supports"):
        compile_config(cfg, tmp_path)


def test_ai_toolkit_compiler_maps_visible_sampling_fields(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [768, 1024]},
            "backend": {"type": "ai_toolkit"},
            "sampling": {
                "enabled": True,
                "everyNEpochs": 2,
                "everyNSteps": 250,
                "resolution": [832, 1216],
                "seed": 123,
                "inferenceSteps": 28,
                "inferenceCfg": 4.5,
                "prompts": [
                    {
                        "prompt": "test prompt",
                        "negative": "low quality",
                        "width": 768,
                        "height": 1152,
                        "seed": 321,
                        "cfg": 3.5,
                        "steps": 18,
                    }
                ],
            },
        }
    )

    _, files = compile_config(cfg, tmp_path)
    data = yaml.safe_load(next(iter(files.values())))
    process = data["config"]["process"][0]

    assert process["train"]["disable_sampling"] is False
    assert process["sample"] == {
        "sample_every": 250,
        "sample_every_n_epochs": 2,
        "sampler": "flowmatch",
        "width": 832,
        "height": 1216,
        "neg": "",
        "seed": 123,
        "sample_steps": 28,
        "guidance_scale": 4.5,
        "format": "jpg",
        "walk_seed": False,
        "network_multiplier": 1.0,
        "prompts": ["test prompt"],
        "samples": [
            {
                "prompt": "test prompt",
                "neg": "low quality",
                "width": 768,
                "height": 1152,
                "seed": 321,
                "guidance_scale": 3.5,
                "sample_steps": 18,
            }
        ],
    }


def test_ai_toolkit_compiler_supports_epoch_only_sampling(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {"type": "ai_toolkit"},
            "sampling": {"enabled": True, "everyNEpochs": 3},
        }
    )

    _, files = compile_config(cfg, tmp_path)
    data = yaml.safe_load(next(iter(files.values())))
    sample = data["config"]["process"][0]["sample"]

    assert sample["sample_every_n_epochs"] == 3
    assert sample["sample_every"] is None


def test_ai_toolkit_compiler_supports_step_only_sampling(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {"type": "ai_toolkit"},
            "sampling": {
                "enabled": True,
                "everyNEpochs": None,
                "everyNSteps": 250,
            },
        }
    )

    _, files = compile_config(cfg, tmp_path)
    sample = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]["sample"]

    assert sample["sample_every_n_epochs"] is None
    assert sample["sample_every"] == 250


def test_ai_toolkit_no_worker_loader_omits_prefetch_factor(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {
                "type": "ai_toolkit",
                "aiToolkit": {
                    "dataset": {"numWorkers": 0, "prefetchFactor": 8},
                },
            },
        }
    )

    assert cfg.backend.ai_toolkit is not None
    assert cfg.backend.ai_toolkit.dataset.prefetch_factor is None
    _, files = compile_config(cfg, tmp_path)
    dataset = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]["datasets"][0]

    assert dataset["num_workers"] == 0
    assert "prefetch_factor" not in dataset


@pytest.mark.parametrize(
    ("at_first", "skip_first_sample"),
    [(True, False), (False, True)],
)
def test_ai_toolkit_compiler_honors_sampling_at_first(
    at_first: bool,
    skip_first_sample: bool,
    tmp_path: Path,
) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "sampling": {"atFirst": at_first},
            "backend": {"type": "ai_toolkit"},
        }
    )

    _, files = compile_config(cfg, tmp_path)
    train = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]["train"]

    assert train["skip_first_sample"] is skip_first_sample


def test_ai_toolkit_compiler_uses_epochs_with_max_steps_as_cap(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "schedule": {
                "epochs": 10,
                "maxSteps": 200000,
                "batchSize": 1,
                "gradAccum": 4,
            },
            "backend": {"type": "ai_toolkit"},
        }
    )

    _, files = compile_config(cfg, tmp_path)
    train = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]["train"]

    assert train["epochs"] == 10
    assert train["max_steps"] == 200000
    assert "steps" not in train


def test_ai_toolkit_epoch_schedule_defers_cosine_length_to_runtime(
    tmp_path: Path,
) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "schedule": {"epochs": 10, "maxSteps": 200000},
            "optimizer": {"schedule": "cosine_with_restarts"},
            "backend": {"type": "ai_toolkit"},
        }
    )

    _, files = compile_config(cfg, tmp_path)
    train = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]["train"]

    assert "total_iters" not in train["lr_scheduler_params"]


def test_ai_toolkit_scheduler_default_survives_normalized_round_trip(
    tmp_path: Path,
) -> None:
    original = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {"type": "ai_toolkit"},
        }
    )
    normalized = original.model_dump(mode="json", by_alias=True)
    reloaded = TrainingConfig.model_validate(normalized)

    _, files = compile_config(reloaded, tmp_path)
    train = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]["train"]

    assert train["lr_scheduler"] == "constant"


def test_ai_toolkit_rejects_incompatible_text_encoder_qtype() -> None:
    with pytest.raises(ValueError, match="qtypeTextEncoder"):
        TrainingConfig.model_validate(
            {
                "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
                "dataset": {"source": ".", "resolution": [1024, 1024]},
                "backend": {
                    "type": "ai_toolkit",
                    "aiToolkit": {
                        "model": {"qtypeTextEncoder": "float8"},
                    },
                },
            }
        )


def test_ai_toolkit_compiler_maps_dedicated_options(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {
                "source": ".",
                "resolution": [832, 1216],
                "caption": {"ext": ".txt", "dropRate": 0.1},
            },
            "network": {
                "type": "lokr",
                "rank": 8,
                "alpha": 8,
                "networkDropout": 0.1,
                "rankDropout": 0.2,
                "moduleDropout": 0.3,
                "initFrom": "seed.safetensors",
            },
            "optimizer": {
                "type": "adamw8bit",
                "lr": {"unet": 2e-5},
                "schedule": "constant_with_warmup",
                "warmupSteps": 50,
                "weightDecay": 0.01,
                "maxGradNorm": 0.5,
            },
            "schedule": {"maxSteps": 777, "batchSize": 2, "gradAccum": 3},
            "output": {
                "saveEveryNEpochs": 2,
                "saveEveryNSteps": None,
                "saveLastNSteps": 3,
            },
            "backend": {
                "type": "ai_toolkit",
                "aiToolkit": {
                    "model": {
                        "quantize": False,
                        "quantizeTextEncoder": False,
                        "lowVram": True,
                        "vaePath": "Qwen/Qwen-Image",
                    },
                    "dataset": {
                        "resolutions": [512, 768],
                        "shuffleTokens": True,
                        "tokenDropoutRate": 0.2,
                        "cacheTextEmbeddings": True,
                    },
                    "network": {"lokrFactor": 8},
                    "train": {
                        "contentOrStyle": "style",
                        "timestepType": "linear",
                        "lossType": "pseudo_huber",
                    },
                    "sample": {"format": "png", "networkMultiplier": 0.8},
                    "logging": {"logEvery": 5, "projectName": "test-project"},
                },
            },
        }
    )

    _, files = compile_config(cfg, tmp_path)
    process = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]

    assert process["model"]["quantize"] is False
    assert process["model"]["quantize_te"] is False
    assert process["model"]["low_vram"] is True
    assert process["model"]["model_kwargs"]["vae_path"] == "Qwen/Qwen-Image"
    assert process["datasets"][0]["resolution"] == [512, 768]
    assert process["datasets"][0]["shuffle_tokens"] is True
    assert process["datasets"][0]["token_dropout_rate"] == 0.2
    assert process["network"]["pretrained_lora_path"] == "seed.safetensors"
    assert process["network"]["lokr_factor"] == 8
    assert process["network"]["network_kwargs"] == {
        "rank_dropout": 0.2,
        "module_dropout": 0.3,
    }
    assert process["train"]["steps"] == 777
    assert process["train"]["lr_scheduler"] == "constant_with_warmup"
    assert process["train"]["lr_scheduler_params"]["num_warmup_steps"] == 50
    assert process["train"]["content_or_style"] == "style"
    assert process["train"]["timestep_type"] == "linear"
    assert process["train"]["loss_type"] == "pseudo_huber"
    assert process["save"]["save_every"] is None
    assert process["save"]["save_every_n_epochs"] == 2
    assert process["sample"]["format"] == "png"
    assert process["sample"]["network_multiplier"] == 0.8
    assert process["logging"]["log_every"] == 5


def test_ai_toolkit_compiler_converts_legacy_width_height_to_pixel_budget(
    tmp_path: Path,
) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [832, 1216]},
            "backend": {"type": "ai_toolkit"},
        }
    )

    _, files = compile_config(cfg, tmp_path)
    datasets = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]["datasets"]

    assert len(datasets) == 1
    assert datasets[0]["resolution"] == 1008


def test_ai_toolkit_extra_args_can_override_boolean_with_false(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {
                "type": "ai_toolkit",
                "extraArgs": {"model.quantize": False},
            },
        }
    )

    _, files = compile_config(cfg, tmp_path)
    model = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]["model"]

    assert model["quantize"] is False


@pytest.mark.parametrize(
    ("network", "message"),
    [
        ({"targetUnet": False}, "requires network.target_unet=true"),
        ({"targetTextEncoder": True}, "does not support text-encoder"),
    ],
)
def test_ai_toolkit_rejects_unsupported_training_targets(
    network: dict[str, bool],
    message: str,
    tmp_path: Path,
) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "network": network,
            "backend": {"type": "ai_toolkit"},
        }
    )

    with pytest.raises(ValueError, match=message):
        compile_config(cfg, tmp_path)


@pytest.mark.parametrize(
    ("optimizer", "expected"),
    [
        ("adamw", {"betas": [0.9, 0.999], "weight_decay": 0.01}),
        ("adagrad", {"weight_decay": 0.01}),
        ("automagic3", {"beta2": 0.999, "weight_decay": 0.01}),
    ],
)
def test_ai_toolkit_emits_only_supported_optimizer_defaults(
    optimizer: str,
    expected: dict[str, object],
    tmp_path: Path,
) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "optimizer": {"type": optimizer, "weightDecay": 0.01},
            "backend": {"type": "ai_toolkit"},
        }
    )

    _, files = compile_config(cfg, tmp_path)
    process = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]

    assert process["train"]["optimizer_params"] == expected


def test_ai_toolkit_forces_single_gpu_dispatch() -> None:
    cfg = _cfg()
    cfg.backend.gpu_dispatch.mode = "distributed"
    cfg.backend.gpu_dispatch.num_gpus = 2

    _apply_settings_gpu_dispatch_default(cfg)

    assert cfg.backend.gpu_dispatch.mode == "one-job-per-gpu"
    assert cfg.backend.gpu_dispatch.num_gpus is None


@pytest.mark.parametrize(
    "name",
    [
        "ai_toolkit_krea2",
        "ai_toolkit_krea2_dora",
        "ai_toolkit_krea2_loha",
        "ai_toolkit_krea2_lokr",
        "ai_toolkit_krea2_lorm",
    ],
)
def test_builtin_ai_toolkit_templates_compile(name: str, tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw.pop("_template", None)
    raw.pop("_placeholders", None)
    # Built-in templates intentionally leave the user-owned dataset path
    # blank. Compiler tests provide a concrete path because launch-time
    # compilation correctly rejects incomplete form state.
    raw["dataset"]["source"] = str(tmp_path / "dataset")

    cfg = TrainingConfig.model_validate(raw)
    _, files = compile_config(cfg, tmp_path)
    process = yaml.safe_load(next(iter(files.values())))["config"]["process"][0]

    assert process["train"]["epochs"] == 10
    assert "steps" not in process["train"]
    assert process["datasets"][0]["resolution"] == 1024


def test_ai_toolkit_epoch_sampling_detects_crossed_boundaries() -> None:
    cadence_path = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "ai_toolkit"
        / "toolkit"
        / "training_cadence.py"
    )
    spec = importlib.util.spec_from_file_location("ai_toolkit_training_cadence", cadence_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.epoch_cadence_due(0, 1, 1) is True
    assert module.epoch_cadence_due(1, 2, 2) is True
    assert module.epoch_cadence_due(2, 3, 2) is False
    assert module.epoch_cadence_due(2, 6, 2) is True
    assert module.epoch_cadence_due(3, 3, 1) is False
    assert module.epoch_cadence_due(0, 1, None) is False
    assert module.epoch_training_plan(
        epochs=10,
        batches_per_epoch=860,
        gradient_accumulation=4,
        max_steps=None,
    ) == (215, 2150)
    assert module.epoch_training_plan(
        epochs=10,
        batches_per_epoch=860,
        gradient_accumulation=4,
        max_steps=1000,
    ) == (215, 1000)


def test_ai_toolkit_compiler_honors_disabled_sampling(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {"type": "ai_toolkit"},
            "sampling": {"enabled": False},
        }
    )

    _, files = compile_config(cfg, tmp_path)
    data = yaml.safe_load(next(iter(files.values())))
    process = data["config"]["process"][0]

    assert process["train"]["disable_sampling"] is True


def test_ai_toolkit_parser_recognizes_step_and_checkpoint() -> None:
    step = parse_line("training 12/100 loss: 0.1234")
    assert step is not None
    assert step.type is EventType.step
    assert step.payload["step"] == 12
    assert step.payload["total_steps"] == 100
    assert step.payload["loss"] == 0.1234

    saved = parse_line("Saved checkpoint to C:/runs/foo.safetensors")
    assert saved is not None
    assert saved.type is EventType.checkpoint_saved
    assert saved.payload["path"] == "C:/runs/foo.safetensors"


def test_ai_toolkit_compiler_sanitizes_sample_prompt_types(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {
                "type": "ai_toolkit",
                "extraArgs": {
                    "sample.neg": True,
                    "sample.prompts": True,
                    "sample.samples": True,
                },
            },
        }
    )

    _, files = compile_config(cfg, tmp_path)
    data = yaml.safe_load(next(iter(files.values())))
    sample = data["config"]["process"][0]["sample"]

    assert sample["neg"] == ""
    assert sample["prompts"] == ["a high quality image"]
    assert "samples" not in sample


def test_ai_toolkit_parser_summarizes_progress_lines() -> None:
    ev = parse_line("Caching latents to disk:  25%|██▌| 70/280 [00:23<01:03, 3.31it/s]")

    assert ev is not None
    assert ev.type is EventType.cache_progress
    assert ev.payload["phase"] == "latents"
    assert ev.payload["done"] == 70
    assert ev.payload["total"] == 280

    text_ev = parse_line(
        "Caching text embeddings: 50%|█████| 10/20 [00:02<00:02, 5.00it/s]"
    )
    assert text_ev is not None
    assert text_ev.type is EventType.cache_progress
    assert text_ev.payload["phase"] == "text_encoder"


def test_ai_toolkit_parser_reads_tqdm_training_steps() -> None:
    ev = parse_line(
        "krea2_real:  27%|██▋| 27/100 [03:04<08:18, 6.83s/it, "
        "epoch: 3 lr: 1.0e-04 loss: 1.964e-01 snr: 2.5 grad_norm: 0.75]"
    )

    assert ev is not None
    assert ev.type is EventType.step
    assert ev.payload["step"] == 27
    assert ev.payload["total_steps"] == 100
    assert ev.payload["loss"] == 0.1964
    assert ev.payload["lr"] == 1.0e-04
    assert ev.payload["epoch"] == 3
    assert ev.payload["snr"] == 2.5
    assert ev.payload["grad_norm"] == 0.75
    assert ev.payload["rate"] == "6.83s/it"
    assert ev.payload["eta"] == "08:18"


def test_ai_toolkit_parser_keeps_checkpoint_step() -> None:
    ev = parse_line("Saved checkpoint at step 250 to /tmp/krea2_000000250.safetensors")

    assert ev is not None
    assert ev.type is EventType.checkpoint_saved
    assert ev.payload["step"] == 250
    assert ev.payload["path"] == "/tmp/krea2_000000250.safetensors"

    legacy = parse_line("Saved checkpoint to /tmp/krea2_000000500.safetensors")
    assert legacy is not None
    assert legacy.payload["step"] == 500


def test_ai_toolkit_sample_scan_emits_relative_sample_ready(tmp_path: Path) -> None:
    sample_dir = tmp_path / "ai_toolkit_output" / "krea2_lora" / "samples"
    sample_dir.mkdir(parents=True)
    sample = sample_dir / "preview.jpg"
    sample.write_bytes(b"jpg")
    seen: set[str] = set()

    events = ai_backend._scan_new_samples(
        tmp_path / "ai_toolkit_output", tmp_path, seen, job_id="job-1"
    )

    assert len(events) == 1
    assert events[0].type is EventType.sample_ready
    assert events[0].job_id == "job-1"
    assert (
        events[0].payload["path"]
        == "ai_toolkit_output/krea2_lora/samples/preview.jpg"
    )
    assert (
        ai_backend._scan_new_samples(tmp_path / "ai_toolkit_output", tmp_path, seen)
        == []
    )


def test_ai_toolkit_installer_adds_matching_torchaudio(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(installer._common, "install_torch", lambda plan, progress=None: None)
    monkeypatch.setattr(
        installer,
        "pip_install_with_torch_index_fallback",
        lambda _plan, args, *, step, progress=None: calls.append((step, args)),
    )

    plan = installer.BootstrapPlan(
        target=tmp_path / "ai_toolkit",
        torch_version="2.7.1+cu128",
        torchvision_version="0.22.1+cu128",
        cuda_version="cu128",
    )
    installer.install_torch(plan)

    assert calls == [
        (
            "install torchaudio==2.7.1+cu128 (cu128)",
            ["torchaudio==2.7.1+cu128", "--index-url", plan.torch_index],
        )
    ]


def test_ai_toolkit_launch_defaults_hf_cache_under_models(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    repo = tmp_path / "repo"
    repo.mkdir()
    py = tmp_path / "python"
    py.write_text("", encoding="utf-8")

    class FakeRunner:
        pid = 123

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self) -> None:
            captured["started"] = True

        def stop(self, *, graceful: bool = True) -> None:
            captured["stopped"] = graceful

        def wait(self, timeout=None):
            return SimpleNamespace(returncode=0)

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(
        ai_backend._bootstrap,
        "resolve",
        lambda **_: SimpleNamespace(repo_path=repo, python_executable=py),
    )
    monkeypatch.setattr(ai_backend, "AIToolkitRunner", FakeRunner)
    monkeypatch.setattr(ai_backend, "project_root", lambda: tmp_path)

    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": str(tmp_path), "resolution": [1024, 1024]},
            "backend": {"type": "ai_toolkit"},
        }
    )
    AIToolkitBackend().launch(cfg, tmp_path / "run", lambda _ev: None)

    assert captured["started"] is True
    assert captured["env"] == {
        "HF_HOME": str(tmp_path / "models" / "huggingface"),
        "HF_HUB_CACHE": str(tmp_path / "models" / "huggingface" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(tmp_path / "models" / "huggingface" / "hub"),
        "MODELS_PATH": str(tmp_path / "models"),
    }


def test_select_backend_returns_ai_toolkit_backend() -> None:
    backend = _select_backend(_cfg())
    assert isinstance(backend, AIToolkitBackend)
    assert backend.name == "ai_toolkit"


def test_ai_toolkit_validate_rejects_conditioning_dataset_fields() -> None:
    cfg = _cfg()
    cfg.dataset.conditioning_dir = Path("reference")

    issues = AIToolkitBackend().validate(cfg)

    assert any(
        issue.field == "dataset.subsets" and "conditioning" in issue.message
        for issue in issues
    )


def test_ai_toolkit_validate_rejects_uncompiled_dataset_fields() -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {
                "source": ".",
                "valSplit": 0.1,
                "subsets": [
                    {
                        "path": ".",
                        "captionPrefix": "trigger",
                        "arBuckets": [1.0],
                    }
                ],
            },
            "backend": {"type": "ai_toolkit"},
        }
    )

    issues = AIToolkitBackend().validate(cfg)

    assert any(issue.field == "dataset.valSplit" for issue in issues)
    assert any(issue.field == "dataset.subsets" and "captionPrefix" in issue.message for issue in issues)


def test_ai_toolkit_default_repo_uses_project_root(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "external" / "ai_toolkit"
    repo.mkdir(parents=True)
    (repo / "run.py").write_text("", encoding="utf-8")
    monkeypatch.delenv("LORAHUB_AI_TOOLKIT_REPO", raising=False)
    monkeypatch.setattr(ai_bootstrap, "project_root", lambda: tmp_path)

    assert ai_bootstrap.default_repo_path() == repo


def test_backend_python_probe_rejects_dll_init_failure(
    monkeypatch, tmp_path: Path
) -> None:
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    monkeypatch.setattr(
        common_bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0xC0000142,
            stdout="",
            stderr="",
        ),
    )

    with pytest.raises(common_bootstrap.BootstrapError, match="0xC0000142"):
        common_bootstrap.check_python(python)


def test_probe_ai_toolkit_backend_reports_vendored_ready(tmp_path: Path) -> None:
    repo = tmp_path / "ai_toolkit"
    (repo / "toolkit").mkdir(parents=True)
    (repo / "extensions_built_in" / "sd_trainer").mkdir(parents=True)
    (repo / "run.py").write_text("", encoding="utf-8")
    (repo / "toolkit" / "job.py").write_text("", encoding="utf-8")
    (repo / "extensions_built_in" / "sd_trainer" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (repo / "requirements.txt").write_text("", encoding="utf-8")
    py = repo / ".venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")

    status = probe_ai_toolkit_backend(
        Settings(ai_toolkit_repo_path=str(repo), ai_toolkit_python=sys.executable)
    )

    assert status["id"] == "ai_toolkit"
    assert status["repo_ok"] is True
    assert status["python_ok"] is True
    assert status["venv_detected"] is True
    assert status["ready"] is True


def test_probe_ai_toolkit_backend_rejects_python_that_cannot_start(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "ai_toolkit"
    (repo / "toolkit").mkdir(parents=True)
    (repo / "extensions_built_in" / "sd_trainer").mkdir(parents=True)
    for name in ("run.py", "toolkit/job.py", "extensions_built_in/sd_trainer/__init__.py"):
        (repo / name).write_text("", encoding="utf-8")
    python = repo / "venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    monkeypatch.setattr(
        common_bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0xC0000142,
            stdout="",
            stderr="",
        ),
    )

    status = probe_ai_toolkit_backend(
        Settings(ai_toolkit_repo_path=str(repo), ai_toolkit_python=str(python))
    )

    assert status["python_ok"] is False
    assert "0xC0000142" in status["python_error"]
    assert status["ready"] is False


def test_terminal_resolves_ai_toolkit_session(tmp_path: Path) -> None:
    repo = tmp_path / "ai_toolkit"
    repo.mkdir()
    py = repo / ".venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")

    session = resolve_backend_session(
        "ai_toolkit",
        Settings(ai_toolkit_repo_path=str(repo), ai_toolkit_python=str(py)),
    )

    assert session.backend_id == "ai_toolkit"
    assert session.display_name == "AI Toolkit"
    assert session.repo_path == repo
    assert session.python_path == py
    assert session.ready is True

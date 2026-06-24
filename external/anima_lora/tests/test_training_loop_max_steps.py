from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace


class _Value:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class _Accelerator:
    sync_gradients = True
    is_main_process = True
    device = "cpu"

    def __init__(self) -> None:
        self.waits = 0

    def print(self, *_args, **_kwargs) -> None:
        pass

    def unwrap_model(self, network):
        return network

    def wait_for_everyone(self) -> None:
        self.waits += 1


class _Network:
    def on_epoch_start(self, *_args, **_kwargs) -> None:
        pass


def _state(**overrides):
    state = SimpleNamespace(
        args=SimpleNamespace(max_train_steps=4000, output_dir="."),
        accelerator=_Accelerator(),
        saver=SimpleNamespace(
            maybe_save_epoch=lambda *_args, **_kwargs: None,
            maybe_save_resumable=lambda *_args, **_kwargs: None,
            maybe_save_step=lambda *_args, **_kwargs: None,
        ),
        network=_Network(),
        unet=None,
        text_encoder=None,
        vae=None,
        tokenizers=None,
        training_model=None,
        train_dataloader=[object()],
        current_epoch=_Value(),
        current_step=_Value(),
        num_train_epochs=3,
        epoch_to_start=0,
        initial_step=0,
        metadata={},
        global_step=3999,
        progress_bar=SimpleNamespace(update=lambda *_args, **_kwargs: None),
        optimizer_train_fn=lambda: None,
        optimizer_eval_fn=lambda: None,
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _module(name: str, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_loop_module(monkeypatch):
    loop_path = Path(__file__).resolve().parents[1] / "library" / "training" / "loop.py"
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _module(
            "torch",
            Tensor=object,
            distributed=_module(
                "torch.distributed",
                is_available=lambda: False,
                is_initialized=lambda: False,
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "accelerate", _module("accelerate", Accelerator=object))
    monkeypatch.setitem(
        sys.modules,
        "accelerate.utils",
        _module("accelerate.utils", send_to_device=lambda batch, _device: batch),
    )
    monkeypatch.setitem(sys.modules, "tqdm", _module("tqdm", tqdm=lambda *args, **_kwargs: args[0]))
    monkeypatch.setitem(sys.modules, "library", _module("library"))
    monkeypatch.setitem(sys.modules, "library.train_util", _module("library.train_util"))
    monkeypatch.setitem(
        sys.modules,
        "library.datasets",
        _module("library.datasets", LossRecorder=object),
    )
    monkeypatch.setitem(sys.modules, "library.runtime", _module("library.runtime"))
    monkeypatch.setitem(
        sys.modules,
        "library.runtime.device",
        _module("library.runtime.device", clean_memory_on_device=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "library.training", _module("library.training"))
    monkeypatch.setitem(
        sys.modules,
        "library.training.checkpoints",
        _module("library.training.checkpoints", CheckpointSaver=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "library.training.contexts",
        _module("library.training.contexts", TrainCtx=object, ValCtx=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "library.training.method_adapter",
        _module("library.training.method_adapter", StepCtx=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "library.training.metrics",
        _module(
            "library.training.metrics",
            MetricContext=object,
            collect_metrics=lambda *_args, **_kwargs: {},
        ),
    )

    spec = importlib.util.spec_from_file_location("_lorahub_training_loop_under_test", loop_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_distributed_any_returns_local_value_without_distributed(monkeypatch) -> None:
    loop = _load_loop_module(monkeypatch)

    assert loop._distributed_any(_Accelerator(), True) is True
    assert loop._distributed_any(_Accelerator(), False) is False


def test_distributed_any_reduces_across_ranks(monkeypatch) -> None:
    loop = _load_loop_module(monkeypatch)
    calls: list[int] = []

    class _Flag:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    def all_reduce(flag, op=None) -> None:
        calls.append(flag.value)
        flag.value = 1

    loop.torch.tensor = lambda value, device=None: _Flag(value)
    loop.torch.distributed = _module(
        "torch.distributed",
        ReduceOp=SimpleNamespace(MAX="max"),
        is_available=lambda: True,
        is_initialized=lambda: True,
        all_reduce=all_reduce,
    )

    assert loop._distributed_any(_Accelerator(), False) is True
    assert calls == [0]


def test_training_loop_stops_before_epoch_end_side_effects(monkeypatch) -> None:
    loop = _load_loop_module(monkeypatch)
    calls: list[str] = []

    def finish_epoch_at_max_steps(_trainer, state, _epoch, prefetch=None):
        state.global_step = state.args.max_train_steps
        return prefetch

    monkeypatch.setattr(loop, "_run_epoch_steps", finish_epoch_at_max_steps)
    monkeypatch.setattr(loop, "_run_epoch_validation", lambda *_args: calls.append("validation"))
    monkeypatch.setattr(loop, "_log_epoch_average", lambda *_args: calls.append("log"))
    monkeypatch.setattr(loop, "_run_adapter_epoch_hooks", lambda *_args: calls.append("hooks"))

    state = _state()
    state.saver = SimpleNamespace(
        maybe_save_epoch=lambda *_args, **_kwargs: calls.append("save_epoch"),
        maybe_save_resumable=lambda *_args, **_kwargs: calls.append("save_state"),
    )
    trainer = SimpleNamespace(sample_images=lambda *_args, **_kwargs: calls.append("sample"))

    loop.run_training_loop(trainer, state)

    assert calls == []
    assert state.accelerator.waits == 1
    assert state.metadata["ss_training_finished_at"]


def test_training_loop_cancels_prefetch_when_max_steps_reached(monkeypatch) -> None:
    loop = _load_loop_module(monkeypatch)
    calls: list[str] = []

    class _Prefetch:
        def cancel(self) -> None:
            calls.append("cancel")

    def finish_epoch_at_max_steps(_trainer, state, _epoch, prefetch=None):
        state.global_step = state.args.max_train_steps
        return _Prefetch()

    monkeypatch.setattr(loop, "_run_epoch_steps", finish_epoch_at_max_steps)
    monkeypatch.setattr(loop, "_run_epoch_validation", lambda *_args: calls.append("validation"))
    monkeypatch.setattr(loop, "_log_epoch_average", lambda *_args: calls.append("log"))
    monkeypatch.setattr(loop, "_run_adapter_epoch_hooks", lambda *_args: calls.append("hooks"))

    state = _state()
    state.saver = SimpleNamespace(
        maybe_save_epoch=lambda *_args, **_kwargs: calls.append("save_epoch"),
        maybe_save_resumable=lambda *_args, **_kwargs: calls.append("save_state"),
    )
    trainer = SimpleNamespace(sample_images=lambda *_args, **_kwargs: calls.append("sample"))

    loop.run_training_loop(trainer, state)

    assert calls == ["cancel"]


def test_training_loop_cancels_prefetch_when_paused(monkeypatch) -> None:
    loop = _load_loop_module(monkeypatch)
    calls: list[str] = []

    class _Prefetch:
        def cancel(self) -> None:
            calls.append("cancel")

    def pause_with_prefetch(_trainer, state, _epoch, prefetch=None):
        state._lorahub_pause = True
        return _Prefetch()

    monkeypatch.setattr(loop, "_run_epoch_steps", pause_with_prefetch)
    monkeypatch.setattr(loop, "_run_epoch_validation", lambda *_args: calls.append("validation"))
    monkeypatch.setattr(loop, "_log_epoch_average", lambda *_args: calls.append("log"))
    monkeypatch.setattr(loop, "_run_adapter_epoch_hooks", lambda *_args: calls.append("hooks"))

    state = _state()
    state.saver = SimpleNamespace(
        maybe_save_epoch=lambda *_args, **_kwargs: calls.append("save_epoch"),
        maybe_save_resumable=lambda *_args, **_kwargs: calls.append("save_state"),
    )
    trainer = SimpleNamespace(sample_images=lambda *_args, **_kwargs: calls.append("sample"))

    loop.run_training_loop(trainer, state)

    assert calls == ["cancel"]


def test_epoch_steps_do_not_run_extra_batch_when_already_at_max(monkeypatch) -> None:
    loop = _load_loop_module(monkeypatch)
    monkeypatch.setattr(
        loop,
        "_run_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("extra step")),
    )

    state = _state(global_step=4000)

    assert loop._run_epoch_steps(SimpleNamespace(), state, epoch=1) is None

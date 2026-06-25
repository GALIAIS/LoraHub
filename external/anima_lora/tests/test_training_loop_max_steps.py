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

    def accumulate(self, _model):
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return None

        return _Ctx()


class _Network:
    def on_epoch_start(self, *_args, **_kwargs) -> None:
        pass


class _Loss:
    def __init__(self, finite: bool = True) -> None:
        self.finite = finite

    def all(self):
        return self

    def detach(self):
        return self


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
        optimizer=SimpleNamespace(
            param_groups=[{"lr": 0.001}],
            zero_grad=lambda *_args, **_kwargs: None,
        ),
        lr_scheduler=SimpleNamespace(base_lrs=[0.001]),
        is_tracking=False,
        profile_started=False,
        train_ctx=SimpleNamespace(),
        ema=None,
        nan_skips=0,
        nan_consecutive=0,
        on_step_start_for_network=lambda *_args, **_kwargs: None,
        profile_range=None,
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
            isfinite=lambda value: _Loss(getattr(value, "finite", True)),
            distributed=_module(
                "torch.distributed",
                is_available=lambda: False,
                is_initialized=lambda: False,
            ),
            no_grad=lambda: SimpleNamespace(
                __enter__=lambda _self: None,
                __exit__=lambda _self, *_args: None,
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


def test_nan_recovery_runs_on_non_main_rank(monkeypatch) -> None:
    loop = _load_loop_module(monkeypatch)
    accelerator = _Accelerator()
    accelerator.is_main_process = False
    state = _state(
        args=SimpleNamespace(nan_guard_max_consecutive=1, nan_guard_recover=True),
        accelerator=accelerator,
        nan_consecutive=1,
    )

    loop._maybe_recover_from_nan(state)

    assert state.optimizer.param_groups[0]["lr"] == 0.0005
    assert state.lr_scheduler.base_lrs == [0.0005]
    assert state.nan_consecutive == 0


def test_nan_guard_skip_is_distributed_across_ranks(monkeypatch) -> None:
    loop = _load_loop_module(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(loop, "_distributed_any", lambda _accelerator, _value: True)

    state = _state(
        args=SimpleNamespace(
            nan_guard=True,
            nan_guard_max_consecutive=5,
            nan_guard_recover=False,
        ),
    )
    trainer = SimpleNamespace(
        _cudagraph_mark_step=False,
        on_step_start=lambda *_args, **_kwargs: None,
        process_batch=lambda *_args, **_kwargs: _Loss(finite=True),
        run_after_backward=lambda *_args, **_kwargs: calls.append("after_backward"),
    )
    state.accelerator.backward = lambda *_args, **_kwargs: calls.append("backward")

    loop._run_step(trainer, state, batch={})

    assert calls == []
    assert state.nan_skips == 1

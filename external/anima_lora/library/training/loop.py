"""Training-loop orchestration.

Owns the per-epoch / per-step body that used to live inline in
``AnimaTrainer.train()``. The entrypoint is :func:`run_training_loop`, which
takes a built :class:`LoopState` plus the trainer instance so override hooks
(``process_batch``, ``on_step_start``, ``sample_images``, ``_run_validation``,
``generate_step_logs``, ``step_logging``, ``epoch_logging``) keep working
unchanged.

State that used to be on ``self`` for cross-call signaling —
``_last_router_H_postfix``, ``_cudagraph_mark_step``, ``_hydra_warmup_step``,
``_adapters`` — stays on the trainer; this module reads them through the
``trainer`` handle.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from library import train_util
from library.datasets import LossRecorder
from library.runtime.device import clean_memory_on_device
from library.training.checkpoints import CheckpointSaver
from library.training.contexts import TrainCtx, ValCtx
from library.training.method_adapter import StepCtx
from library.training.metrics import MetricContext, collect_metrics

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Bundles every local that used to live in ``train()``'s for-epoch scope.

    Most fields are constants for the run; ``global_step``, ``profile_started``,
    ``profile_range``, ``initial_step``, and ``text_encoder(s)`` are mutated
    during the loop. ``current_epoch`` / ``current_step`` are mp.Value handles
    shared with :class:`CheckpointSaver` for state persistence.
    """

    args: Any
    accelerator: Accelerator
    train_ctx: TrainCtx
    val_ctx: ValCtx
    saver: CheckpointSaver

    network: Any
    unet: Any
    text_encoder: Any
    text_encoders: list
    vae: Any
    tokenizers: Any
    training_model: Any
    train_dataloader: Any
    optimizer: Any
    lr_scheduler: Any
    lr_descriptions: Optional[list]
    optimizer_train_fn: Callable
    optimizer_eval_fn: Callable
    weight_dtype: Any
    unet_weight_dtype: Any

    current_epoch: Any  # mp.Value
    current_step: Any  # mp.Value
    num_train_epochs: int
    epoch_to_start: int
    initial_step: int

    metadata: dict
    is_tracking: bool
    progress_bar: Any
    loss_recorder: LossRecorder
    val_step_loss_recorder: LossRecorder
    val_epoch_loss_recorder: LossRecorder

    validation_steps: int

    profile_range: Optional[tuple]
    on_step_start_for_network: Callable

    global_step: int = 0
    profile_started: bool = False

    # Optional EMA shadow of the LoRA-network's trainable params.
    # Built in train.py when --ema is set; consumed by _run_step (post
    # optimizer.step) and by the saver via its own ``ema=`` field.
    # None when the user didn't opt in.
    ema: Any = None

    # Counters for the NaN/spike guard. Bumped from _run_step;
    # consumed by the rollback path. Module-level state would break
    # multi-instance trainers, so keep it on the loop state.
    nan_skips: int = 0
    nan_consecutive: int = 0


def build_loop_state(
    trainer,
    *,
    args,
    accelerator: Accelerator,
    saver: CheckpointSaver,
    network,
    unet,
    text_encoder,
    text_encoders,
    vae,
    tokenizers,
    training_model,
    train_dataloader,
    val_dataloader,
    val_dataset_group,
    optimizer,
    lr_scheduler,
    lr_descriptions,
    optimizer_train_fn,
    optimizer_eval_fn,
    weight_dtype,
    unet_weight_dtype,
    vae_dtype,
    text_encoding_strategy,
    tokenize_strategy,
    train_text_encoder,
    train_unet,
    current_epoch,
    current_step,
    num_train_epochs,
    epoch_to_start,
    initial_step,
    metadata,
    ema=None,
) -> LoopState:
    """Build :class:`LoopState`. Mirrors the pre-loop setup that used to sit
    between ``_prepare_with_accelerator()`` and the for-epoch loop in
    ``train()``: noise scheduler, trackers, loss recorders, optional text
    encoder eviction, ``--sample_at_first``, train/val ctx construction,
    progress bar, profiler parsing.
    """
    noise_scheduler = trainer.get_noise_scheduler(args, accelerator.device)

    train_util.init_trackers(accelerator, args, "network_train")

    loss_recorder = LossRecorder()
    val_step_loss_recorder = LossRecorder()
    val_epoch_loss_recorder = LossRecorder()

    if hasattr(accelerator.unwrap_model(network), "on_step_start"):
        on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
    else:

        def on_step_start_for_network(*args, **kwargs):
            return None

    if trainer.is_text_encoder_not_needed_for_training(args):
        logger.info("text_encoder is not needed for training. deleting to save memory.")
        for t_enc in text_encoders:
            del t_enc
        text_encoders = []
        text_encoder = None
        gc.collect()
        clean_memory_on_device(accelerator.device)

    # --sample_at_first
    optimizer_eval_fn()
    trainer.sample_images(
        accelerator,
        args,
        0,
        0,
        accelerator.device,
        vae,
        tokenizers,
        text_encoder,
        unet,
    )
    optimizer_train_fn()
    is_tracking = len(accelerator.trackers) > 0
    if is_tracking:
        accelerator.log({}, step=0)

    train_ctx = TrainCtx(
        args=args,
        accelerator=accelerator,
        network=network,
        unet=unet,
        vae=vae,
        text_encoders=text_encoders,
        noise_scheduler=noise_scheduler,
        text_encoding_strategy=text_encoding_strategy,
        tokenize_strategy=tokenize_strategy,
        vae_dtype=vae_dtype,
        weight_dtype=weight_dtype,
        train_text_encoder=train_text_encoder,
        train_unet=train_unet,
        optimizer_eval_fn=optimizer_eval_fn,
        optimizer_train_fn=optimizer_train_fn,
        is_tracking=is_tracking,
    )

    # Skip prelude: when resuming with skip_until_initial_step, fast-forward
    # the global_step counter before tqdm so the bar total is sized correctly,
    # and consume per-epoch skip credit so dataloader.skip_first_batches has
    # the right offset on the first epoch only. Runs before the dtype log so
    # the log order matches the original train() body.
    global_step = 0
    if initial_step > 0:
        global_step = initial_step // args.gradient_accumulation_steps
        for skip_epoch in range(epoch_to_start):
            logger.info(
                f"skipping epoch {skip_epoch + 1} because initial_step "
                f"(multiplied) is {initial_step}"
            )
            initial_step -= len(train_dataloader)

    logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
    for i, t_enc in enumerate(text_encoders):
        params_itr = t_enc.parameters()
        params_itr.__next__()  # skip the first parameter
        params_itr.__next__()  # skip the second parameter (CLIP first two are embeddings)
        param_3rd = params_itr.__next__()
        logger.info(
            f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}"
        )

    clean_memory_on_device(accelerator.device)

    progress_bar = tqdm(
        range(args.max_train_steps),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
        # LoRaHub patch: when resuming with --skip_until_initial_step,
        # advance the bar to the global_step we recovered from
        # train_state.json so the X/total readout matches the absolute
        # step counter (e.g. ``90/4000`` after pause-at-89), not the
        # relative ``0/3911`` the original ``range(remaining)`` produced.
        # The dataloader skip prelude consumes the first ``initial_step``
        # batches; the bar update inside the inner loop bumps the
        # counter once per real optimizer step from there on.
        initial=global_step,
    )

    validation_steps = (
        min(args.max_validation_steps, len(val_dataloader))
        if args.max_validation_steps is not None
        else len(val_dataloader)
    )
    # Validate at fixed sigma values across the schedule:
    # 0.1 = near-clean / fine detail, 0.4 = mid / bulk structure,
    # 0.7 = high noise / coarse denoising (early inference steps).
    validation_sigmas = (
        args.validation_sigmas
        if args.validation_sigmas is not None
        else [0.1, 0.4, 0.7]
    )
    val_ctx = ValCtx(
        dataloader=val_dataloader,
        sigmas=validation_sigmas,
        steps=validation_steps,
        total_steps=validation_steps * len(validation_sigmas),
        train_loss_recorder=loss_recorder,
        original_t_min=args.t_min,
        original_t_max=args.t_max,
        dataset_group=val_dataset_group,
    )

    # nsys workflow: --profile_steps START-END toggles the cuda profiler API
    # around the requested step window. Wrap the launch with
    #   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
    # so nsys only records that window.
    profile_range = trainer._parse_profile_steps(args)

    return LoopState(
        args=args,
        accelerator=accelerator,
        train_ctx=train_ctx,
        val_ctx=val_ctx,
        saver=saver,
        network=network,
        unet=unet,
        text_encoder=text_encoder,
        text_encoders=text_encoders,
        vae=vae,
        tokenizers=tokenizers,
        training_model=training_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_descriptions=lr_descriptions,
        optimizer_train_fn=optimizer_train_fn,
        optimizer_eval_fn=optimizer_eval_fn,
        weight_dtype=weight_dtype,
        unet_weight_dtype=unet_weight_dtype,
        current_epoch=current_epoch,
        current_step=current_step,
        num_train_epochs=num_train_epochs,
        epoch_to_start=epoch_to_start,
        initial_step=initial_step,
        metadata=metadata,
        is_tracking=is_tracking,
        progress_bar=progress_bar,
        loss_recorder=loss_recorder,
        val_step_loss_recorder=val_step_loss_recorder,
        val_epoch_loss_recorder=val_epoch_loss_recorder,
        validation_steps=validation_steps,
        profile_range=profile_range,
        on_step_start_for_network=on_step_start_for_network,
        global_step=global_step,
        ema=ema,
    )


def run_training_loop(trainer, state: LoopState) -> None:
    """Run the full for-epoch training loop and the post-loop end-of-training
    metadata write. Mutates ``state.global_step``, profiler bookkeeping, and
    the metadata dict; the per-checkpoint saves go through ``state.saver``.
    """
    args = state.args
    accelerator = state.accelerator

    for epoch in range(state.epoch_to_start, state.num_train_epochs):
        accelerator.print(f"\nepoch {epoch + 1}/{state.num_train_epochs}\n")
        state.current_epoch.value = epoch + 1
        state.metadata["ss_epoch"] = str(epoch + 1)

        # network.train() is invoked here
        accelerator.unwrap_model(state.network).on_epoch_start(
            state.text_encoder, state.unet
        )

        _run_epoch_steps(trainer, state, epoch)

        # LoRaHub pause break — when ``_run_epoch_steps`` exited via
        # the pause flag, ``state._lorahub_pause`` is set. Skip the
        # epoch-end saves (they would write a fresh
        # ``<output_name>-NNNNNN`` file *after* the pause hook already
        # wrote ``<output_name>-checkpoint-state`` — that's how a
        # single pause used to drop 4 ckpts: the inner save +
        # maybe_save_epoch + maybe_save_resumable + the next outer
        # epoch's even-cadence save), and break the outer loop so
        # train.py's epilogue (which our patch makes a no-op when
        # ``args._lorahub_paused`` is set) doesn't get a chance to
        # write the final checkpoint either.
        if getattr(state, "_lorahub_pause", False):
            break

        _run_epoch_validation(trainer, state, epoch)
        _log_epoch_average(trainer, state, epoch)
        _run_adapter_epoch_hooks(trainer, state)

        accelerator.wait_for_everyone()

        state.optimizer_eval_fn()
        state.saver.maybe_save_epoch(
            state.network, state.global_step, epoch, state.num_train_epochs
        )
        state.saver.maybe_save_resumable(
            state.network, state.global_step, epoch, state.num_train_epochs
        )

        trainer.sample_images(
            accelerator,
            args,
            epoch + 1,
            state.global_step,
            accelerator.device,
            state.vae,
            state.tokenizers,
            state.text_encoder,
            state.unet,
        )
        state.optimizer_train_fn()

    state.metadata["ss_training_finished_at"] = str(time.time())


def _run_epoch_steps(trainer, state: LoopState, epoch: int) -> None:
    """Inner per-step loop: walk the dataloader, execute the accumulate
    scope, run sample / save / log / step-validation ticks."""
    args = state.args
    accelerator = state.accelerator

    skipped_dataloader = None
    if state.initial_step > 0:
        skipped_dataloader = accelerator.skip_first_batches(
            state.train_dataloader, state.initial_step - 1
        )
        state.initial_step = 1

    for step, batch in enumerate(skipped_dataloader or state.train_dataloader):
        state.current_step.value = state.global_step
        if state.initial_step > 0:
            state.initial_step -= 1
            continue

        _profiler_step_begin(state)

        loss = _run_step(trainer, state, batch)

        _profiler_step_end(state)

        keys_scaled, mean_norm, maximum_norm, max_mean_logs = _maybe_scale_norm(state)

        if accelerator.sync_gradients:
            state.progress_bar.update(1)
            state.global_step += 1
            _sample_at_step(trainer, state)
            state.saver.maybe_save_step(state.network, state.global_step, epoch)
            state.optimizer_train_fn()

        _log_step(
            trainer,
            state,
            loss=loss,
            step=step,
            epoch=epoch,
            keys_scaled=keys_scaled,
            mean_norm=mean_norm,
            maximum_norm=maximum_norm,
            max_mean_logs=max_mean_logs,
        )
        _maybe_run_step_validation(trainer, state, epoch)

        # LoRaHub pause hook — when an external process drops a
        # ``_lorahub_pause`` file under the workspace (parent of
        # ``args.output_dir``), overwrite the resumable
        # ``<output_name>-checkpoint-state`` directory in place and
        # break out of the loop. Reuses sd-scripts' own
        # ``save_checkpoint_state`` (the same one
        # ``checkpointing_epochs`` calls at each cadence boundary), so
        # pausing never adds a *new* state directory — it just bumps
        # the existing checkpoint to the current step. The matching
        # weights are read out of the state dir's ``model.safetensors``
        # at resume time, so we skip writing a separate step ckpt.
        # Only the main process touches the file system; other ranks
        # block in ``wait_for_everyone()``.
        if accelerator.sync_gradients and accelerator.is_main_process:
            try:
                ws_root = os.path.dirname(os.path.abspath(args.output_dir))
                pause_flag = os.path.join(ws_root, "_lorahub_pause")
                if os.path.exists(pause_flag):
                    logger.info(
                        "[lorahub] pause flag detected — saving checkpoint state at step %d and exiting",
                        state.global_step,
                    )
                    try:
                        from library.training.checkpoints import (
                            save_checkpoint_state,
                        )
                        # Mark the run as paused before writing state
                        # so train_state.json (written via accelerate
                        # save-hooks registered by CheckpointSaver)
                        # carries the right step. Then overwrite the
                        # in-place checkpoint-state dir.
                        save_checkpoint_state(args, accelerator)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "[lorahub] forced save failed: %s", exc
                        )
                    # Best-effort flag cleanup so a later /resume
                    # doesn't immediately re-pause. The LoRaHub side
                    # also clears this on its end.
                    try:
                        os.remove(pause_flag)
                    except OSError:
                        pass
                    # Tell train.py's epilogue not to write the
                    # final ``<output_name>-state`` and
                    # ``<output_name>.safetensors`` — they would
                    # otherwise be written *after* this point and
                    # add two more files the user doesn't want.
                    # ``cleanup_resumable`` is also skipped via the
                    # same flag so the checkpoint we just wrote
                    # survives until /resume picks it up.
                    args._lorahub_paused = True  # type: ignore[attr-defined]
                    state._lorahub_pause = True  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning("[lorahub] pause-flag check failed: %s", exc)

        if getattr(state, "_lorahub_pause", False):
            accelerator.wait_for_everyone()
            break

        if state.global_step >= args.max_train_steps:
            break


def _run_step(trainer, state: LoopState, batch) -> torch.Tensor:
    """The accumulate-scope body: on_step_start hooks, cudagraph mark, forward,
    backward gating, sync_gradients hooks (hydra warmup, grad capture, clip),
    optimizer step + zero_grad. Returns the loss (detached or live)."""
    args = state.args
    accelerator = state.accelerator
    network = state.network

    with accelerator.accumulate(state.training_model):
        state.on_step_start_for_network(state.text_encoder, state.unet)

        # preprocess batch for each model
        trainer.on_step_start(state.train_ctx, batch, is_train=True)

        # Clear last-step gate/σ tensor refs + memoized router-stats caches
        # before the next forward. Called unconditionally — the cudagraph
        # branch below also needs it (lingering refs into the cudagraph
        # memory pool block pool reclamation, demoting the run to eager),
        # and the eager path needs it so per-step memoized stats
        # (``_router_stats_cache`` / ``_chimera_router_stats_cache``) get
        # invalidated each step instead of freezing at their first computed
        # values. Cost is ~60 Python attr writes; stats compute itself is
        # already log-step-gated by callers.
        net_unwrapped = accelerator.unwrap_model(network)
        if hasattr(net_unwrapped, "clear_step_caches"):
            net_unwrapped.clear_step_caches()

        # CUDAGraphs (reduce-overhead / max-autotune) also need an explicit
        # iteration boundary for inductor's cudagraph_trees. Without this
        # call, the "pending, uninvoked backwards" fast-path check fails
        # every step and cudagraphs silently fall back to the eager path —
        # you pay compile latency and keep launch overhead. Must be called
        # before the forward on every step.
        if trainer._cudagraph_mark_step:
            torch.compiler.cudagraph_mark_step_begin()

        if state.profile_started:
            torch.cuda.nvtx.range_push("forward")
        loss = trainer.process_batch(state.train_ctx, batch, is_train=True)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        # NaN/spike guard — pre-backward. A NaN/Inf loss must NOT
        # propagate through ``accelerator.backward()`` because the
        # gradient computation will poison every parameter touched by
        # the graph. We zero_grad and bail out early; the optimizer
        # will skip this step entirely. Cheap (one .isfinite() reduction
        # per accumulation slice) so always-on when the user opts in.
        nan_guard_on = bool(getattr(args, "nan_guard", False))
        skipped_for_nan = False
        if nan_guard_on and not torch.isfinite(loss).all():
            state.nan_skips += 1
            state.nan_consecutive += 1
            state.optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                logger.warning(
                    "non-finite loss at global_step=%d (skip #%d, consecutive=%d)",
                    state.global_step, state.nan_skips, state.nan_consecutive,
                )
            _maybe_recover_from_nan(state)
            return loss.detach()

        if state.profile_started:
            torch.cuda.nvtx.range_push("backward")
        accelerator.backward(loss)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        if accelerator.sync_gradients:
            net_unwrapped = accelerator.unwrap_model(network)
            # Snapshot Hydra up-weight grad norms before zero_grad wipes them.
            # The metric ``hydra_up_grad`` reads this stash later in the step.
            # Also runs pre-clip so absolute magnitudes aren't distorted by
            # the global rescale (clipping preserves the below/above ratio).
            # Skip on non-log steps — the metric only fires at log cadence,
            # so capturing every step burns kernels whose output is never
            # read. global_step increments below, so predict the
            # post-increment value.
            _log_every = max(1, int(getattr(args, "log_every_n_steps", 1) or 1))
            _will_log_after = state.is_tracking and (
                ((state.global_step + 1) % _log_every == 0)
                or ((state.global_step + 1) >= args.max_train_steps)
            )
            if _will_log_after and hasattr(net_unwrapped, "capture_up_grad_stats"):
                net_unwrapped.capture_up_grad_stats()
            if args.max_grad_norm != 0.0:
                params_to_clip = accelerator.unwrap_model(
                    network
                ).get_trainable_params()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            # NaN/spike guard — post-backward, post-clip. Catches the
            # case where the loss was finite but per-param grads
            # contain NaN/Inf (mixed-precision under/overflow,
            # poison-pill module). Bail out before optimizer.step()
            # so the parameters stay clean.
            if nan_guard_on:
                params_to_check = accelerator.unwrap_model(network).get_trainable_params()
                bad = False
                for p in params_to_check:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad = True
                        break
                if bad:
                    state.nan_skips += 1
                    state.nan_consecutive += 1
                    state.optimizer.zero_grad(set_to_none=True)
                    if accelerator.is_main_process:
                        logger.warning(
                            "non-finite gradient at global_step=%d "
                            "(skip #%d, consecutive=%d)",
                            state.global_step, state.nan_skips,
                            state.nan_consecutive,
                        )
                    _maybe_recover_from_nan(state)
                    skipped_for_nan = True

        if skipped_for_nan:
            return loss.detach()

        if state.profile_started:
            torch.cuda.nvtx.range_push("optimizer")
        state.optimizer.step()
        state.lr_scheduler.step()
        state.optimizer.zero_grad(set_to_none=True)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        # Successful step — reset the consecutive counter so a future
        # spike has to clear the fresh threshold.
        if state.nan_consecutive > 0 and accelerator.sync_gradients:
            state.nan_consecutive = 0

        # EMA shadow update — must run after the optimizer step so
        # we read post-update parameter values, and only when the
        # gradients have actually synchronised (gradient_accumulation
        # otherwise applies decay on a half-finished step).
        if (
            state.ema is not None
            and accelerator.sync_gradients
        ):
            state.ema.step(accelerator.unwrap_model(network))

    return loss


def _maybe_recover_from_nan(state: LoopState) -> None:
    """Trigger recovery when too many consecutive NaN steps stack up.

    Two modes, picked by ``--nan_guard_recover``:

    * Off (default): log a hard warning when threshold is exceeded.
      The user gets a clear signal that the run is unhealthy without
      the script silently mutating training state.
    * On: halve every param group's LR and, if EMA is configured,
      swap the shadow back into the live network. This is conservative
      — we don't restart the LR schedule, just clamp it once.
    """
    args = state.args
    threshold = max(1, int(getattr(args, "nan_guard_max_consecutive", 5) or 5))
    if state.nan_consecutive < threshold:
        return
    if not state.accelerator.is_main_process:
        return
    if not bool(getattr(args, "nan_guard_recover", False)):
        logger.error(
            "NaN/spike threshold reached (%d consecutive). "
            "Pass --nan_guard_recover to auto-halve LR + restore EMA, "
            "or stop training and inspect the data pipeline.",
            state.nan_consecutive,
        )
        # Reset so we don't log this every step until end of run.
        state.nan_consecutive = 0
        return

    # Halve every group's LR. We don't touch the scheduler — it'll
    # carry on but from a lower base. Most schedulers (cosine, polynomial,
    # constant_with_warmup) work off ``base_lrs`` not ``lr``, so we
    # update both fields where present.
    for group in state.optimizer.param_groups:
        old = float(group.get("lr", 0.0))
        new = old * 0.5
        group["lr"] = new
    if hasattr(state.lr_scheduler, "base_lrs"):
        try:
            state.lr_scheduler.base_lrs = [
                lr * 0.5 for lr in state.lr_scheduler.base_lrs
            ]
        except Exception:  # noqa: BLE001
            pass
    logger.warning(
        "nan_guard recovery: halved LR (now %s)",
        [g.get("lr") for g in state.optimizer.param_groups],
    )

    if state.ema is not None:
        # Swap the EMA shadow into the live network so we restart
        # from the last *good* parameter snapshot. The swap context
        # manager normally restores live values on exit, but here we
        # want the swap to persist — so do it explicitly.
        unwrapped = state.accelerator.unwrap_model(state.network)
        from library.training.ema import _named_trainables

        named = _named_trainables(unwrapped)
        if len(named) == len(state.ema.shadow_params):
            with torch.no_grad():
                for shadow, (_, live) in zip(state.ema.shadow_params, named):
                    live.data.copy_(shadow.to(live.device, dtype=live.dtype))
            logger.warning(
                "nan_guard recovery: restored network params from EMA shadow",
            )

    state.nan_consecutive = 0


def _profiler_step_begin(state: LoopState) -> None:
    if (
        state.profile_range
        and state.global_step == state.profile_range[0]
        and not state.profile_started
    ):
        state.accelerator.print(f"\n[profiler] starting at step {state.global_step}")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        state.profile_started = True

    if state.profile_started:
        torch.cuda.nvtx.range_push(f"step={state.global_step}")


def _profiler_step_end(state: LoopState) -> None:
    if state.profile_started:
        torch.cuda.nvtx.range_pop()  # close per-step NVTX range
    if state.profile_started and state.global_step >= state.profile_range[1]:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        state.accelerator.print(f"\n[profiler] stopped at step {state.global_step}")
        state.accelerator.print(
            "[profiler] open the .nsys-rep with the Nsight Systems GUI\n"
        )
        state.profile_started = False
        state.profile_range = None  # don't re-trigger
        # Hard-exit so the accelerate launcher exits and nsys finalizes the
        # report. sys.exit(0) hangs in interpreter shutdown here (DataLoader
        # workers + NCCL/CUDA atexit handlers all wait on futexes). os._exit
        # skips that cleanup; the profile buffer is already flushed by the
        # preceding cuda.synchronize() + cuProfilerStop, and the .nsys-rep is
        # owned by nsys, not this process, so the hard exit is safe.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _maybe_scale_norm(state: LoopState):
    args = state.args
    if args.scale_weight_norms:
        keys_scaled, mean_norm, maximum_norm = state.accelerator.unwrap_model(
            state.network
        ).apply_max_norm_regularization(
            args.scale_weight_norms, state.accelerator.device
        )
        max_mean_logs = {
            "Keys Scaled": keys_scaled,
            "Average key norm": mean_norm,
        }
        return keys_scaled, mean_norm, maximum_norm, max_mean_logs
    return None, None, None, {}


def _sample_at_step(trainer, state: LoopState) -> None:
    state.optimizer_eval_fn()
    trainer.sample_images(
        state.accelerator,
        state.args,
        None,
        state.global_step,
        state.accelerator.device,
        state.vae,
        state.tokenizers,
        state.text_encoder,
        state.unet,
    )


def _log_step(
    trainer,
    state: LoopState,
    *,
    loss,
    step: int,
    epoch: int,
    keys_scaled,
    mean_norm,
    maximum_norm,
    max_mean_logs,
) -> None:
    args = state.args
    log_every = max(1, int(getattr(args, "log_every_n_steps", 1) or 1))
    # Gate on sync_gradients: with gradient_accumulation_steps > 1 the
    # dataloader fires this hook once per micro-batch but global_step only
    # advances on sync. Without the gate, log_every_n_steps decides the same
    # answer for every micro-batch in an accumulation cycle, producing bursts
    # of N back-to-back tracker writes followed by N silent ones (exactly the
    # "sometimes honored, sometimes ignored" pattern).
    should_log_step = state.accelerator.sync_gradients and (
        (state.global_step % log_every == 0)
        or (state.global_step >= args.max_train_steps)
    )

    current_loss = loss.detach().item()
    state.loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
    avr_loss: float = state.loss_recorder.moving_average
    logs = {"avr_loss": avr_loss}
    _unwrapped_net = state.accelerator.unwrap_model(state.network)
    # Refresh router_H only on log cadence — get_router_entropy → full
    # get_router_stats compute (with D2H syncs) is wasted if the only
    # consumer is the progress-bar postfix. Cache last value on trainer
    # so tqdm shows a stale value harmlessly between log steps.
    if getattr(_unwrapped_net, "_use_hydra", False) and should_log_step:
        _router_H = _unwrapped_net.get_router_entropy()
        if _router_H is not None:
            trainer._last_router_H_postfix = _router_H
    _router_H_cached = getattr(trainer, "_last_router_H_postfix", None)
    if _router_H_cached is not None:
        logs["router_H"] = f"{_router_H_cached:.3f}"
    state.progress_bar.set_postfix(refresh=False, **{**max_mean_logs, **logs})

    if state.is_tracking and should_log_step:
        logs = trainer.generate_step_logs(
            args,
            current_loss,
            avr_loss,
            state.lr_scheduler,
            state.lr_descriptions,
            state.optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            None,  # mean_grad_norm — not tracked here
            None,  # mean_combined_norm — not tracked here
        )
        producers = [_unwrapped_net, *trainer._adapters]
        logs.update(
            collect_metrics(
                producers,
                MetricContext(args=args, network=_unwrapped_net),
            )
        )
        trainer.step_logging(state.accelerator, logs, state.global_step, epoch + 1)


def _maybe_run_step_validation(trainer, state: LoopState, epoch: int) -> None:
    args = state.args
    should_validate_step = (
        args.validate_every_n_steps is not None
        and state.global_step % args.validate_every_n_steps == 0
    )
    if (
        state.accelerator.sync_gradients
        and state.validation_steps > 0
        and should_validate_step
    ):
        trainer._run_validation(
            state.train_ctx,
            state.val_ctx,
            val_loss_recorder=state.val_step_loss_recorder,
            epoch=epoch,
            global_step=state.global_step,
            progress_bar=state.progress_bar,
            progress_desc="validation steps",
            postfix_label="val_avg_loss",
            log_avg_key="loss/validation/step_average",
            log_div_key="loss/validation/step_divergence",
            logging_fn=trainer.step_logging,
        )


def _run_epoch_validation(trainer, state: LoopState, epoch: int) -> None:
    args = state.args
    should_validate_epoch = (
        (epoch + 1) % args.validate_every_n_epochs == 0
        if args.validate_every_n_epochs is not None
        else True
    )
    if should_validate_epoch and len(state.val_ctx.dataloader) > 0:
        trainer._run_validation(
            state.train_ctx,
            state.val_ctx,
            val_loss_recorder=state.val_epoch_loss_recorder,
            epoch=epoch,
            global_step=state.global_step,
            progress_bar=state.progress_bar,
            progress_desc="epoch validation steps",
            postfix_label="val_epoch_avg_loss",
            log_avg_key="loss/validation/epoch_average",
            log_div_key="loss/validation/epoch_divergence",
            logging_fn=trainer.epoch_logging,
        )


def _log_epoch_average(trainer, state: LoopState, epoch: int) -> None:
    if not state.is_tracking:
        return
    logs = {"loss/epoch_average": state.loss_recorder.moving_average}
    trainer.epoch_logging(state.accelerator, logs, state.global_step, epoch + 1)


def _run_adapter_epoch_hooks(trainer, state: LoopState) -> None:
    """Per-method end-of-epoch hooks (IP-Adapter diagnostic dump, …).
    Main process only — adapters that need cross-rank reduction should do
    that internally."""
    if not (trainer._adapters and state.accelerator.is_main_process):
        return
    epoch_end_ctx = StepCtx(
        args=state.args,
        accelerator=state.accelerator,
        network=state.network,
        weight_dtype=state.weight_dtype,
    )
    for adapter in trainer._adapters:
        adapter.on_epoch_end(epoch_end_ctx)

from __future__ import annotations


def epoch_training_plan(
    *,
    epochs: int,
    batches_per_epoch: int,
    gradient_accumulation: int,
    max_steps: int | None,
) -> tuple[int, int]:
    """Return ``(steps_per_epoch, total_steps)`` for an ai-toolkit job."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batches_per_epoch < 1:
        raise ValueError("epoch-based training dataset produced no batches")
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1")

    steps_per_epoch = -(-batches_per_epoch // gradient_accumulation)
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        total_steps = min(total_steps, max_steps)
    return steps_per_epoch, total_steps


def epoch_cadence_due(
    previous_epoch: int,
    current_epoch: int,
    every_n_epochs: int | None,
) -> bool:
    """Return whether an epoch cadence boundary was crossed."""
    if every_n_epochs is None or every_n_epochs < 1 or current_epoch <= previous_epoch:
        return False
    return current_epoch // every_n_epochs > previous_epoch // every_n_epochs

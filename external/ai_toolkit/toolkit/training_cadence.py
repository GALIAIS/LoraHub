from __future__ import annotations


def epoch_cadence_due(
    previous_epoch: int,
    current_epoch: int,
    every_n_epochs: int | None,
) -> bool:
    """Return whether an epoch cadence boundary was crossed."""
    if every_n_epochs is None or every_n_epochs < 1 or current_epoch <= previous_epoch:
        return False
    return current_epoch // every_n_epochs > previous_epoch // every_n_epochs

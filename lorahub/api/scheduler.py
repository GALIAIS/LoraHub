"""In-memory single-slot job scheduler with VRAM-aware matching.

`POST /jobs` enqueues a job and returns immediately with `state=queued`.
Background worker threads (one per slot) pop pending jobs from a FIFO
deque and run the launch + wait sequence serially. The default is
`concurrency=1` (one training subprocess at a time); the slot list is
exposed so a future multi-GPU patch can dispatch by device id.

When `slot_capacities` (a `{slot_id: vram_gb}` mapping) is provided and a
`submit()` call carries `vram_required`, workers only pop tasks they have
the VRAM headroom to run. Tasks that exceed every declared capacity are
evicted via `capacity_reject_callback` so they don't deadlock the queue.

Stopping the scheduler is cooperative: workers finish whatever job they
are currently waiting on, then drain the queue and exit. Pending jobs
that never start are left in the registry as `queued` so the next
process can resume them (orphan recovery already handles non-terminal
states on reboot).
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)


# A submitted task is a callable that receives the assigned slot id.
# The slot id is opaque metadata for now; multi-GPU patches will use it
# to set CUDA_VISIBLE_DEVICES, etc.
TaskFn = Callable[[int], None]
# Fired when the head of the queue exceeds every declared slot capacity
# and gets evicted. Args: (job_id, required_gb, max_available_gb).
RejectCallback = Callable[[str, float, float], None]


@dataclass(slots=True)
class _PendingTask:
    job_id: str
    fn: TaskFn
    vram_required: float | None = None


class JobScheduler:
    """Thread-safe FIFO scheduler with N independent worker threads.

    Workers block on the deque via a `threading.Condition`. `submit()`
    appends and notifies. `stop()` flips a flag and wakes everyone.

    When `slot_capacities` is supplied, each worker only pops tasks
    whose `vram_required` (declared in `submit()`) fits its slot's
    declared capacity. A task that fits *no* slot is evicted via
    `capacity_reject_callback` so the queue can keep moving.
    """

    def __init__(
        self,
        *,
        concurrency: int = 1,
        available_slots: list[int] | None = None,
        slot_capacities: dict[int, float] | None = None,
        capacity_reject_callback: RejectCallback | None = None,
    ) -> None:
        if concurrency < 1:
            msg = "concurrency must be >= 1"
            raise ValueError(msg)
        self._concurrency = concurrency
        self._available_slots: list[int] = (
            list(available_slots) if available_slots is not None else list(range(concurrency))
        )
        if len(self._available_slots) != concurrency:
            msg = (
                f"available_slots must have len={concurrency}, "
                f"got {len(self._available_slots)}"
            )
            raise ValueError(msg)

        # Capacity defaults to +inf for any slot not explicitly declared,
        # so legacy `submit(job_id, fn)` callers (vram_required=None)
        # still match every slot.
        self._slot_capacities: dict[int, float] = {
            slot: float("inf") for slot in self._available_slots
        }
        if slot_capacities:
            for slot, cap in slot_capacities.items():
                self._slot_capacities[slot] = float(cap)
        self._reject_cb = capacity_reject_callback

        self._queue: deque[_PendingTask] = deque()
        self._cv = threading.Condition()
        self._workers: list[threading.Thread] = []
        self._stopping = False
        self._started = False

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @property
    def available_slots(self) -> list[int]:
        return list(self._available_slots)

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopping

    @property
    def queue_size(self) -> int:
        with self._cv:
            return len(self._queue)

    def queue_depth(self) -> int:
        with self._cv:
            return len(self._queue)

    def pending_job_ids(self) -> list[str]:
        with self._cv:
            return [t.job_id for t in self._queue]

    def cancel_pending(self, job_id: str) -> bool:
        """Remove a job from the pending queue before a worker claims it."""
        with self._cv:
            for idx, task in enumerate(self._queue):
                if task.job_id == job_id:
                    del self._queue[idx]
                    self._cv.notify_all()
                    return True
        return False

    def start(self) -> None:
        """Spawn worker threads. Idempotent."""
        with self._cv:
            if self._started:
                return
            self._started = True
            for i, slot in enumerate(self._available_slots):
                t = threading.Thread(
                    target=self._loop,
                    args=(slot,),
                    name=f"lorahub-worker-{i}",
                    daemon=True,
                )
                t.start()
                self._workers.append(t)

    def submit(
        self,
        job_id: str,
        fn: TaskFn,
        *,
        vram_required: float | None = None,
    ) -> None:
        """Enqueue a launch closure. Lazily starts workers on first submit.

        `vram_required` is in GB. `None` means "unknown / accept anywhere"
        and matches every slot regardless of declared capacity.
        """
        with self._cv:
            self._queue.append(
                _PendingTask(job_id=job_id, fn=fn, vram_required=vram_required)
            )
            # notify_all so any slot that could fit this task wakes up,
            # not just one (which might be too small and go right back to
            # sleep while the queue still has work for someone else).
            self._cv.notify_all()
            needs_start = not self._started
        if needs_start:
            self.start()

    def stop(self, *, timeout: float | None = 5.0) -> None:
        """Signal workers to drain and exit. Joins with `timeout` per worker."""
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        for t in self._workers:
            t.join(timeout=timeout)

    def _max_capacity(self) -> float:
        # `self._slot_capacities` always has at least one entry (built
        # from `available_slots` in `__init__`).
        return max(self._slot_capacities.values())

    def _fits(self, slot: int, task: _PendingTask) -> bool:
        if task.vram_required is None:
            return True
        cap = self._slot_capacities.get(slot, math.inf)
        return task.vram_required <= cap

    def _pick_next(
        self, slot: int
    ) -> tuple[_PendingTask | None, list[tuple[str, float, float]]]:
        """Caller must hold `self._cv`.

        Walks the deque looking for the first task this slot can run.
        While scanning, evicts any head-of-queue task that exceeds every
        declared capacity (deadlock-avoidance) and records it so the
        caller can fire `capacity_reject_callback` *outside* the lock.

        Strict FIFO is preserved: we never skip a task that *some*
        live slot could run; we only evict when nothing in the cluster
        can ever serve it.

        Returns `(task_or_none, rejects_to_fire)`.
        """
        rejects: list[tuple[str, float, float]] = []
        max_cap = self._max_capacity()

        while self._queue:
            head = self._queue[0]
            head_req = head.vram_required
            # Evict head if no slot can ever serve it. This is the only
            # scenario where strict FIFO is broken, and it has to be —
            # otherwise the whole queue stalls behind an impossible job.
            if head_req is not None and head_req > max_cap:
                self._queue.popleft()
                rejects.append((head.job_id, head_req, max_cap))
                continue

            # Head is feasible somewhere. If *we* can run it, take it —
            # never let a smaller item further back jump the line.
            if self._fits(slot, head):
                return self._queue.popleft(), rejects

            # We can't run head, but someone else can. Look further only
            # for a task we can run; do *not* pop anything ahead of head.
            picked: _PendingTask | None = None
            for i in range(1, len(self._queue)):
                cand = self._queue[i]
                if self._fits(slot, cand):
                    picked = cand
                    del self._queue[i]
                    break
            return picked, rejects

        return None, rejects

    def _loop(self, slot: int) -> None:
        while True:
            task: _PendingTask | None = None
            rejects: list[tuple[str, float, float]] = []
            with self._cv:
                while True:
                    if self._stopping and not self._queue:
                        # Final drain: nothing to do, exit after firing
                        # any rejects we already collected.
                        break
                    task, new_rejects = self._pick_next(slot)
                    if new_rejects:
                        rejects.extend(new_rejects)
                        # Head changed; wake peers so they re-check.
                        self._cv.notify_all()
                    if task is not None:
                        break
                    if rejects:
                        # Release the lock to fire callbacks, then loop.
                        break
                    if self._stopping:
                        break
                    # Nothing fits this slot right now. Sleep until a
                    # submit/stop/completion notifies us.
                    self._cv.wait()

            for job_id, req, cap in rejects:
                if self._reject_cb is not None:
                    try:
                        self._reject_cb(job_id, req, cap)
                    except Exception:  # noqa: BLE001
                        log.exception(
                            "capacity_reject_callback raised for job %s", job_id
                        )

            if task is None:
                if self._stopping:
                    return
                # Either we just fired rejects (loop again to re-pick)
                # or a spurious wake — either way, re-evaluate.
                continue

            try:
                task.fn(slot)
            except Exception:  # noqa: BLE001
                log.exception("scheduler task for job %s raised", task.job_id)
            # Task done: wake every worker so any blocked-on-capacity
            # peer can re-check. notify_all (not notify) avoids waking
            # the wrong slot for a long-tailed task.
            with self._cv:
                self._cv.notify_all()


# Default process-wide scheduler. Replaced by tests for isolation.
scheduler = JobScheduler(concurrency=1)


__all__ = ["JobScheduler", "RejectCallback", "TaskFn", "scheduler"]

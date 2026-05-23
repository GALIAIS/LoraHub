"""In-memory single-slot job scheduler.

`POST /jobs` enqueues a job and returns immediately with `state=queued`.
Background worker threads (one per slot) pop pending jobs from a FIFO
deque and run the launch + wait sequence serially. The default is
`concurrency=1` (one training subprocess at a time); the slot list is
exposed so a future multi-GPU patch can dispatch by device id.

Stopping the scheduler is cooperative: workers finish whatever job they
are currently waiting on, then drain the queue and exit. Pending jobs
that never start are left in the registry as `queued` so the next
process can resume them (orphan recovery already handles non-terminal
states on reboot).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)


# A submitted task is a callable that receives the assigned slot id.
# The slot id is opaque metadata for now (always 0 in single-slot mode);
# multi-GPU patches will use it to set CUDA_VISIBLE_DEVICES, etc.
TaskFn = Callable[[int], None]


@dataclass(slots=True)
class _PendingTask:
    job_id: str
    fn: TaskFn


class JobScheduler:
    """Thread-safe FIFO scheduler with N independent worker threads.

    Workers block on the deque via a `threading.Condition`. `submit()`
    appends and notifies. `stop()` flips a flag and wakes everyone.
    """

    def __init__(
        self,
        *,
        concurrency: int = 1,
        available_slots: list[int] | None = None,
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

    def queue_depth(self) -> int:
        with self._cv:
            return len(self._queue)

    def pending_job_ids(self) -> list[str]:
        with self._cv:
            return [t.job_id for t in self._queue]

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

    def submit(self, job_id: str, fn: TaskFn) -> None:
        """Enqueue a launch closure. Lazily starts workers on first submit."""
        with self._cv:
            self._queue.append(_PendingTask(job_id=job_id, fn=fn))
            self._cv.notify()
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

    def _loop(self, slot: int) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stopping:
                    self._cv.wait()
                if self._stopping and not self._queue:
                    return
                task = self._queue.popleft()
            try:
                task.fn(slot)
            except Exception:  # noqa: BLE001
                log.exception("scheduler task for job %s raised", task.job_id)


# Default process-wide scheduler. Replaced by tests for isolation.
scheduler = JobScheduler(concurrency=1)


__all__ = ["JobScheduler", "scheduler"]

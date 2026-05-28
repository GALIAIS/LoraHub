"""Tests for VRAM-aware scheduling in `JobScheduler`.

These cover the new `slot_capacities` / `vram_required` /
`capacity_reject_callback` knobs added for heterogeneous GPU rigs
(e.g. 4090 + 3060). The legacy FIFO behaviour is exercised by
`tests/test_scheduler.py` and must remain green alongside these.
"""

from __future__ import annotations

import threading
import time

from lorahub.api.scheduler import JobScheduler


def _drain(sched: JobScheduler, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while sched.queue_size > 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    sched.stop(timeout=timeout)


def test_no_capacities_falls_back_to_fifo() -> None:
    """Without `slot_capacities`, the scheduler behaves exactly like the
    legacy FIFO single-slot path: tasks run in submission order regardless
    of any `vram_required` hints.
    """
    sched = JobScheduler(concurrency=1)
    seen: list[str] = []
    done = threading.Event()
    lock = threading.Lock()

    def make(name: str, last: bool = False):
        def fn(_slot: int) -> None:
            with lock:
                seen.append(name)
            if last:
                done.set()
        return fn

    sched.submit("a", make("a"), vram_required=99.0)  # huge but unconstrained
    sched.submit("b", make("b"))
    sched.submit("c", make("c", last=True), vram_required=4.0)

    assert done.wait(timeout=2.0)
    assert seen == ["a", "b", "c"]
    sched.stop(timeout=2.0)


def test_heterogeneous_skips_oversized() -> None:
    """slot_capacities={0: 24, 1: 12}, queue [(A, 16GB), (B, 8GB)].

    Slot 1 (12GB) cannot run A (16GB) but can run B (8GB), so it must
    pick B while slot 0 (24GB) takes A. Without VRAM-awareness the
    smaller GPU would block on A and starve B.
    """
    sched = JobScheduler(
        concurrency=2,
        available_slots=[0, 1],
        slot_capacities={0: 24.0, 1: 12.0},
    )
    assignments: dict[str, int] = {}
    started_a = threading.Event()
    finished = threading.Event()
    counter = {"n": 0}
    lock = threading.Lock()
    # Hold A long enough that slot 1 must take B before A finishes,
    # otherwise the test would also pass with strict FIFO.
    a_can_finish = threading.Event()

    def task_a(slot: int) -> None:
        with lock:
            assignments["a"] = slot
        started_a.set()
        a_can_finish.wait(timeout=2.0)
        with lock:
            counter["n"] += 1
            if counter["n"] == 2:
                finished.set()

    def task_b(slot: int) -> None:
        # Wait until A has been picked up so we can prove B was
        # dispatched concurrently rather than after A.
        started_a.wait(timeout=2.0)
        with lock:
            assignments["b"] = slot
            counter["n"] += 1
            if counter["n"] == 2:
                finished.set()
        a_can_finish.set()

    sched.submit("a", task_a, vram_required=16.0)
    sched.submit("b", task_b, vram_required=8.0)

    assert finished.wait(timeout=3.0), f"deadlocked, assignments={assignments}"
    assert assignments == {"a": 0, "b": 1}, assignments
    sched.stop(timeout=2.0)


def test_capacity_reject_callback() -> None:
    """If the queue head exceeds every slot's capacity, the scheduler
    must evict it via `capacity_reject_callback` and keep draining.
    """
    rejects: list[tuple[str, float, float]] = []
    cb_lock = threading.Lock()

    def on_reject(job_id: str, required: float, max_avail: float) -> None:
        with cb_lock:
            rejects.append((job_id, required, max_avail))

    sched = JobScheduler(
        concurrency=2,
        available_slots=[0, 1],
        slot_capacities={0: 24.0, 1: 12.0},
        capacity_reject_callback=on_reject,
    )

    seen: list[str] = []
    seen_lock = threading.Lock()
    done = threading.Event()

    def small_task(name: str, last: bool = False):
        def fn(_slot: int) -> None:
            with seen_lock:
                seen.append(name)
            if last:
                done.set()
        return fn

    # Head requires 48GB — larger than every slot. Should be rejected.
    sched.submit("oversized", lambda _s: None, vram_required=48.0)
    sched.submit("b", small_task("b"), vram_required=8.0)
    sched.submit("c", small_task("c", last=True), vram_required=4.0)

    assert done.wait(timeout=3.0)
    assert sorted(seen) == ["b", "c"]
    assert len(rejects) == 1
    job_id, required, max_avail = rejects[0]
    assert job_id == "oversized"
    assert required == 48.0
    assert max_avail == 24.0
    sched.stop(timeout=2.0)


def test_no_priority_inversion() -> None:
    """slot_capacities={0: 24}, queue [(A, 8GB), (B, 4GB)].

    Slot 0 can run *both*, so strict FIFO requires A to run first.
    Smaller B must never jump the line just because the slot has
    plenty of headroom.
    """
    sched = JobScheduler(
        concurrency=1,
        available_slots=[0],
        slot_capacities={0: 24.0},
    )
    seen: list[str] = []
    lock = threading.Lock()
    done = threading.Event()

    def make(name: str, last: bool = False):
        def fn(_slot: int) -> None:
            with lock:
                seen.append(name)
            if last:
                done.set()
        return fn

    sched.submit("a", make("a"), vram_required=8.0)
    sched.submit("b", make("b", last=True), vram_required=4.0)

    assert done.wait(timeout=2.0)
    assert seen == ["a", "b"], seen
    sched.stop(timeout=2.0)




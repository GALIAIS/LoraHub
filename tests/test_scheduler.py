"""Tests for the in-memory job scheduler."""

from __future__ import annotations

import threading
import time

import pytest

from lorahub.api.scheduler import JobScheduler


def test_scheduler_runs_submitted_tasks() -> None:
    sched = JobScheduler(concurrency=1)
    flag = threading.Event()
    sched.submit("j1", lambda _slot: flag.set())
    assert flag.wait(timeout=2.0)
    sched.stop(timeout=2.0)


def test_scheduler_runs_tasks_in_fifo_order() -> None:
    sched = JobScheduler(concurrency=1)
    seen: list[str] = []
    done = threading.Event()

    def task_a(_slot: int) -> None:
        seen.append("a")

    def task_b(_slot: int) -> None:
        seen.append("b")
        done.set()

    sched.submit("a", task_a)
    sched.submit("b", task_b)
    assert done.wait(timeout=2.0)
    assert seen == ["a", "b"]
    sched.stop(timeout=2.0)


def test_scheduler_serialises_overlapping_tasks() -> None:
    sched = JobScheduler(concurrency=1)
    overlap_detected = threading.Event()
    in_flight = threading.Lock()
    finished = threading.Event()
    counter = {"n": 0}

    def slow(_slot: int) -> None:
        if not in_flight.acquire(blocking=False):
            overlap_detected.set()
            return
        try:
            time.sleep(0.05)
            counter["n"] += 1
            if counter["n"] == 2:
                finished.set()
        finally:
            in_flight.release()

    sched.submit("a", slow)
    sched.submit("b", slow)
    assert finished.wait(timeout=3.0)
    assert not overlap_detected.is_set()
    sched.stop(timeout=2.0)


def test_scheduler_exposes_queue_depth_and_pending_ids() -> None:
    """Spawn no workers (concurrency built but stop before start), inspect queue."""
    sched = JobScheduler(concurrency=1)
    # Don't start workers — use a sentinel that holds the worker.
    barrier = threading.Event()

    def block(_slot: int) -> None:
        barrier.wait(timeout=2.0)

    # First task occupies the worker; subsequent submits pile up.
    sched.submit("hold", block)
    sched.submit("a", lambda _s: None)
    sched.submit("b", lambda _s: None)
    # Allow the scheduler thread to pick up the first task.
    time.sleep(0.05)

    pending = sched.pending_job_ids()
    assert pending == ["a", "b"]
    assert sched.queue_depth() == 2

    barrier.set()
    sched.stop(timeout=2.0)


def test_scheduler_exposes_available_slots_for_multi_gpu() -> None:
    sched = JobScheduler(concurrency=2, available_slots=[3, 5])
    slots_seen: list[int] = []
    started = threading.Barrier(2, timeout=2.0)
    done = threading.Event()
    lock = threading.Lock()
    counter = {"n": 0}

    def task(slot: int) -> None:
        with lock:
            slots_seen.append(slot)
            counter["n"] += 1
        started.wait()
        if counter["n"] == 2:
            done.set()

    sched.submit("a", task)
    sched.submit("b", task)
    assert done.wait(timeout=3.0)
    assert sorted(slots_seen) == [3, 5]
    sched.stop(timeout=2.0)


def test_scheduler_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        JobScheduler(concurrency=0)


def test_scheduler_rejects_mismatched_slot_list() -> None:
    with pytest.raises(ValueError, match="available_slots"):
        JobScheduler(concurrency=2, available_slots=[0])

"""Tests for LocalReflectionExecutor lifecycle: submit after shutdown, double shutdown, pending drain."""

from __future__ import annotations

import time

import pytest
from langchain_core.runnables import Runnable
from langgraph.store.memory import InMemoryStore

from langmem import ReflectionExecutor


class _NoopReflector(Runnable):
    namespace = ("test",)

    def invoke(self, payload):
        return {"echo": payload}


def test_submit_after_shutdown_raises():
    """Calling submit() after shutdown() must raise RuntimeError, not hang forever."""
    store = InMemoryStore()
    executor = ReflectionExecutor(_NoopReflector(), store=store)
    executor.shutdown(wait=True)

    with pytest.raises(RuntimeError, match="shut down"):
        executor.submit({"x": 1}, {"configurable": {}}, thread_id="post")


def test_double_shutdown_is_safe():
    """Calling shutdown() twice must not raise."""
    store = InMemoryStore()
    executor = ReflectionExecutor(_NoopReflector(), store=store)
    executor.shutdown(wait=True)
    executor.shutdown(wait=True)  # second call must be a no-op


def test_pending_tasks_drain_on_shutdown():
    """Tasks submitted before shutdown must complete when shutdown(wait=True)."""
    store = InMemoryStore()
    executor = ReflectionExecutor(_NoopReflector(), store=store)
    try:
        f = executor.submit({"x": 1}, {"configurable": {}}, thread_id="t1")
        result = f.result(timeout=5)
        assert result == {"echo": {"x": 1}}
    finally:
        executor.shutdown(wait=True)


def test_shutdown_cancel_futures():
    """shutdown(cancel_futures=True) must not raise on a fresh executor."""
    store = InMemoryStore()
    executor = ReflectionExecutor(_NoopReflector(), store=store)
    executor.shutdown(wait=True, cancel_futures=True)


def test_same_thread_id_replacement_callback_no_deadlock():
    """Replacing a pending task whose future has a done_callback that re-enters
    submit() must not deadlock. The old future.cancel() runs outside the
    admission lock so user callbacks can safely re-enter."""
    store = InMemoryStore()
    executor = ReflectionExecutor(_NoopReflector(), store=store)
    try:
        f1 = executor.submit({"v": 1}, {"configurable": {}}, thread_id="shared")

        # Attach a callback that submits another task when f1 is cancelled.
        # If cancel() is called while holding the admission lock, this callback
        # would block on the same lock -> deadlock.
        def reenter(_fut):
            executor.submit({"v": 2}, {"configurable": {}}, thread_id="other")

        f1.add_done_callback(reenter)
        # Second submit with same thread_id cancels f1 and triggers the callback.
        f2 = executor.submit({"v": 2}, {"configurable": {}}, thread_id="shared")
        assert f1.cancelled()
        # Both callbacks and the new task settled without deadlock.
        assert f2.result(timeout=5) == {"echo": {"v": 2}}
    finally:
        executor.shutdown(wait=True)


def test_concurrent_submit_shutdown_no_orphan_future():
    """Hammer concurrent submit() + shutdown(): every future returned by a
    successful submit must either complete or be cancelled; none may hang."""
    store = InMemoryStore()
    executor = ReflectionExecutor(_NoopReflector(), store=store)
    futures = []
    errors = []

    def producer():
        i = 0
        while True:
            try:
                f = executor.submit({"i": i}, {"configurable": {}}, thread_id=f"t{i}")
                futures.append(f)
            except RuntimeError:
                return  # post-shutdown rejection
            i += 1

    import threading

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    time.sleep(0.005)  # let it submit a handful of tasks
    executor.shutdown(wait=True)
    t.join(timeout=5)
    for f in futures:
        try:
            f.result(timeout=2)
        except Exception as e:
            errors.append(repr(e))
    assert not errors, f"orphan futures / errors: {errors[:3]}"

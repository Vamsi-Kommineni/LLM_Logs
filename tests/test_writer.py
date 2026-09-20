from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Sequence

import pytest

import llm_logs as ll
from tests.conftest import flushed


class SlowSink(ll.InMemorySink):
    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    def write_batch(self, records: Sequence[ll.Record]) -> None:
        time.sleep(self.delay)
        super().write_batch(records)


class BlockedSink(ll.InMemorySink):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def write_batch(self, records: Sequence[ll.Record]) -> None:
        self.release.wait(30)
        super().write_batch(records)


class BrokenSink:
    def __init__(self) -> None:
        self.calls = 0

    def write_batch(self, records: Sequence[ll.Record]) -> None:
        self.calls += 1
        raise OSError("disk on fire")

    def flush(self) -> None:
        raise OSError("still on fire")

    def close(self) -> None:
        raise OSError("and again")


def work(prompt: str) -> str:
    return prompt[::-1]


traced_work = ll.trace(work)


@pytest.mark.slow
def test_a_slow_sink_adds_under_a_millisecond_to_the_traced_call() -> None:
    """The core promise. A sink that needs a full second per batch must not be felt by callers.

    Medians over many interleaved calls, compared against an untraced baseline,
    keep this stable on noisy CI machines.
    """
    sink = SlowSink(delay=1.0)
    ll.configure(sinks=[sink], batch_size=50, flush_interval=0.05)
    prompt = "What is the capital of France? " * 20
    for _ in range(50):  # warm up, and get the writer thread stuck in its first slow batch
        traced_work(prompt)
        work(prompt)

    baseline: list[int] = []
    traced: list[int] = []
    for _ in range(400):
        started = time.perf_counter_ns()
        work(prompt)
        baseline.append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        traced_work(prompt)
        traced.append(time.perf_counter_ns() - started)

    overhead_ms = (statistics.median(traced) - statistics.median(baseline)) / 1e6
    worst_ms = max(traced) / 1e6
    sink.delay = 0.0  # let teardown finish quickly
    assert overhead_ms < 1.0, f"median overhead {overhead_ms:.3f} ms"
    assert worst_ms < 250, f"a call took {worst_ms:.1f} ms; the caller was blocked by the sink"
    assert ll.stats().dropped == 0


def test_a_full_queue_drops_and_counts_instead_of_blocking() -> None:
    sink = BlockedSink()
    ll.configure(sinks=[sink], queue_size=10, batch_size=5, flush_interval=0.01)
    started = time.perf_counter()
    for _ in range(200):
        traced_work("x")
    elapsed = time.perf_counter() - started
    stats = ll.stats()
    assert elapsed < 1.0, "callers must never wait for the writer"
    assert stats.enqueued + stats.dropped == 200
    assert stats.dropped >= 150
    assert stats.enqueued <= 10 + 5 + 5  # queue, plus batches the writer already took

    sink.release.set()
    assert ll.flush(5)
    assert len(sink.records) == ll.stats().enqueued == ll.stats().written


def test_a_failing_sink_does_not_affect_the_others() -> None:
    broken, healthy = BrokenSink(), ll.InMemorySink()
    ll.configure(sinks=[broken, healthy], flush_interval=0.01)
    for _ in range(10):
        traced_work("x")
    assert len(flushed(healthy)) == 10
    stats = ll.stats()
    assert stats.written == 10 and stats.failed == 10
    assert stats.failed_by_sink == {"BrokenSink": 10}
    assert stats.written_by_sink == {"InMemorySink": 10}
    assert broken.calls >= 1
    assert ll.shutdown(), "a sink that fails to close must not break shutdown"
    assert healthy.closed


def test_flush_respects_its_timeout_and_then_drains() -> None:
    sink = SlowSink(delay=0.4)
    ll.configure(sinks=[sink], flush_interval=0.01)
    traced_work("x")
    started = time.perf_counter()
    assert ll.flush(timeout=0.05) is False
    assert time.perf_counter() - started < 0.3
    assert ll.flush(timeout=5) is True
    assert len(sink.records) == 1 and sink.flushed >= 1


def test_flush_covers_everything_submitted_before_it(sink: ll.InMemorySink) -> None:
    for number in range(1_000):
        traced_work(str(number))
    assert ll.flush(10)
    assert [r.input["prompt"] for r in sink.records] == [str(n) for n in range(1_000)]


def test_flush_is_not_starved_by_a_busy_producer(sink: ll.InMemorySink) -> None:
    stop = threading.Event()

    def produce() -> None:
        while not stop.is_set():
            traced_work("busy")

    producer = threading.Thread(target=produce)
    producer.start()
    try:
        time.sleep(0.05)
        assert ll.flush(5)
    finally:
        stop.set()
        producer.join()


def test_records_arrive_in_batches() -> None:
    sizes: list[int] = []

    class Counting(ll.InMemorySink):
        def write_batch(self, records: Sequence[ll.Record]) -> None:
            sizes.append(len(records))
            super().write_batch(records)

    sink = Counting()
    ll.configure(sinks=[sink], batch_size=25, flush_interval=5.0)
    for _ in range(100):
        traced_work("x")
    assert len(flushed(sink)) == 100
    assert max(sizes) <= 25 and len(sizes) < 100


def test_many_threads_can_trace_at_once(sink: ll.InMemorySink) -> None:
    def hammer(thread: int) -> None:
        for call in range(250):
            traced_work(f"{thread}-{call}")

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = flushed(sink)
    assert len(records) == 2_000
    assert len({r.span_id for r in records}) == 2_000
    assert ll.stats().dropped == 0


def test_the_writer_thread_is_a_daemon_and_stops_on_shutdown(sink: ll.InMemorySink) -> None:
    traced_work("x")
    writers = [t for t in threading.enumerate() if t.name == "llm-logs-writer"]
    assert len(writers) == 1 and writers[0].daemon
    assert ll.shutdown()
    assert not writers[0].is_alive()
    assert len(sink.records) == 1, "shutdown drains what was queued"


def test_no_thread_is_started_until_something_is_traced() -> None:
    ll.configure(sinks=[ll.InMemorySink()])
    assert not [t for t in threading.enumerate() if t.name == "llm-logs-writer"]
    assert ll.flush(1)

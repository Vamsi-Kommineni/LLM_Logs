"""How much time does ``@trace`` add to a call?

    uv run python benchmarks/overhead.py

Measures the per-call cost on the caller's thread for a few payload sizes, with
the writer draining into a sink that discards everything (so the numbers are
about capture, not about disk). Traced and untraced calls are interleaved and
the median of each is compared, which keeps the result stable on a busy machine.

Numbers depend on the machine. What should hold anywhere: the overhead is a
fraction of a millisecond, it stops growing once the payload exceeds
``max_payload_chars``, and a slow sink does not change it. The p99 column can
show a few milliseconds: when the writer thread is busy, the calling thread can
wait up to one interpreter switch interval (5 ms by default) for the GIL.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import llm_logs as ll

CALLS = 3_000


class NullSink:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def write_batch(self, records: Sequence[ll.Record]) -> None:
        if self.delay:
            time.sleep(self.delay)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def call(messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
    return {"role": "assistant", "content": "A short answer."}


def measure(
    plain: Callable[..., Any], traced: Callable[..., Any], payload: Any
) -> dict[str, float]:
    for _ in range(200):
        plain(payload)
        traced(payload)
    base: list[int] = []
    with_trace: list[int] = []
    for _ in range(CALLS):
        started = time.perf_counter_ns()
        plain(payload)
        base.append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        traced(payload)
        with_trace.append(time.perf_counter_ns() - started)
    with_trace.sort()
    return {
        "median_us": (statistics.median(with_trace) - statistics.median(base)) / 1_000,
        "p99_us": with_trace[int(len(with_trace) * 0.99)] / 1_000,
    }


def messages_of(chars: int) -> list[dict[str, str]]:
    text = "The quick brown fox jumps over the lazy dog. " * (chars // 45 + 1)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text[:chars]},
    ]


def main() -> None:
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"{CALLS:,} interleaved calls per row; overhead = median(traced) - median(untraced)\n")
    print(f"{'scenario':<44}{'median overhead':>18}{'p99 traced call':>18}")

    scenarios: list[tuple[str, dict[str, Any], int]] = [
        ("200-char prompt", {}, 200),
        ("2,000-char prompt", {}, 2_000),
        ("20,000-char prompt (at the limit)", {}, 20_000),
        ("2,000,000-char prompt (truncated)", {}, 2_000_000),
        ("2,000-char prompt, capture_content=False", {"capture_content": False}, 2_000),
        ("2,000-char prompt, sampled out", {"sample_rate": 0.0}, 2_000),
        ("2,000-char prompt, sink needs 1 s per batch", {"slow": True}, 2_000),
    ]
    traced = ll.trace(call)
    for label, options, chars in scenarios:
        options = dict(options)
        sink = NullSink(delay=1.0 if options.pop("slow", False) else 0.0)
        ll.configure(sinks=[sink], queue_size=100_000, **options)
        result = measure(call, traced, messages_of(chars))
        dropped = ll.stats().dropped
        sink.delay = 0.0
        ll.shutdown(10)
        note = f"   ({dropped} dropped)" if dropped else ""
        print(f"{label:<44}{result['median_us']:>15.1f} µs{result['p99_us']:>15.1f} µs{note}")


if __name__ == "__main__":
    sys.exit(main())

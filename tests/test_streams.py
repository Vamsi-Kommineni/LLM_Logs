from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

import llm_logs as ll
from tests.conftest import FakeLLM, FakeStream, flushed

llm = FakeLLM(chunks=("Hel", "lo", " wor", "ld"))


@ll.trace
def stream_call(prompt: str, fail_at: int | None = None) -> FakeStream:
    return llm.stream(prompt, fail_at=fail_at)


@ll.trace
def astream_call(prompt: str, fail_at: int | None = None) -> AsyncIterator[str]:
    return llm.astream(prompt, fail_at=fail_at)


def only(sink: ll.InMemorySink) -> ll.Record:
    records = flushed(sink)
    assert len(records) == 1, f"expected exactly one record, got {len(records)}"
    return records[0]


# 1
def test_full_consumption(sink: ll.InMemorySink) -> None:
    stream = stream_call("hi")
    assert flushed(sink) == [], "nothing is recorded until the stream ends"
    assert "".join(stream) == "Hello world"
    record = only(sink)
    assert (record.streamed, record.stream_outcome, record.status) == (True, "completed", "ok")
    assert record.output == "Hello world"
    assert record.time_to_first_chunk_ms is not None
    assert 0 <= record.time_to_first_chunk_ms <= record.duration_ms


# 2
def test_early_break_then_close(sink: ll.InMemorySink) -> None:
    stream = stream_call("hi")
    for chunk in stream:
        if chunk == "lo":
            break
    stream.close()
    stream.close()  # closing twice must not produce a second record
    record = only(sink)
    assert (record.stream_outcome, record.status, record.output) == ("closed_early", "ok", "Hello")
    assert stream.closed, "close() reaches the wrapped stream"


# 3
def test_exception_mid_stream(sink: ll.InMemorySink) -> None:
    stream = stream_call("hi", fail_at=2)
    received = []
    with pytest.raises(ConnectionError, match="stream broke"):
        for chunk in stream:
            received.append(chunk)
    record = only(sink)
    assert received == ["Hel", "lo"]
    assert (record.stream_outcome, record.status, record.error_type) == (
        "error",
        "error",
        "ConnectionError",
    )
    assert record.output == "Hello", "what arrived before the failure is kept"


# 4
async def test_async_stream(sink: ll.InMemorySink) -> None:
    chunks = [chunk async for chunk in astream_call("hi")]
    assert chunks == ["Hel", "lo", " wor", "ld"]
    record = only(sink)
    assert (record.stream_outcome, record.output) == ("completed", "Hello world")


async def test_async_stream_error_and_early_close(sink: ll.InMemorySink) -> None:
    with pytest.raises(ConnectionError):
        async for _ in astream_call("hi", fail_at=1):
            pass
    stream = astream_call("hi")
    async for _ in stream:
        break
    await stream.aclose()  # type: ignore[attr-defined]
    first, second = flushed(sink)
    assert (first.stream_outcome, first.error_type) == ("error", "ConnectionError")
    assert (second.stream_outcome, second.output) == ("closed_early", "Hel")


# 5
class StreamManager:
    """Shaped like Anthropic's ``client.messages.stream()``: the stream comes from ``__enter__``."""

    def __init__(self) -> None:
        self.inner = FakeStream(["a", "b", "c"])
        self.exited = False

    def __enter__(self) -> FakeStream:
        return self.inner

    def __exit__(self, *exc: object) -> None:
        self.exited = True


def test_context_manager_stream(sink: ll.InMemorySink) -> None:
    manager = StreamManager()

    @ll.trace(stream=True)
    def open_stream(prompt: str) -> StreamManager:
        return manager

    with open_stream("hi") as stream:
        assert stream.helper() == "sdk helper still works"
        assert list(stream) == ["a", "b", "c"]
    record = only(sink)
    assert (record.stream_outcome, record.output) == ("completed", "abc")
    assert manager.exited

    with open_stream("again"):  # no ``as``: must not look abandoned while the block runs
        gc.collect()
        assert len(flushed(sink)) == 1
    assert flushed(sink)[1].stream_outcome == "closed_early"


def test_stream_used_as_its_own_context_manager(sink: ll.InMemorySink) -> None:
    with stream_call("hi") as stream:
        assert next(stream) == "Hel"
    record = only(sink)
    assert (record.stream_outcome, record.output) == ("closed_early", "Hel")


def test_an_error_in_the_callers_loop_body_is_not_an_llm_error(sink: ll.InMemorySink) -> None:
    with pytest.raises(ZeroDivisionError), stream_call("hi") as stream:
        for _ in stream:
            raise ZeroDivisionError
    record = only(sink)
    assert (record.stream_outcome, record.status) == ("closed_early", "ok")


# 6
def test_abandoned_stream_is_recorded_when_collected(sink: ll.InMemorySink) -> None:
    stream = stream_call("hi")
    assert next(stream) == "Hel"
    del stream
    gc.collect()
    record = only(sink)
    assert (record.stream_outcome, record.status, record.output) == ("abandoned", "ok", "Hel")


# 7
def test_traced_generator_function(sink: ll.InMemorySink) -> None:
    @ll.trace
    def generate(prompt: str) -> Iterator[str]:
        with ll.span("inside-generator"):
            yield "one"
        yield "two"
        return "return value"

    generator = generate("hi")
    assert flushed(sink) == [], "a generator body does not run until it is iterated"
    collected = []
    try:
        while True:
            collected.append(next(generator))
    except StopIteration as stop:
        assert stop.value == "return value"
    assert collected == ["one", "two"]

    inner, outer = flushed(sink)
    assert (outer.stream_outcome, outer.output, outer.streamed) == ("completed", "onetwo", True)
    assert inner.parent_span_id == outer.span_id, "spans opened inside the generator nest under it"

    from llm_logs import context

    assert context.current() is None, "the generator's span never leaks to the consumer"


def test_traced_generator_supports_send_throw_close_and_errors(sink: ll.InMemorySink) -> None:
    @ll.trace
    def echo() -> Any:
        received = yield "ready"
        try:
            yield f"got {received}"
        except KeyError:
            yield "handled"
        raise RuntimeError("generator failed")

    generator = echo()
    assert next(generator) == "ready"
    assert generator.send("ping") == "got ping"
    assert generator.throw(KeyError()) == "handled"
    with pytest.raises(RuntimeError, match="generator failed"):
        next(generator)

    closing = echo()
    next(closing)
    closing.close()

    failed, closed = flushed(sink)
    assert (failed.stream_outcome, failed.error_type) == ("error", "RuntimeError")
    assert failed.output == "readygot pinghandled"
    assert (closed.stream_outcome, closed.status) == ("closed_early", "ok")


async def test_traced_async_generator_function(sink: ll.InMemorySink) -> None:
    @ll.trace
    async def generate(prompt: str) -> AsyncIterator[str]:
        for piece in ("x", "y", "z"):
            await asyncio.sleep(0)
            yield piece

    assert [piece async for piece in generate("hi")] == ["x", "y", "z"]
    closing = generate("hi")
    async for _ in closing:
        break
    await closing.aclose()  # type: ignore[attr-defined]
    completed, closed = flushed(sink)
    assert (completed.stream_outcome, completed.output) == ("completed", "xyz")
    assert (closed.stream_outcome, closed.output) == ("closed_early", "x")


# ------------------------------------------------------------- properties


def test_the_proxy_is_transparent(sink: ll.InMemorySink) -> None:
    stream = stream_call("hi")
    assert stream.response == "raw-http-response"
    assert stream.helper() == "sdk helper still works"
    stream.custom = 5  # type: ignore[attr-defined]
    assert stream._llm_logs_inner.custom == 5  # type: ignore[attr-defined]
    assert "FakeStream" in repr(stream)
    with pytest.raises(AttributeError):
        _ = stream.does_not_exist  # type: ignore[attr-defined]
    list(stream)


def test_time_to_first_chunk_is_measured_from_the_call(sink: ll.InMemorySink) -> None:
    @ll.trace
    def slow_start() -> Iterator[str]:
        time.sleep(0.05)
        yield "first"
        time.sleep(0.05)
        yield "second"

    list(slow_start())
    record = only(sink)
    assert record.time_to_first_chunk_ms is not None
    assert 40 <= record.time_to_first_chunk_ms < record.duration_ms - 40


def test_stream_false_leaves_the_result_alone(sink: ll.InMemorySink) -> None:
    @ll.trace(stream=False)
    def raw() -> FakeStream:
        return FakeStream(["a"])

    assert isinstance(raw(), FakeStream)
    record = only(sink)
    assert record.streamed is False and record.stream_outcome is None


def test_non_text_chunks_and_bounded_accumulation() -> None:
    sink = ll.InMemorySink()
    ll.configure(sinks=[sink], max_payload_chars=1_000, flush_interval=0.01)

    @ll.trace
    def objects() -> Iterator[dict[str, int]]:
        yield from ({"n": n} for n in range(3))

    @ll.trace
    def endless_text() -> Iterator[str]:
        yield "START "
        for _ in range(5_000):
            yield "filler words "
        yield "END"

    list(objects())
    list(endless_text())
    small, big = flushed(sink)
    assert small.output == [{"n": 0}, {"n": 1}, {"n": 2}]
    assert big.truncated and big.output.startswith("START") and big.output.endswith("END")
    assert len(big.to_json()) < 3_000


def test_unsampled_streams_pass_through_and_record_nothing() -> None:
    sink = ll.InMemorySink()
    ll.configure(sinks=[sink], sample_rate=0.0, flush_interval=0.01)

    @ll.trace
    def generate() -> Iterator[str]:
        yield "a"

    assert "".join(stream_call("hi")) == "Hello world"
    assert list(generate()) == ["a"]
    assert flushed(sink) == [] and ll.stats().enqueued == 0


# ------------------------------------------------- async managers and helpers


class AsyncFakeStream:
    """Async counterpart of FakeStream, with an SDK-style helper that bypasses iteration."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = list(chunks)
        self.closed = False
        self.text_stream = self._texts()

    async def _texts(self) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk.upper()

    def __aiter__(self) -> AsyncFakeStream:
        return self

    async def __anext__(self) -> str:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def __aenter__(self) -> AsyncFakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True

    async def close(self) -> None:
        self.closed = True


class AsyncStreamManager:
    def __init__(self) -> None:
        self.inner = AsyncFakeStream(["a", "b"])
        self.exited = False

    async def __aenter__(self) -> AsyncFakeStream:
        return self.inner

    async def __aexit__(self, *exc: object) -> None:
        self.exited = True


async def test_async_context_manager_stream(sink: ll.InMemorySink) -> None:
    manager = AsyncStreamManager()

    @ll.trace(stream=True)
    async def open_stream() -> AsyncStreamManager:
        return manager

    async with await open_stream() as stream:
        assert [chunk async for chunk in stream] == ["a", "b"]
    record = only(sink)
    assert (record.stream_outcome, record.output, manager.exited) == ("completed", "ab", True)


async def test_async_stream_as_its_own_context_manager_and_close(sink: ll.InMemorySink) -> None:
    @ll.trace
    async def open_stream() -> AsyncFakeStream:
        return AsyncFakeStream(["a", "b", "c"])

    async with await open_stream() as stream:
        assert await stream.__anext__() == "a"
    closing = await open_stream()
    assert await closing.__anext__() == "a"
    await closing.close()  # the SDK's close() is a coroutine here; it is forwarded as is
    first, second = flushed(sink)
    assert (first.stream_outcome, first.output) == ("closed_early", "a")
    assert (second.stream_outcome, second.output) == ("closed_early", "a")
    assert closing.closed


async def test_helpers_that_bypass_the_proxy_still_time_the_first_chunk(
    sink: ll.InMemorySink,
) -> None:
    @ll.trace
    async def open_stream() -> AsyncFakeStream:
        return AsyncFakeStream(["a", "b"])

    async with await open_stream() as stream:
        assert [text async for text in stream.text_stream] == ["A", "B"]
    record = only(sink)
    assert record.time_to_first_chunk_ms is not None
    assert record.stream_outcome == "closed_early" and record.output is None


def test_a_plain_generator_result_can_be_used_in_a_with_block(sink: ll.InMemorySink) -> None:
    @ll.trace
    def make() -> Iterator[str]:
        return (piece for piece in ("x", "y"))  # a generator object, not a generator function

    with make() as stream:
        assert next(stream) == "x"
    record = only(sink)
    assert (record.stream_outcome, record.output) == ("closed_early", "x")


def test_a_stream_finished_from_another_thread_is_recorded_once(sink: ll.InMemorySink) -> None:
    import threading

    stream = stream_call("hi")
    next(stream)
    closers = [threading.Thread(target=stream.close) for _ in range(8)]
    for thread in closers:
        thread.start()
    for thread in closers:
        thread.join()
    assert only(sink).stream_outcome == "closed_early"

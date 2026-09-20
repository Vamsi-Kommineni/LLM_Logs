from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import threading
from typing import Any

import pytest

import llm_logs as ll
from llm_logs import capture, context
from tests.conftest import FakeLLM, flushed

SECRET = "gsk_THISISAFAKEKEYTHATMUSTNEVERBELOGGED00"


# ------------------------------------------------------------ arguments


def test_args_kwargs_defaults_and_params_are_captured(sink: ll.InMemorySink) -> None:
    @ll.trace
    def ask(prompt: str, temperature: float = 0.2, **kwargs: Any) -> str:
        return prompt.upper()

    assert ask("hello", model="m-1", max_tokens=16, tag="x") == "HELLO"
    (record,) = flushed(sink)
    assert record.kind == "llm" and record.status == "ok"
    assert record.name.endswith("ask")
    assert record.model == "m-1"
    assert record.params == {"temperature": 0.2, "max_tokens": 16}
    assert record.input == {"prompt": "hello", "tag": "x"}  # params are not repeated
    assert record.output == "HELLO"
    assert record.duration_ms >= 0 and record.end_time >= record.start_time


def test_capture_args_ignore_args_and_self(sink: ll.InMemorySink) -> None:
    class Service:
        def __init__(self) -> None:
            self.api_key = SECRET

        @ll.trace(ignore_args=["context"])
        def ask(self, question: str, context: str) -> str:
            return "a"

        @ll.trace(capture_args=["question"])
        def only(self, question: str, context: str) -> str:
            return "b"

    service = Service()
    service.ask("q1", "private context")
    service.only("q2", "private context")
    first, second = flushed(sink)
    assert first.input == {"question": "q1"}
    assert second.input == {"question": "q2"}


def test_a_client_holding_an_api_key_never_reaches_the_record(sink: ll.InMemorySink) -> None:
    client = FakeLLM()

    @ll.trace
    def ask(client: FakeLLM, prompt: str, headers: dict[str, str]) -> dict[str, Any]:
        return client.complete(prompt)

    ask(client, "hi", {"Authorization": f"Bearer {SECRET}", "x-request": "1"})
    (record,) = flushed(sink)
    dumped = record.to_json()
    assert SECRET not in dumped
    assert record.input["client"] == "<tests.conftest.FakeLLM>"
    assert record.input["headers"] == {"Authorization": "[REDACTED]", "x-request": "1"}


def test_direct_wrapping_of_a_bound_method(sink: ll.InMemorySink) -> None:
    client = FakeLLM()
    complete = ll.trace(client.complete, name="fake.complete", provider="fake")
    complete("hi", model="m-2", temperature=0.0)
    (record,) = flushed(sink)
    assert (record.name, record.provider, record.model) == ("fake.complete", "fake", "m-2")
    assert record.params == {"temperature": 0.0}


def test_capture_content_off_keeps_params_but_no_payloads() -> None:
    sink = ll.InMemorySink()
    ll.configure(sinks=[sink], capture_content=False, flush_interval=0.01)

    @ll.trace
    def ask(prompt: str, temperature: float = 0.5) -> str:
        return "secret answer"

    ask("secret question")
    (record,) = flushed(sink)
    assert record.input is None and record.output is None
    assert record.params == {"temperature": 0.5}


def test_oversized_payloads_are_truncated_at_both_ends_and_flagged() -> None:
    sink = ll.InMemorySink()
    ll.configure(sinks=[sink], max_payload_chars=1_000, flush_interval=0.01)

    @ll.trace
    def ask(prompt: str) -> str:
        return "short"

    ask("BEGIN " + "filler text " * 2_000 + "END")
    (record,) = flushed(sink)
    assert record.truncated
    assert record.input["prompt"].startswith("BEGIN") and record.input["prompt"].endswith("END")
    assert len(record.to_json()) < 2_500


# --------------------------------------------------------------- errors


def test_an_exception_is_recorded_and_reraised_unchanged(sink: ll.InMemorySink) -> None:
    boom = ValueError("bad prompt")

    @ll.trace
    def ask(prompt: str) -> str:
        raise boom

    with pytest.raises(ValueError) as caught:
        ask("x")
    assert caught.value is boom
    (record,) = flushed(sink)
    assert (record.status, record.error_type, record.error_message) == (
        "error",
        "ValueError",
        "bad prompt",
    )
    assert record.output is None


async def test_cancellation_is_recorded_and_propagates(sink: ll.InMemorySink) -> None:
    started = asyncio.Event()

    @ll.trace
    async def slow(prompt: str) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    task = asyncio.create_task(slow("x"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (record,) = flushed(sink)
    assert (record.status, record.error_type) == ("error", "CancelledError")


def test_base_exceptions_are_recorded_too(sink: ll.InMemorySink) -> None:
    @ll.trace
    def ask() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ask()
    (record,) = flushed(sink)
    assert record.error_type == "KeyboardInterrupt"


def test_a_wrong_call_still_raises_the_callers_type_error(sink: ll.InMemorySink) -> None:
    @ll.trace
    def ask(prompt: str) -> str:
        return prompt

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        ask()  # type: ignore[call-arg]
    (record,) = flushed(sink)
    assert record.error_type == "TypeError"


# ---------------------------------------------------- never raise, never block


def test_failures_inside_the_library_do_not_reach_the_caller(
    sink: ll.InMemorySink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("library bug")

    @ll.trace
    def ask(prompt: str) -> str:
        return "fine"

    caplog.set_level(logging.WARNING, logger="llm_logs")
    for target in ("to_jsonable", "Record"):
        monkeypatch.setattr(capture, target, explode)
        assert ask("x") == "fine"
    monkeypatch.setattr(context, "child_of_current", explode)
    assert ask("x") == "fine"
    with ll.span("s"):
        pass
    assert ll.stats().internal_errors >= 3
    assert any("internal error" in message for message in caplog.messages)


def test_internal_warnings_are_rate_limited(
    sink: ll.InMemorySink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        context, "child_of_current", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bug"))
    )

    @ll.trace
    def ask() -> str:
        return "fine"

    caplog.set_level(logging.WARNING, logger="llm_logs")
    for _ in range(50):
        ask()
    assert len([m for m in caplog.messages if "internal error in start" in m]) == 1
    assert ll.stats().internal_errors == 50


def test_unconfigured_is_a_passthrough_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    @ll.trace
    def ask(prompt: str) -> str:
        return prompt

    caplog.set_level(logging.WARNING, logger="llm_logs")
    assert [ask("a"), ask("b")] == ["a", "b"]
    with ll.span("s"):
        pass
    assert len([m for m in caplog.messages if "configure() was never called" in m]) == 1


def test_after_shutdown_tracing_is_a_passthrough(sink: ll.InMemorySink) -> None:
    @ll.trace
    def ask(prompt: str) -> str:
        return prompt

    ask("before")
    assert ll.shutdown()
    assert ll.shutdown(), "shutdown is idempotent"
    assert ask("after") == "after"
    assert [r.input for r in sink.records] == [{"prompt": "before"}]
    assert sink.closed


def test_decorating_a_non_callable_fails_at_import_time() -> None:
    with pytest.raises(TypeError, match=r"use @trace\(name="):
        ll.trace("my-span-name")  # type: ignore[call-overload]


# ------------------------------------------------------------------ spans


def test_nested_spans_get_the_right_parents(sink: ll.InMemorySink) -> None:
    @ll.trace
    def ask(prompt: str) -> str:
        return "a"

    with ll.span("pipeline", session_id="s1", user_id="u1", metadata={"route": "/chat"}) as outer:
        with ll.span("retrieve") as inner:
            inner.set(output=["doc1", "doc2"])
            inner.metadata["documents"] = 2
        ask("q")
    ask("outside")

    records = {r.name.split(".")[-1] + str(i): r for i, r in enumerate(flushed(sink))}
    retrieve, inner_ask, pipeline, outside = records.values()
    assert pipeline.parent_span_id is None and pipeline.kind == "span"
    assert retrieve.parent_span_id == pipeline.span_id == outer.span_id
    assert inner_ask.parent_span_id == pipeline.span_id
    assert {retrieve.trace_id, inner_ask.trace_id} == {pipeline.trace_id}
    assert outside.trace_id != pipeline.trace_id and outside.parent_span_id is None
    # session and user flow down from the enclosing span
    assert (inner_ask.session_id, inner_ask.user_id) == ("s1", "u1")
    assert outside.session_id is None
    assert pipeline.metadata == {"route": "/chat"}
    assert retrieve.metadata == {"documents": 2} and retrieve.output == ["doc1", "doc2"]


def test_a_span_records_the_error_that_passes_through_it(sink: ll.InMemorySink) -> None:
    with pytest.raises(LookupError), ll.span("outer"):
        raise LookupError("nope")
    (record,) = flushed(sink)
    assert (record.status, record.error_type) == ("error", "LookupError")


async def test_concurrent_tasks_do_not_leak_context(sink: ll.InMemorySink) -> None:
    @ll.trace
    async def ask(prompt: str) -> str:
        await asyncio.sleep(0.01)
        return prompt

    async def conversation(number: int) -> None:
        async with ll.span(f"conv-{number}", session_id=f"s{number}"):
            await asyncio.sleep(0.005 * (number % 3))
            await ask(f"q{number}")

    await asyncio.gather(*(conversation(n) for n in range(20)))
    records = flushed(sink)
    spans = {r.session_id: r for r in records if r.kind == "span"}
    calls = [r for r in records if r.kind == "llm"]
    assert len(spans) == 20 and len(calls) == 20
    for call in calls:
        parent = spans[call.session_id]
        assert call.parent_span_id == parent.span_id and call.trace_id == parent.trace_id
        assert call.input == {"prompt": f"q{call.session_id[1:]}"}
    assert context.current() is None


def test_threads_need_copy_context(sink: ll.InMemorySink) -> None:
    @ll.trace
    def ask(prompt: str) -> str:
        return prompt

    with ll.span("parent") as parent:
        bare = threading.Thread(target=ask, args=("bare",))
        carried = threading.Thread(target=contextvars.copy_context().run, args=(ask, "carried"))
        for thread in (bare, carried):
            thread.start()
            thread.join()
    by_prompt = {r.input["prompt"]: r for r in flushed(sink) if r.kind == "llm"}
    assert by_prompt["bare"].trace_id != parent.trace_id
    assert by_prompt["carried"].parent_span_id == parent.span_id


# --------------------------------------------------------------- sampling


def test_sampling_is_per_trace_never_partial() -> None:
    sink = ll.InMemorySink()
    ll.configure(sinks=[sink], sample_rate=0.5, flush_interval=0.01)

    @ll.trace
    def ask(prompt: str) -> str:
        return prompt

    for number in range(300):
        with ll.span("pipeline", metadata={"n": number}):
            ask("a")
            with ll.span("inner"):
                ask("b")
    by_trace: dict[str, int] = {}
    for record in flushed(sink):
        by_trace[record.trace_id] = by_trace.get(record.trace_id, 0) + 1
    assert set(by_trace.values()) == {4}, "a sampled trace is always complete"
    assert 90 < len(by_trace) < 210


def test_sampling_is_deterministic_in_the_trace_id() -> None:
    trace_id = "0123456789abcdef0123456789abcdef"
    assert context.is_sampled(trace_id, 1.0) and not context.is_sampled(trace_id, 0.0)
    assert {context.is_sampled(trace_id, 0.3) for _ in range(10)} in ({True}, {False})


# ------------------------------------------------------- function identity


def test_decorated_functions_keep_their_nature() -> None:
    @ll.trace
    def sync(a: int) -> int:
        """doc"""
        return a

    @ll.trace(name="x")
    async def coro(a: int) -> int:
        return a

    @ll.trace
    def gen(a: int) -> Any:
        yield a

    @ll.trace
    async def agen(a: int) -> Any:
        yield a

    assert sync.__name__ == "sync" and sync.__doc__ == "doc"
    assert list(inspect.signature(sync).parameters) == ["a"]
    assert inspect.iscoroutinefunction(coro)
    assert inspect.isgeneratorfunction(gen)
    assert inspect.isasyncgenfunction(agen)

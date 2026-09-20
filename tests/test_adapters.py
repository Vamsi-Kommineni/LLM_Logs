"""Adapters, tested against fixtures parsed into the providers' own SDK types. No network."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anthropic.types
import groq.types.chat
import openai.types
import openai.types.chat
import openai.types.responses
import pytest
from pydantic import TypeAdapter

import llm_logs as ll
from llm_logs.adapters.base import Extracted, GenericAccumulator
from tests.conftest import FakeStream, flushed

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def one(sink: ll.InMemorySink) -> ll.Record:
    (record,) = flushed(sink)
    return record


def deltas(chunks: list[dict[str, Any]], field: str) -> str:
    return "".join(
        choice["delta"].get(field) or "" for chunk in chunks for choice in chunk.get("choices", [])
    )


# --------------------------------------------------------------------- groq


def test_groq_response_is_detected_and_fully_extracted(sink: ll.InMemorySink) -> None:
    fixture = load("groq_chat")
    response = groq.types.chat.ChatCompletion.model_validate(fixture)

    @ll.trace
    def ask(client: object, prompt: str) -> groq.types.chat.ChatCompletion:
        return response

    assert ask(object(), "What is 2 + 2?") is response
    record = one(sink)
    usage = fixture["usage"]
    assert record.provider == "groq" and record.operation == "chat"
    assert record.response_model == fixture["model"]
    assert record.model == fixture["model"], "falls back to the response when the request hides it"
    assert record.response_id == "fixture-id"
    assert record.finish_reasons == ["stop"]
    assert record.input_tokens == usage["prompt_tokens"]
    assert record.output_tokens == usage["completion_tokens"]
    assert record.reasoning_output_tokens == usage["completion_tokens_details"]["reasoning_tokens"]
    assert record.provider_extras["queue_time"] == usage["queue_time"]
    assert set(record.provider_extras) >= {
        "queue_time",
        "prompt_time",
        "completion_time",
        "total_time",
        "service_tier",
        "system_fingerprint",
    }
    message = fixture["choices"][0]["message"]
    assert record.output["role"] == "assistant"
    assert record.output["content"] == message["content"]
    assert record.output["reasoning"] == message["reasoning"]
    assert "tool_calls" not in record.output, "None fields are left out"


def test_request_fields_come_from_the_actual_call(sink: ll.InMemorySink) -> None:
    response = groq.types.chat.ChatCompletion.model_validate(load("groq_chat"))

    def create(**kwargs: Any) -> Any:
        return response

    traced = ll.trace(create, name="chat.completions.create")
    traced(
        model="requested-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
        max_completion_tokens=256,
        reasoning_effort="low",
        stream=False,
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    record = one(sink)
    assert record.model == "requested-model" and record.response_model != "requested-model"
    assert record.params == {
        "temperature": 0,
        "max_completion_tokens": 256,
        "reasoning_effort": "low",
        "stream": False,
    }
    assert set(record.input) == {"messages", "tools"}


@pytest.mark.parametrize("name", ["groq_chat_stream", "groq_chat_stream_include_usage"])
def test_groq_stream_matches_what_the_server_reported(sink: ll.InMemorySink, name: str) -> None:
    raw = load(name)
    chunks = [groq.types.chat.ChatCompletionChunk.model_validate(chunk) for chunk in raw]

    @ll.trace
    def ask(prompt: str) -> FakeStream:
        return FakeStream(chunks)

    assert list(ask("What is 2 + 2?")) == chunks, "chunks pass through untouched"
    record = one(sink)
    reported = next(
        c.get("usage") or (c.get("x_groq") or {}).get("usage")
        for c in reversed(raw)
        if c.get("usage") or (c.get("x_groq") or {}).get("usage")
    )
    assert record.provider == "groq", "detected from the first chunk"
    assert (record.streamed, record.stream_outcome) == (True, "completed")
    assert record.input_tokens == reported["prompt_tokens"]
    assert record.output_tokens == reported["completion_tokens"]
    assert record.finish_reasons == ["stop"]
    assert record.output["content"] == deltas(raw, "content")
    assert record.output["reasoning"] == deltas(raw, "reasoning")
    assert record.response_model == raw[0]["model"]
    assert "queue_time" in record.provider_extras
    # Same prompt, streamed or not: the prompt is counted the same way.
    assert record.input_tokens == load("groq_chat")["usage"]["prompt_tokens"]


def test_tool_calls_are_reassembled_from_stream_fragments(sink: ll.InMemorySink) -> None:
    raw = load("groq_tool_call_stream")
    chunks = [groq.types.chat.ChatCompletionChunk.model_validate(chunk) for chunk in raw]
    list(ll.trace(lambda: FakeStream(chunks), name="tools")())
    record = one(sink)
    (call,) = record.output["tool_calls"]
    assert call["function"]["name"] == "get_weather"
    assert "paris" in json.loads(call["function"]["arguments"])["city"].lower()
    assert record.finish_reasons == ["tool_calls"]


# ------------------------------------------------------------------- openai


def test_openai_chat_completion(sink: ll.InMemorySink) -> None:
    response = openai.types.chat.ChatCompletion.model_validate(load("openai_chat"))
    ll.trace(lambda: response, name="openai.chat")()
    record = one(sink)
    assert (record.provider, record.operation, record.finish_reasons) == (
        "openai",
        "chat",
        ["stop"],
    )
    assert (record.input_tokens, record.output_tokens) == (25, 9)
    assert (record.cache_read_input_tokens, record.reasoning_output_tokens) == (16, 6)
    assert record.output == {"role": "assistant", "content": "4", "annotations": []}


def test_openai_chat_stream_with_a_usage_only_final_chunk(sink: ll.InMemorySink) -> None:
    raw = load("openai_chat_stream")
    chunks = [openai.types.chat.ChatCompletionChunk.model_validate(chunk) for chunk in raw]
    list(ll.trace(lambda: FakeStream(chunks), name="openai.stream")())
    record = one(sink)
    assert record.provider == "openai"
    assert record.output == {"role": "assistant", "content": "The answer is 4."}
    assert (record.input_tokens, record.output_tokens, record.finish_reasons) == (25, 9, ["stop"])


def test_openai_responses_api(sink: ll.InMemorySink) -> None:
    response = openai.types.responses.Response.model_validate(load("openai_responses"))
    ll.trace(lambda: response, name="openai.responses")()
    record = one(sink)
    assert (record.provider, record.operation, record.finish_reasons) == (
        "openai",
        "chat",
        ["completed"],
    )
    assert (record.input_tokens, record.output_tokens, record.reasoning_output_tokens) == (25, 9, 6)
    assert (record.cache_read_input_tokens, record.cache_write_input_tokens) == (0, 7)
    assert record.output[1]["content"][0]["text"] == "The answer is 4."


def test_openai_responses_stream_events(sink: ll.InMemorySink) -> None:
    events = TypeAdapter(list[openai.types.responses.ResponseStreamEvent]).validate_python(
        load("openai_responses_stream")
    )
    list(ll.trace(lambda: FakeStream(events), name="openai.responses.stream")())
    record = one(sink)
    assert record.provider == "openai" and record.stream_outcome == "completed"
    assert (record.input_tokens, record.output_tokens) == (25, 9)
    assert record.output[1]["content"][0]["text"] == "The answer is 4."


def test_embeddings_record_counts_not_vectors(sink: ll.InMemorySink) -> None:
    response = openai.types.CreateEmbeddingResponse.model_validate(load("openai_embeddings"))
    ll.trace(lambda: response, name="openai.embeddings")()
    record = one(sink)
    assert record.operation == "embeddings" and record.input_tokens == 5
    assert record.output == {"embeddings": 2, "dimensions": 8}


def test_plain_dicts_work_with_an_explicit_provider(sink: ll.InMemorySink) -> None:
    """For OpenAI-compatible servers called over raw HTTP, with no SDK in sight."""
    payload = load("openai_chat")
    ll.trace(lambda: payload, name="raw", provider="my-vllm")()
    ll.trace(lambda: payload, name="typed", provider="openai")()
    unknown, typed = flushed(sink)
    assert unknown.provider == "my-vllm" and unknown.input_tokens is None
    assert unknown.output["choices"][0]["message"]["content"] == "4"
    assert (typed.provider, typed.input_tokens, typed.output["content"]) == ("openai", 25, "4")


# ---------------------------------------------------------------- anthropic


def test_anthropic_message(sink: ll.InMemorySink) -> None:
    message = anthropic.types.Message.model_validate(load("anthropic_message"))
    ll.trace(lambda: message, name="anthropic.messages")()
    record = one(sink)
    assert (record.provider, record.operation, record.finish_reasons) == (
        "anthropic",
        "chat",
        ["end_turn"],
    )
    assert record.input_tokens == 12 + 100 + 2000, "input always means the full prompt"
    assert (record.cache_read_input_tokens, record.cache_write_input_tokens) == (2000, 100)
    assert record.output_tokens == 9
    assert record.output["content"][0] == {"type": "text", "text": "The answer is 4."}
    assert record.provider_extras == {"service_tier": "standard"}


def test_anthropic_stream_events(sink: ll.InMemorySink) -> None:
    events = TypeAdapter(list[anthropic.types.RawMessageStreamEvent]).validate_python(
        load("anthropic_stream")
    )
    list(ll.trace(lambda: FakeStream(events), name="anthropic.stream")())
    record = one(sink)
    assert record.provider == "anthropic" and record.finish_reasons == ["tool_use"]
    assert (record.input_tokens, record.output_tokens) == (2112, 42)
    text, tool = record.output["content"]
    assert text == {"type": "text", "text": "Let me check the weather."}
    assert (tool["type"], tool["name"]) == ("tool_use", "get_weather")
    assert json.loads(tool["input"]) == {"city": "Paris"}


def test_anthropic_stream_manager_read_through_a_helper(sink: ll.InMemorySink) -> None:
    """Reading ``stream.text_stream`` bypasses the proxy; the SDK's snapshot fills the record."""
    final = anthropic.types.Message.model_validate(load("anthropic_message"))

    class MessageStream:
        __module__ = "anthropic.lib.streaming._messages"

        def __init__(self) -> None:
            self.text_stream = (piece for piece in ("The answer", " is 4."))
            self.current_message_snapshot = final

        def __iter__(self) -> MessageStream:
            return self

        def __next__(self) -> Any:
            raise StopIteration

    class MessageStreamManager:
        __module__ = "anthropic.lib.streaming._messages"

        def __enter__(self) -> MessageStream:
            return MessageStream()

        def __exit__(self, *exc: object) -> None:
            return None

    @ll.trace  # no stream=True: the adapter recognises the manager
    def open_stream() -> MessageStreamManager:
        return MessageStreamManager()

    with open_stream() as stream:
        assert "".join(stream.text_stream) == "The answer is 4."
    record = one(sink)
    assert (record.provider, record.streamed, record.stream_outcome) == (
        "anthropic",
        True,
        "completed",
    )
    assert record.output_tokens == 9 and record.time_to_first_chunk_ms is not None
    assert record.output["content"][0]["text"] == "The answer is 4."


# ------------------------------------------------------------------ general


def test_tokens_survive_when_content_capture_is_off() -> None:
    sink = ll.InMemorySink()
    ll.configure(sinks=[sink], capture_content=False, flush_interval=0.01)
    response = groq.types.chat.ChatCompletion.model_validate(load("groq_chat"))
    ll.trace(lambda: response, name="quiet")()
    record = one(sink)
    assert record.output is None and record.input is None
    assert record.input_tokens and record.output_tokens and record.response_model


class ShoutAdapter:
    name = "shout"

    def matches(self, obj: Any) -> bool:
        return isinstance(obj, Shout)

    def extract_request(self, arguments: Mapping[str, Any]) -> Extracted:
        return {"model": "shout-1", "params": {"volume": arguments.get("volume")}}

    def extract_response(self, result: Any) -> Extracted:
        if result.text == "explode":
            raise RuntimeError("adapter bug")
        if result.text == "garbage":
            return {"input_tokens": "many", "output": result.text}
        return {"output": result.text.upper(), "output_tokens": len(result.text)}

    def new_accumulator(self, max_chars: int) -> GenericAccumulator:
        return GenericAccumulator(max_chars)

    def is_stream_manager(self, obj: Any) -> bool:
        return False

    def stream_snapshot(self, stream: Any) -> Extracted | None:
        return None


class Shout:
    def __init__(self, text: str) -> None:
        self.text = text


def test_third_party_adapters_can_be_registered(sink: ll.InMemorySink) -> None:
    ll.adapters.register(ShoutAdapter())

    @ll.trace(provider="shout")
    def shout(text: str, volume: int = 11) -> Shout:
        return Shout(text)

    shout("hello")
    record = one(sink)
    assert (record.provider, record.model, record.params) == ("shout", "shout-1", {"volume": 11})
    assert (record.output, record.output_tokens) == ("HELLO", 5)


def test_a_broken_adapter_never_costs_the_record(sink: ll.InMemorySink) -> None:
    ll.adapters.register(ShoutAdapter())
    make = ll.trace(lambda text: Shout(text), name="shout")
    assert make("explode").text == "explode"
    assert make("garbage").text == "garbage"
    exploded, garbage = flushed(sink)
    assert exploded.status == "ok" and exploded.output.endswith("Shout>")
    assert garbage.status == "ok" and garbage.input_tokens is None
    assert ll.stats().internal_errors == 2

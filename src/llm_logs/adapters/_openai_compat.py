"""One extractor for every OpenAI-shaped API.

Groq, OpenAI and the many servers that imitate the Chat Completions API (vLLM,
Ollama, OpenRouter, Together, ...) return the same shapes with small additions.
Everything here reads attributes or keys, so it works on SDK objects and on
plain dicts alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_logs.adapters.base import (
    BoundedText,
    Extracted,
    as_int,
    extract_common_request,
    get,
    module_of,
)

_TIMING_FIELDS = ("queue_time", "prompt_time", "completion_time", "total_time")


def _first_int(*values: Any) -> int | None:
    for value in values:
        number = as_int(value)
        if number is not None:
            return number
    return None


def extract_usage(usage: Any) -> Extracted:
    """Token counts from either naming scheme (Chat Completions or Responses)."""
    if usage is None:
        return {}
    input_details = get(usage, "prompt_tokens_details") or get(usage, "input_tokens_details")
    output_details = get(usage, "completion_tokens_details") or get(usage, "output_tokens_details")
    out: Extracted = {
        "input_tokens": _first_int(get(usage, "prompt_tokens"), get(usage, "input_tokens")),
        "output_tokens": _first_int(get(usage, "completion_tokens"), get(usage, "output_tokens")),
        "cache_read_input_tokens": as_int(get(input_details, "cached_tokens")),
        "cache_write_input_tokens": as_int(get(input_details, "cache_write_tokens")),
        "reasoning_output_tokens": as_int(get(output_details, "reasoning_tokens")),
    }
    timings = {
        name: get(usage, name)
        for name in _TIMING_FIELDS
        if isinstance(get(usage, name), (int, float))
    }
    if timings:
        out["provider_extras"] = timings
    return {k: v for k, v in out.items() if v is not None}


def _extras(result: Any, usage_part: Extracted) -> dict[str, Any]:
    extras: dict[str, Any] = dict(usage_part.get("provider_extras") or {})
    for name in ("system_fingerprint", "service_tier"):
        value = get(result, name)
        if isinstance(value, str):
            extras[name] = value
    request_id = get(get(result, "x_groq"), "id")
    if isinstance(request_id, str):
        extras["request_id"] = request_id
    return extras


def extract_response(result: Any) -> Extracted:
    """Fields from a Chat Completions, Responses, legacy completions or embeddings result."""
    out: Extracted = {}
    response_id, model = get(result, "id"), get(result, "model")
    if isinstance(response_id, str):
        out["response_id"] = response_id
    if isinstance(model, str):
        out["response_model"] = model
    usage_part = extract_usage(get(result, "usage"))
    out.update(usage_part)
    extras = _extras(result, usage_part)
    out.pop("provider_extras", None)
    if extras:
        out["provider_extras"] = extras

    kind = get(result, "object")
    choices = get(result, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        out["operation"] = "text_completion" if kind == "text_completion" else "chat"
        out["finish_reasons"] = [
            reason for c in choices if isinstance(reason := get(c, "finish_reason"), str)
        ]
        messages = [get(c, "message") or get(c, "text") for c in choices]
        messages = [m for m in messages if m is not None]
        if messages:
            out["output"] = messages[0] if len(messages) == 1 else messages
    elif kind == "response":
        out["operation"] = "chat"
        status = get(result, "status")
        if isinstance(status, str):
            out["finish_reasons"] = [status]
        output = get(result, "output")
        if output is not None:
            out["output"] = output
    elif kind == "list":
        data = get(result, "data") or []
        if data and get(data[0], "object") == "embedding":
            out["operation"] = "embeddings"
            vector = get(data[0], "embedding")
            # Vectors are large and say nothing to a human reader.
            out["output"] = {
                "embeddings": len(data),
                "dimensions": len(vector) if isinstance(vector, (list, tuple)) else None,
            }
    return out


class StreamAccumulator:
    """Rebuilds the final message from Chat Completions chunks or Responses events.

    Handles what real servers send: chunks with empty ``choices``, usage on any
    chunk (the last one wins, including Groq's ``x_groq.usage``), separate
    ``content`` and ``reasoning`` deltas, and tool calls split across chunks.
    """

    def __init__(self, max_chars: int) -> None:
        self._content = BoundedText(max_chars)
        self._reasoning = BoundedText(max_chars)
        self._role: str | None = None
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finish_reasons: list[str] = []
        self._fields: Extracted = {}
        self._usage: Any = None
        self._final_response: Any = None
        self._last_chunk: Any = None

    def add(self, chunk: Any) -> None:
        event_type = get(chunk, "type")
        if isinstance(event_type, str) and event_type.startswith("response."):
            self._add_responses_event(event_type, chunk)
            return
        self._last_chunk = chunk
        for name, key in (("id", "response_id"), ("model", "response_model")):
            value = get(chunk, name)
            if isinstance(value, str) and value:
                self._fields[key] = value
        usage = get(chunk, "usage") or get(get(chunk, "x_groq"), "usage")
        if usage is not None:
            self._usage = usage
        for choice in get(chunk, "choices") or ():
            reason = get(choice, "finish_reason")
            if isinstance(reason, str):
                self._finish_reasons.append(reason)
            delta = get(choice, "delta")
            if delta is None:
                continue
            role = get(delta, "role")
            if isinstance(role, str):
                self._role = role
            content = get(delta, "content")
            if isinstance(content, str):
                self._content.add(content)
            reasoning = get(delta, "reasoning") or get(delta, "reasoning_content")
            if isinstance(reasoning, str):
                self._reasoning.add(reasoning)
            for call in get(delta, "tool_calls") or ():
                self._add_tool_call(call)

    def _add_tool_call(self, call: Any) -> None:
        index = as_int(get(call, "index")) or 0
        if index not in self._tool_calls and len(self._tool_calls) >= 64:
            return
        slot = self._tool_calls.setdefault(
            index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
        )
        call_id = get(call, "id")
        if isinstance(call_id, str):
            slot["id"] = call_id
        function = get(call, "function")
        name, arguments = get(function, "name"), get(function, "arguments")
        if isinstance(name, str):
            slot["function"]["name"] += name
        if isinstance(arguments, str) and len(slot["function"]["arguments"]) < 20_000:
            slot["function"]["arguments"] += arguments

    def _add_responses_event(self, event_type: str, event: Any) -> None:
        if event_type == "response.output_text.delta":
            delta = get(event, "delta")
            if isinstance(delta, str):
                self._content.add(delta)
        elif event_type in ("response.completed", "response.incomplete", "response.failed"):
            self._final_response = get(event, "response")

    def result(self) -> Extracted:
        if self._final_response is not None:
            out = extract_response(self._final_response)
            if "output" not in out and self._content:
                out["output"] = {"role": "assistant", "content": self._content.text()}
            return out
        out = dict(self._fields)
        usage_part = extract_usage(self._usage)
        out.update(usage_part)
        extras = _extras(self._last_chunk, usage_part)
        out.pop("provider_extras", None)
        if extras:
            out["provider_extras"] = extras
        out["operation"] = "chat"
        if self._finish_reasons:
            out["finish_reasons"] = self._finish_reasons
        message: dict[str, Any] = {"role": self._role or "assistant"}
        if self._content:
            message["content"] = self._content.text()
        if self._reasoning:
            message["reasoning"] = self._reasoning.text()
        if self._tool_calls:
            message["tool_calls"] = [self._tool_calls[i] for i in sorted(self._tool_calls)]
        if len(message) > 1:
            out["output"] = message
        if self._content.truncated or self._reasoning.truncated:
            out["truncated"] = True
        return out


class OpenAICompatibleAdapter:
    """Shared behaviour; subclasses set the name and the SDK module prefix."""

    name = "openai_compatible"
    module_prefixes: tuple[str, ...] = ()

    def matches(self, obj: Any) -> bool:
        return module_of(obj).startswith(self.module_prefixes)

    def extract_request(self, arguments: Mapping[str, Any]) -> Extracted:
        return extract_common_request(arguments)

    def extract_response(self, result: Any) -> Extracted:
        return extract_response(result)

    def new_accumulator(self, max_chars: int) -> StreamAccumulator:
        return StreamAccumulator(max_chars)

    def is_stream_manager(self, obj: Any) -> bool:
        return self.matches(obj) and type(obj).__name__.endswith("StreamManager")

    def stream_snapshot(self, stream: Any) -> Extracted | None:
        return None

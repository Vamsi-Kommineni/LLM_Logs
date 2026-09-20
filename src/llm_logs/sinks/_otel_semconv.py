"""OpenTelemetry GenAI semantic conventions: the only place attribute names appear.

The conventions live in https://github.com/open-telemetry/semantic-conventions-genai
and are at *Development* status, so names still change. Names are written out
here as literals rather than imported from ``opentelemetry-semantic-conventions``:
that package lags the spec (at the time of writing it still says
``gen_ai.usage.cache_creation.input_tokens`` where the spec says ``cache_write``),
and its GenAI module is marked incubating and may move.

Check the spec before editing this file, and update ``SPEC_CHECKED`` when you do.
"""

from __future__ import annotations

import json
from typing import Any

from llm_logs.record import Record

SPEC_CHECKED = "2026-09-20"

# --- inference span -----------------------------------------------------------
OPERATION_NAME = "gen_ai.operation.name"  # required
PROVIDER_NAME = "gen_ai.provider.name"  # required
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_STREAM = "gen_ai.request.stream"
REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_ID = "gen_ai.response.id"
RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
RESPONSE_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"  # seconds
CONVERSATION_ID = "gen_ai.conversation.id"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"  # includes cached tokens
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
USAGE_CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
USAGE_CACHE_WRITE = "gen_ai.usage.cache_write.input_tokens"
USAGE_REASONING = "gen_ai.usage.reasoning.output_tokens"
ERROR_TYPE = "error.type"

# Opt-in: may contain personal data.
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"

# General conventions reused here.
USER_ID = "user.id"
SESSION_ID = "session.id"

# Record.params key -> request attribute. Several spellings mean "max tokens".
REQUEST_PARAMS = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "seed": "gen_ai.request.seed",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "max_tokens": "gen_ai.request.max_tokens",
    "max_completion_tokens": "gen_ai.request.max_tokens",
    "max_output_tokens": "gen_ai.request.max_tokens",
    "max_new_tokens": "gen_ai.request.max_tokens",
}

# Everything this library knows that the conventions have no name for.
OWN_PREFIX = "llm_logs."
DEFAULT_OPERATION = "chat"
UNKNOWN_PROVIDER = "unknown"

Primitive = str | bool | int | float


def span_name(record: Record) -> str:
    """``"{operation} {model}"`` for model calls, the plain name for other spans."""
    if record.kind != "llm":
        return record.name
    operation = record.operation or DEFAULT_OPERATION
    return f"{operation} {record.model}" if record.model else operation


def _text_part(text: Any) -> dict[str, Any]:
    return {"type": "text", "content": text if isinstance(text, str) else _dump(text)}


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append({"type": "reasoning", "content": reasoning})
    content = message.get("content")
    if message.get("role") == "tool":
        parts.append(
            {"type": "tool_call_response", "id": message.get("tool_call_id"), "response": content}
        )
        return parts
    if isinstance(content, str):
        if content:
            parts.append(_text_part(content))
    elif isinstance(content, list):
        for block in content:
            parts.append(_block_part(block))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {} if isinstance(call, dict) else {}
        parts.append(
            {
                "type": "tool_call",
                "id": call.get("id") if isinstance(call, dict) else None,
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return parts


def _block_part(block: Any) -> dict[str, Any]:
    """One OpenAI content part or Anthropic content block."""
    if not isinstance(block, dict):
        return _text_part(block)
    kind = block.get("type")
    if kind in ("text", "output_text", "input_text"):
        return _text_part(block.get("text", ""))
    if kind == "thinking":
        return {"type": "reasoning", "content": block.get("thinking", "")}
    if kind == "tool_use":
        return {
            "type": "tool_call",
            "id": block.get("id"),
            "name": block.get("name"),
            "arguments": block.get("input"),
        }
    if kind == "tool_result":
        return {
            "type": "tool_call_response",
            "id": block.get("tool_use_id"),
            "response": block.get("content"),
        }
    return {"type": str(kind or "unknown")}  # images, audio: say what it was, carry nothing


def input_messages(value: Any) -> list[dict[str, Any]]:
    """Convert a record's ``input`` to the conventions' message list."""
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        messages = []
        for message in value["messages"]:
            if isinstance(message, dict):
                messages.append({"role": message.get("role", "user"), "parts": _parts(message)})
            else:
                messages.append({"role": "user", "parts": [_text_part(message)]})
        return messages
    if isinstance(value, dict):
        texts = [v for k, v in value.items() if k != "system" and isinstance(v, str)]
        if len(texts) == 1:
            return [{"role": "user", "parts": [_text_part(texts[0])]}]
    return [{"role": "user", "parts": [_text_part(value)]}]


def system_instructions(value: Any) -> list[dict[str, Any]] | None:
    """Instructions passed apart from the chat history, as Anthropic's ``system`` is."""
    if isinstance(value, dict) and value.get("system"):
        system = value["system"]
        if isinstance(system, list):
            return [_block_part(block) for block in system]
        return [_text_part(system)]
    return None


def output_messages(value: Any, finish_reasons: list[str]) -> list[dict[str, Any]]:
    """Convert a record's ``output`` to the conventions' message list."""
    candidates = value if isinstance(value, list) and _all_messages(value) else [value]
    messages = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, dict) and ("role" in candidate or "content" in candidate):
            message = {"role": candidate.get("role", "assistant"), "parts": _parts(candidate)}
        else:
            message = {"role": "assistant", "parts": [_text_part(candidate)]}
        if index < len(finish_reasons):
            message["finish_reason"] = finish_reasons[index]
        messages.append(message)
    return messages


def _all_messages(items: list[Any]) -> bool:
    return bool(items) and all(isinstance(item, dict) and "role" in item for item in items)


def attributes(record: Record, *, capture_content: bool) -> dict[str, Any]:
    """All span attributes for one record. Values are primitives or lists of primitives."""
    out: dict[str, Any] = {OWN_PREFIX + "name": record.name}
    if record.kind == "llm":
        out[OPERATION_NAME] = record.operation or DEFAULT_OPERATION
        out[PROVIDER_NAME] = record.provider or UNKNOWN_PROVIDER
    elif record.operation:
        out[OPERATION_NAME] = record.operation

    optional: dict[str, Any] = {
        REQUEST_MODEL: record.model,
        RESPONSE_MODEL: record.response_model,
        RESPONSE_ID: record.response_id,
        RESPONSE_FINISH_REASONS: record.finish_reasons or None,
        CONVERSATION_ID: record.session_id,
        SESSION_ID: record.session_id,
        USER_ID: record.user_id,
        USAGE_INPUT_TOKENS: record.input_tokens,
        USAGE_OUTPUT_TOKENS: record.output_tokens,
        USAGE_CACHE_READ: record.cache_read_input_tokens,
        USAGE_CACHE_WRITE: record.cache_write_input_tokens,
        USAGE_REASONING: record.reasoning_output_tokens,
        ERROR_TYPE: record.error_type,
        OWN_PREFIX + "cost": record.cost,
        OWN_PREFIX + "stream_outcome": record.stream_outcome,
        OWN_PREFIX + "truncated": True if record.truncated else None,
    }
    if record.streamed:
        optional[REQUEST_STREAM] = True
    if record.time_to_first_chunk_ms is not None:
        optional[RESPONSE_TIME_TO_FIRST_CHUNK] = record.time_to_first_chunk_ms / 1000.0
    out.update({k: v for k, v in optional.items() if v is not None})

    for name, value in record.params.items():
        target = REQUEST_PARAMS.get(name)
        if target is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            out[target] = value
    stops = record.params.get("stop") or record.params.get("stop_sequences")
    if isinstance(stops, str):
        stops = [stops]
    if isinstance(stops, list) and all(isinstance(s, str) for s in stops):
        out[REQUEST_STOP_SEQUENCES] = stops
    choices = record.params.get("n")
    if isinstance(choices, int) and not isinstance(choices, bool) and choices != 1:
        out[REQUEST_CHOICE_COUNT] = choices

    for prefix, values in (("metadata.", record.metadata), ("provider.", record.provider_extras)):
        for key, value in values.items():
            if isinstance(value, (str, bool, int, float)):
                out[f"{OWN_PREFIX}{prefix}{key}"] = value

    if capture_content:
        if record.input is not None:
            out[INPUT_MESSAGES] = _dump(input_messages(record.input))
            instructions = system_instructions(record.input)
            if instructions:
                out[SYSTEM_INSTRUCTIONS] = _dump(instructions)
            if isinstance(record.input, dict) and record.input.get("tools"):
                out[TOOL_DEFINITIONS] = _dump(record.input["tools"])
        if record.output is not None:
            out[OUTPUT_MESSAGES] = _dump(output_messages(record.output, record.finish_reasons))
    return out

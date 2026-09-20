"""Anthropic Messages API."""

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


def _usage(usage: Any) -> Extracted:
    if usage is None:
        return {}
    uncached = as_int(get(usage, "input_tokens"))
    cache_read = as_int(get(usage, "cache_read_input_tokens"))
    cache_write = as_int(get(usage, "cache_creation_input_tokens"))
    out: Extracted = {
        "output_tokens": as_int(get(usage, "output_tokens")),
        "cache_read_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
    }
    if uncached is not None:
        # Anthropic reports uncached input separately. Records always carry the
        # full prompt size, so the three parts are added up.
        out["input_tokens"] = uncached + (cache_read or 0) + (cache_write or 0)
    return {k: v for k, v in out.items() if v is not None}


def _extras(usage: Any) -> dict[str, Any]:
    tier = get(usage, "service_tier")
    return {"service_tier": tier} if isinstance(tier, str) else {}


def extract_message(message: Any) -> Extracted:
    out: Extracted = {"operation": "chat"}
    message_id, model = get(message, "id"), get(message, "model")
    if isinstance(message_id, str):
        out["response_id"] = message_id
    if isinstance(model, str):
        out["response_model"] = model
    stop_reason = get(message, "stop_reason")
    if isinstance(stop_reason, str):
        out["finish_reasons"] = [stop_reason]
    usage = get(message, "usage")
    out.update(_usage(usage))
    extras = _extras(usage)
    if extras:
        out["provider_extras"] = extras
    content = get(message, "content")
    if content is not None:
        out["output"] = {"role": get(message, "role", "assistant"), "content": content}
    return out


class AnthropicStreamAccumulator:
    """Rebuilds a message from raw stream events.

    The SDK's ``MessageStream`` also yields derived events such as ``text``.
    Only raw event types are read here, so nothing is counted twice.
    """

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars
        self._blocks: dict[int, dict[str, Any]] = {}
        self._texts: dict[int, BoundedText] = {}
        self._fields: Extracted = {}
        self._usage: dict[str, Any] = {}
        self._role = "assistant"

    def add(self, event: Any) -> None:
        event_type = get(event, "type")
        if event_type == "message_start":
            message = get(event, "message")
            for name, key in (("id", "response_id"), ("model", "response_model")):
                value = get(message, name)
                if isinstance(value, str):
                    self._fields[key] = value
            self._role = get(message, "role", "assistant")
            self._merge_usage(get(message, "usage"))
        elif event_type == "content_block_start":
            index = as_int(get(event, "index")) or 0
            block = get(event, "content_block")
            if len(self._blocks) < 64:
                self._blocks[index] = {
                    "type": get(block, "type", "text"),
                    "id": get(block, "id"),
                    "name": get(block, "name"),
                }
                self._texts[index] = BoundedText(self._max_chars)
        elif event_type == "content_block_delta":
            index = as_int(get(event, "index")) or 0
            delta = get(event, "delta")
            piece = get(delta, "text") or get(delta, "partial_json") or get(delta, "thinking")
            if isinstance(piece, str) and index in self._texts:
                self._texts[index].add(piece)
        elif event_type == "message_delta":
            stop_reason = get(get(event, "delta"), "stop_reason")
            if isinstance(stop_reason, str):
                self._fields["finish_reasons"] = [stop_reason]
            self._merge_usage(get(event, "usage"))

    def _merge_usage(self, usage: Any) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "service_tier",
        ):
            value = get(usage, name)
            if value is not None:
                self._usage[name] = value

    def result(self) -> Extracted:
        out: Extracted = {"operation": "chat", **self._fields}
        out.update(_usage(self._usage))
        extras = _extras(self._usage)
        if extras:
            out["provider_extras"] = extras
        content: list[dict[str, Any]] = []
        truncated = False
        for index in sorted(self._blocks):
            block, text = self._blocks[index], self._texts[index]
            truncated = truncated or text.truncated
            kind = block["type"]
            if kind == "tool_use":
                content.append(
                    {"type": kind, "id": block["id"], "name": block["name"], "input": text.text()}
                )
            elif kind == "thinking":
                content.append({"type": kind, "thinking": text.text()})
            else:
                content.append({"type": kind, "text": text.text()})
        if content:
            out["output"] = {"role": self._role, "content": content}
        if truncated:
            out["truncated"] = True
        return out


class AnthropicAdapter:
    name = "anthropic"

    def matches(self, obj: Any) -> bool:
        return module_of(obj).startswith("anthropic.")

    def extract_request(self, arguments: Mapping[str, Any]) -> Extracted:
        return extract_common_request(arguments)

    def extract_response(self, result: Any) -> Extracted:
        return extract_message(result)

    def new_accumulator(self, max_chars: int) -> AnthropicStreamAccumulator:
        return AnthropicStreamAccumulator(max_chars)

    def is_stream_manager(self, obj: Any) -> bool:
        return self.matches(obj) and type(obj).__name__.endswith("StreamManager")

    def stream_snapshot(self, stream: Any) -> Extracted | None:
        # MessageStream keeps the message it has assembled so far. The property
        # asserts when nothing has arrived yet, hence the guard.
        try:
            snapshot = stream.current_message_snapshot
        except Exception:
            return None
        return extract_message(snapshot) if snapshot is not None else None

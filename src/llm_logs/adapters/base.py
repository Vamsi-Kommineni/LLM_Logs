"""Adapter protocol, registry and the helpers adapters share.

An adapter knows one provider's request and response shapes. Adapters are
duck-typed: they read attributes and keys and never import the provider's SDK,
so ``import llm_logs`` stays light and works with no extras installed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any, Protocol

# Field values destined for a Record, keyed by Record field name. ``output`` may
# hold raw Python objects; the capture layer makes it JSON-safe and bounded.
Extracted = dict[str, Any]

# Request parameters worth recording as ``params``, across providers.
PARAM_NAMES = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "max_new_tokens",
        "seed",
        "stop",
        "stop_sequences",
        "frequency_penalty",
        "presence_penalty",
        "n",
        "stream",
        "reasoning_effort",
        "service_tier",
        "tool_choice",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
    }
)


def get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object, whichever ``obj`` is."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    try:
        value = getattr(obj, name, default)
    except Exception:
        return default
    return default if value is None else value


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def module_of(obj: Any) -> str:
    return getattr(type(obj), "__module__", "") or ""


def extract_common_request(arguments: Mapping[str, Any]) -> Extracted:
    """Pick the model and the sampling parameters out of the call's arguments."""
    out: Extracted = {}
    model = arguments.get("model")
    if isinstance(model, str):
        out["model"] = model
    params = {k: v for k, v in arguments.items() if k in PARAM_NAMES and v is not None}
    if params:
        out["params"] = params
    return out


class BoundedText:
    """Accumulates streamed text in bounded memory, keeping the head and the tail."""

    def __init__(self, max_chars: int) -> None:
        self._head_limit = max_chars
        self._tail_limit = max(max_chars // 2, 1)
        self._head: list[str] = []
        self._head_chars = 0
        self._tail: deque[str] = deque()
        self._tail_chars = 0
        self._omitted = 0

    def __bool__(self) -> bool:
        return self._head_chars > 0

    @property
    def truncated(self) -> bool:
        return self._omitted > 0

    def add(self, piece: str) -> None:
        if not piece:
            return
        if not self._tail and self._head_chars + len(piece) <= self._head_limit:
            self._head.append(piece)
            self._head_chars += len(piece)
            return
        self._tail.append(piece)
        self._tail_chars += len(piece)
        while len(self._tail) > 1 and self._tail_chars - len(self._tail[0]) >= self._tail_limit:
            dropped = self._tail.popleft()
            self._tail_chars -= len(dropped)
            self._omitted += len(dropped)

    def text(self) -> str:
        head = "".join(self._head)
        if not self._tail:
            return head
        tail = "".join(self._tail)
        if self._omitted:
            return f"{head}...[{self._omitted} chars omitted]...{tail}"
        return head + tail


class Accumulator(Protocol):
    """Builds the final response fields from the chunks of a stream."""

    def add(self, chunk: Any) -> None: ...

    def result(self) -> Extracted: ...


class Adapter(Protocol):
    name: str

    def matches(self, obj: Any) -> bool:
        """True if ``obj`` (a response, a stream or a chunk) comes from this provider."""
        ...

    def extract_request(self, arguments: Mapping[str, Any]) -> Extracted: ...

    def extract_response(self, result: Any) -> Extracted: ...

    def new_accumulator(self, max_chars: int) -> Accumulator: ...

    def is_stream_manager(self, obj: Any) -> bool:
        """True for context-manager style streams that are not iterators themselves."""
        ...

    def stream_snapshot(self, stream: Any) -> Extracted | None:
        """Final state kept by the SDK's own stream object, if it keeps one."""
        ...


class GenericAccumulator:
    """Used when no adapter recognises the chunks: keeps text, or a bounded list."""

    def __init__(self, max_chars: int) -> None:
        self._text = BoundedText(max_chars)
        self._other: list[Any] = []
        self._other_budget = 200
        self._dropped = 0

    def add(self, chunk: Any) -> None:
        if isinstance(chunk, str):
            self._text.add(chunk)
        elif isinstance(chunk, (bytes, bytearray)):
            self._text.add(f"<bytes len={len(chunk)}>")
        elif len(self._other) < self._other_budget:
            self._other.append(chunk)
        else:
            self._dropped += 1

    def result(self) -> Extracted:
        out: Extracted = {}
        if self._other:
            chunks: list[Any] = list(self._other)
            if self._text:
                chunks.append(self._text.text())
            out["output"] = chunks
        elif self._text:
            out["output"] = self._text.text()
        if self._text.truncated or self._dropped:
            out["truncated"] = True
        return out


_registry: dict[str, Adapter] = {}


def register(adapter: Adapter) -> None:
    """Add or replace an adapter. Third-party adapters use this too."""
    _registry[adapter.name] = adapter


def by_name(name: str | None) -> Adapter | None:
    return _registry.get(name) if name else None


def detect(obj: Any) -> Adapter | None:
    """Find the adapter that recognises ``obj``. Never raises."""
    for adapter in _registry.values():
        try:
            if adapter.matches(obj):
                return adapter
        except Exception:
            continue
    return None

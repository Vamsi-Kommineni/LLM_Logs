"""Transparent proxies around streamed responses.

A traced call that returns a stream is not finished when the function returns.
The proxy passes chunks through untouched, accumulates what the record needs,
and finalises the record exactly once, whichever comes first:

- the stream is exhausted              -> ``completed``
- the stream raises while iterating    -> ``error``
- ``close()`` / ``aclose()`` / leaving a ``with`` block early -> ``closed_early``
- the caller drops the stream without closing it -> ``abandoned`` (from a
  ``weakref.finalize`` callback, so a record is still written)

Proxies delegate every unknown attribute to the wrapped object, so SDK helpers
such as ``stream.response`` or ``stream.get_final_message()`` keep working.
"""

from __future__ import annotations

import inspect
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any

from llm_logs.adapters.base import Accumulator, Adapter, Extracted, GenericAccumulator, detect
from llm_logs.record import StreamOutcome

FinishCallback = Callable[
    [StreamOutcome, "BaseException | None", Extracted, "float | None", "Adapter | None"], None
]


class StreamState:
    """What a stream has produced so far, plus the one-shot ``finish``.

    Kept separate from the proxy on purpose: the finaliser must not hold a
    reference to the proxy it watches, or the proxy would never be collected.
    """

    def __init__(
        self,
        *,
        on_finish: FinishCallback,
        on_error: Callable[[str], None],
        adapter: Adapter | None,
        max_chars: int,
        sampled: bool,
        started: float,
    ) -> None:
        self._on_finish = on_finish
        self._on_error = on_error
        self._adapter = adapter
        self._max_chars = max_chars
        self._sampled = sampled
        self._started = started
        self._lock = threading.Lock()
        self._done = False
        self._accumulator: Accumulator | None = None
        self._first_chunk_at: float | None = None
        self._snapshot_source: Any = None

    @property
    def done(self) -> bool:
        return self._done

    def mark_first_chunk(self) -> None:
        if self._first_chunk_at is None:
            self._first_chunk_at = time.perf_counter()

    def watch_snapshot(self, stream: Any) -> None:
        """Remember the SDK's own stream object; some keep a final snapshot."""
        self._snapshot_source = stream

    def add(self, chunk: Any) -> None:
        if self._done:
            return
        try:
            self.mark_first_chunk()
            if not self._sampled:
                return
            accumulator = self._accumulator
            if accumulator is None:
                if self._adapter is None:
                    self._adapter = detect(chunk)
                accumulator = (
                    self._adapter.new_accumulator(self._max_chars)
                    if self._adapter is not None
                    else GenericAccumulator(self._max_chars)
                )
                self._accumulator = accumulator
            accumulator.add(chunk)
        except Exception:
            self._on_error("stream-add")

    def finish(self, outcome: StreamOutcome, exc: BaseException | None = None) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        try:
            if not self._sampled:
                return
            extracted: Extracted = {}
            if self._accumulator is not None:
                try:
                    extracted = self._accumulator.result()
                except Exception:
                    self._on_error("stream-result")  # still emit the record
            elif self._adapter is not None and self._snapshot_source is not None:
                # Chunks were read through an SDK helper and bypassed the proxy.
                snapshot = self._adapter.stream_snapshot(self._snapshot_source)
                if snapshot:
                    extracted = snapshot
                    if outcome == "closed_early" and snapshot.get("finish_reasons"):
                        outcome = "completed"
            ttfc_ms = (
                None
                if self._first_chunk_at is None
                else (self._first_chunk_at - self._started) * 1000.0
            )
            self._on_finish(outcome, exc, extracted, ttfc_ms, self._adapter)
        except Exception:
            self._on_error("stream-finish")


class _ProxyBase:
    __slots__ = ("__weakref__", "_llm_logs_child", "_llm_logs_inner", "_llm_logs_state")

    _llm_logs_inner: Any
    _llm_logs_state: StreamState
    _llm_logs_child: Any

    def __init__(self, inner: Any, state: StreamState) -> None:
        object.__setattr__(self, "_llm_logs_inner", inner)
        object.__setattr__(self, "_llm_logs_state", state)
        object.__setattr__(self, "_llm_logs_child", None)
        # Runs when the proxy is collected or at interpreter exit. It references
        # the state, never the proxy.
        weakref.finalize(self, state.finish, "abandoned")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_llm_logs_"):
            raise AttributeError(name)
        value = getattr(self._llm_logs_inner, name)
        state = self._llm_logs_state
        if not state.done:
            # Helpers such as Anthropic's ``text_stream`` iterate the SDK stream
            # directly. Tap them so time-to-first-chunk is still measured.
            if inspect.isgenerator(value):
                return _SyncTap(value, state)
            if inspect.isasyncgen(value):
                return _AsyncTap(value, state)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._llm_logs_inner, name, value)

    def __repr__(self) -> str:
        return f"<llm_logs proxy of {self._llm_logs_inner!r}>"

    def _llm_logs_adopt(self, entered: Any, wrap: Callable[[Any, StreamState], Any]) -> Any:
        state = self._llm_logs_state
        state.watch_snapshot(entered)
        if entered is self._llm_logs_inner:
            return self
        if hasattr(type(entered), "__next__") or hasattr(type(entered), "__anext__"):
            child = wrap(entered, state)
            # Keep the child alive as long as the manager, or a ``with`` block
            # without ``as`` would look abandoned the moment it starts.
            object.__setattr__(self, "_llm_logs_child", child)
            return child
        return entered


class SyncStreamProxy(_ProxyBase):
    """Wraps an iterator, which may also be a context manager."""

    __slots__ = ()

    def __iter__(self) -> SyncStreamProxy:
        return self

    def __next__(self) -> Any:
        state = self._llm_logs_state
        try:
            chunk = next(self._llm_logs_inner)
        except StopIteration:
            state.finish("completed")
            raise
        except BaseException as exc:
            state.finish("error", exc)
            raise
        state.add(chunk)
        return chunk

    def close(self) -> Any:
        try:
            closer = getattr(self._llm_logs_inner, "close", None)
            return closer() if closer is not None else None
        finally:
            self._llm_logs_state.finish("closed_early")

    def __enter__(self) -> Any:
        enter = getattr(self._llm_logs_inner, "__enter__", None)
        if enter is None:
            return self
        return self._llm_logs_adopt(enter(), SyncStreamProxy)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            leave = getattr(self._llm_logs_inner, "__exit__", None)
            if leave is not None:
                return leave(exc_type, exc, tb)
            closer = getattr(self._llm_logs_inner, "close", None)
            if closer is not None:
                closer()
            return None
        finally:
            self._llm_logs_state.finish("closed_early")


class SyncManagerProxy(_ProxyBase):
    """Wraps a context manager that yields the real stream from ``__enter__``."""

    __slots__ = ()

    def __enter__(self) -> Any:
        return self._llm_logs_adopt(self._llm_logs_inner.__enter__(), SyncStreamProxy)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return self._llm_logs_inner.__exit__(exc_type, exc, tb)
        finally:
            self._llm_logs_state.finish("closed_early")


class AsyncStreamProxy(_ProxyBase):
    """Wraps an async iterator, which may also be an async context manager."""

    __slots__ = ()

    def __aiter__(self) -> AsyncStreamProxy:
        return self

    async def __anext__(self) -> Any:
        state = self._llm_logs_state
        try:
            chunk = await self._llm_logs_inner.__anext__()
        except StopAsyncIteration:
            state.finish("completed")
            raise
        except BaseException as exc:
            state.finish("error", exc)
            raise
        state.add(chunk)
        return chunk

    def close(self) -> Any:
        # SDKs differ on whether close() is a coroutine; forward whatever it returns.
        try:
            closer = getattr(self._llm_logs_inner, "close", None)
            return closer() if closer is not None else None
        finally:
            self._llm_logs_state.finish("closed_early")

    async def aclose(self) -> None:
        try:
            closer = getattr(self._llm_logs_inner, "aclose", None)
            if closer is not None:
                await closer()
        finally:
            self._llm_logs_state.finish("closed_early")

    async def __aenter__(self) -> Any:
        enter = getattr(self._llm_logs_inner, "__aenter__", None)
        if enter is None:
            return self
        return self._llm_logs_adopt(await enter(), AsyncStreamProxy)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            leave = getattr(self._llm_logs_inner, "__aexit__", None)
            if leave is not None:
                return await leave(exc_type, exc, tb)
            closer = getattr(self._llm_logs_inner, "aclose", None)
            if closer is not None:
                await closer()
            return None
        finally:
            self._llm_logs_state.finish("closed_early")


class AsyncManagerProxy(_ProxyBase):
    """Wraps an async context manager that yields the real stream from ``__aenter__``."""

    __slots__ = ()

    async def __aenter__(self) -> Any:
        return self._llm_logs_adopt(await self._llm_logs_inner.__aenter__(), AsyncStreamProxy)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return await self._llm_logs_inner.__aexit__(exc_type, exc, tb)
        finally:
            self._llm_logs_state.finish("closed_early")


class _SyncTap:
    """Marks the first chunk of an SDK helper iterator. Does not accumulate."""

    def __init__(self, inner: Any, state: StreamState) -> None:
        self._inner = inner
        self._state = state

    def __iter__(self) -> _SyncTap:
        return self

    def __next__(self) -> Any:
        item = next(self._inner)
        self._state.mark_first_chunk()
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _AsyncTap:
    def __init__(self, inner: Any, state: StreamState) -> None:
        self._inner = inner
        self._state = state

    def __aiter__(self) -> _AsyncTap:
        return self

    async def __anext__(self) -> Any:
        item = await self._inner.__anext__()
        self._state.mark_first_chunk()
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def classify(result: Any, stream: bool | None, adapter: Adapter | None) -> str | None:
    """Decide which proxy, if any, a returned value needs."""
    if stream is False or result is None:
        return None
    t = type(result)
    if hasattr(t, "__anext__"):
        return "async"
    if hasattr(t, "__next__"):
        return "sync"
    is_manager = stream is True
    if not is_manager and adapter is not None:
        try:
            is_manager = adapter.is_stream_manager(result)
        except Exception:
            is_manager = False
    if is_manager:
        if hasattr(t, "__aenter__"):
            return "async_manager"
        if hasattr(t, "__enter__"):
            return "sync_manager"
    return None


def wrap(result: Any, kind: str, state: StreamState) -> Any:
    if kind == "sync":
        return SyncStreamProxy(result, state)
    if kind == "async":
        return AsyncStreamProxy(result, state)
    if kind == "sync_manager":
        return SyncManagerProxy(result, state)
    return AsyncManagerProxy(result, state)

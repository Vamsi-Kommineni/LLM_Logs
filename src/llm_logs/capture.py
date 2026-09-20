"""The ``trace`` decorator and the ``span`` context manager.

Everything here runs on the caller's thread, so every step is guarded: a failure
inside the library is counted and logged, and the caller's function still runs
and still returns or raises exactly what it would have without tracing.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from contextvars import Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from types import TracebackType
from typing import Any, ParamSpec, TypeVar, cast, overload

from llm_logs import context
from llm_logs.adapters.base import Adapter, Extracted, by_name, detect, extract_common_request
from llm_logs.config import Config, active, record_internal_error
from llm_logs.context import SpanContext
from llm_logs.record import Kind, Record, StreamOutcome
from llm_logs.serialize import to_jsonable
from llm_logs.streams import StreamState, classify, wrap
from llm_logs.writer import Writer

P = ParamSpec("P")
R = TypeVar("R")

_SMALL_BUDGET = 2_000
_ERROR_MESSAGE_CHARS = 2_000
_ADAPTER_FIELDS = (
    "response_model",
    "response_id",
    "finish_reasons",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
    "operation",
)

try:
    LIB_VERSION = importlib_metadata.version("llm-logs")
except importlib_metadata.PackageNotFoundError:  # pragma: no cover - source checkout
    LIB_VERSION = "0+unknown"


@dataclass(frozen=True, slots=True)
class _Spec:
    name: str
    kind: Kind
    provider: str | None = None
    operation: str | None = None
    capture_args: frozenset[str] | None = None
    ignore_args: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    user_id: str | None = None
    stream: bool | None = None
    signature: inspect.Signature | None = None
    skip_first: str | None = None
    var_keyword: str | None = None


def _safe_str(exc: BaseException) -> str:
    try:
        return str(exc)[:_ERROR_MESSAGE_CHARS]
    except Exception:
        return type(exc).__name__


class _Call:
    """One traced call in flight."""

    __slots__ = (
        "adapter",
        "config",
        "ctx",
        "fields",
        "parent_span_id",
        "spec",
        "start_perf",
        "start_wall",
        "truncated",
        "writer",
    )

    def __init__(
        self,
        spec: _Spec,
        config: Config,
        writer: Writer,
        ctx: SpanContext,
        parent_span_id: str | None,
    ) -> None:
        self.spec = spec
        self.config = config
        self.writer = writer
        self.ctx = ctx
        self.parent_span_id = parent_span_id
        self.adapter: Adapter | None = by_name(spec.provider)
        self.fields: dict[str, Any] = {}
        self.truncated = False
        self.start_wall = datetime.now(UTC)
        self.start_perf = time.perf_counter()

    # ------------------------------------------------------------- context

    def enter(self) -> Token[SpanContext | None] | None:
        try:
            return context.activate(self.ctx)
        except Exception:
            record_internal_error("enter")
            return None

    def exit(self, token: Token[SpanContext | None] | None) -> None:
        if token is not None:
            try:
                context.deactivate(token)
            except Exception:
                record_internal_error("exit")

    # ------------------------------------------------------------- request

    def capture_request(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        try:
            arguments = self._bind(args, kwargs)
            request = (
                self.adapter.extract_request(arguments)
                if self.adapter is not None
                else extract_common_request(arguments)
            )
            taken = {"model"} if "model" in request else set()
            if "model" in request:
                self.fields["model"] = request["model"]
            if request.get("params"):
                taken.update(request["params"])
                self.fields["params"], _ = to_jsonable(request["params"], max_chars=_SMALL_BUDGET)
            if not self.config.capture_content:
                return
            spec = self.spec
            content = {
                name: value
                for name, value in arguments.items()
                if name not in taken
                and name not in spec.ignore_args
                and (spec.capture_args is None or name in spec.capture_args)
            }
            if content:
                self.fields["input"] = self._jsonable(content)
        except Exception:
            record_internal_error("capture-request")

    def _bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        spec = self.spec
        if spec.signature is not None:
            try:
                bound = spec.signature.bind(*args, **kwargs)
            except TypeError:
                pass  # the call itself will raise the caller's TypeError
            else:
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                if spec.skip_first is not None:
                    arguments.pop(spec.skip_first, None)
                if spec.var_keyword is not None:
                    extra = arguments.pop(spec.var_keyword, None) or {}
                    for key, value in extra.items():
                        arguments.setdefault(key, value)
                return arguments
        arguments = dict(kwargs)
        if args:
            arguments.setdefault("args", list(args))
        return arguments

    def _jsonable(self, value: Any, budget: int | None = None) -> Any:
        out, truncated = to_jsonable(
            value,
            max_chars=budget or self.config.max_payload_chars,
            unknown=self.config.unknown_objects,
        )
        self.truncated = self.truncated or truncated
        return out

    # ------------------------------------------------------------ response

    def merge(self, extracted: Extracted, adapter: Adapter | None) -> None:
        if adapter is not None and "provider" not in self.fields:
            self.fields["provider"] = adapter.name
        for name in _ADAPTER_FIELDS:
            value = extracted.get(name)
            if value is not None:
                self.fields[name] = value
        if extracted.get("truncated"):
            self.truncated = True
        if extracted.get("provider_extras"):
            self.fields["provider_extras"] = self._jsonable(
                extracted["provider_extras"], _SMALL_BUDGET
            )
        if self.config.capture_content and extracted.get("output") is not None:
            self.fields["output"] = self._jsonable(extracted["output"])

    def finish(self, result: R) -> R:
        """Record a returned value. Streams are wrapped and recorded when they end."""
        if not self.ctx.sampled:
            return result
        try:
            adapter = self.adapter or detect(result)
            kind = classify(result, self.spec.stream, adapter)
            if kind is not None:
                return cast(R, wrap(result, kind, self.open_stream(adapter)))
            extracted: Extracted = {"output": result}
            if adapter is not None:
                try:
                    extracted = adapter.extract_response(result)
                except Exception:
                    record_internal_error("adapter-response")
            self.merge(extracted, adapter)
            self.emit()
        except Exception:
            record_internal_error("finish")
        return result

    def fail(self, exc: BaseException) -> None:
        if not self.ctx.sampled:
            return
        try:
            self.fields["status"] = "error"
            self.fields["error_type"] = type(exc).__qualname__
            self.fields["error_message"] = _safe_str(exc)
            self.emit()
        except Exception:
            record_internal_error("fail")

    def open_stream(self, adapter: Adapter | None) -> StreamState:
        return StreamState(
            on_finish=self._stream_finished,
            on_error=record_internal_error,
            adapter=adapter,
            max_chars=self.config.max_payload_chars,
            sampled=self.ctx.sampled,
            started=self.start_perf,
        )

    def _stream_finished(
        self,
        outcome: StreamOutcome,
        exc: BaseException | None,
        extracted: Extracted,
        ttfc_ms: float | None,
        adapter: Adapter | None,
    ) -> None:
        self.merge(extracted, adapter)
        self.fields["streamed"] = True
        self.fields["stream_outcome"] = outcome
        self.fields["time_to_first_chunk_ms"] = ttfc_ms
        if exc is not None:
            self.fields["status"] = "error"
            self.fields["error_type"] = type(exc).__qualname__
            self.fields["error_message"] = _safe_str(exc)
        self.emit()

    # ---------------------------------------------------------------- emit

    def emit(self) -> None:
        spec, ctx, fields = self.spec, self.ctx, self.fields
        end_perf = time.perf_counter()
        if fields.get("model") is None and fields.get("response_model") is not None:
            fields["model"] = fields["response_model"]
        if spec.provider is not None:
            fields["provider"] = spec.provider
        if spec.operation is not None:
            fields["operation"] = spec.operation
        base: dict[str, Any] = {
            "lib_version": LIB_VERSION,
            "trace_id": ctx.trace_id,
            "span_id": ctx.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": spec.kind,
            "name": spec.name,
            "session_id": ctx.session_id,
            "user_id": ctx.user_id,
            "start_time": self.start_wall,
            "end_time": datetime.now(UTC),
            "duration_ms": (end_perf - self.start_perf) * 1000.0,
            "truncated": self.truncated,
        }
        if spec.metadata:
            base["metadata"] = self._jsonable(spec.metadata, _SMALL_BUDGET)
            base["truncated"] = self.truncated
        try:
            record = Record(**base, **fields)
        except Exception:
            # An adapter produced something the schema rejects. Keep the call
            # itself on record rather than losing it entirely.
            record_internal_error("record-build")
            core = {k: fields[k] for k in ("status", "error_type", "error_message") if k in fields}
            record = Record(**base, **core)
        self.writer.submit(record)


def _start(spec: _Spec, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _Call | None:
    """Begin a traced call. Returns None when tracing is off or anything goes wrong."""
    try:
        live = active()
        if live is None:
            return None
        config, writer = live
        ctx, parent_span_id = context.child_of_current(
            sample_rate=config.sample_rate, session_id=spec.session_id, user_id=spec.user_id
        )
        call = _Call(spec, config, writer, ctx, parent_span_id)
        if ctx.sampled and spec.kind == "llm":
            call.capture_request(args, kwargs)
        return call
    except Exception:
        record_internal_error("start")
        return None


# ------------------------------------------------------------------ wrappers


def _wrap_sync(func: Callable[P, R], spec: _Spec) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        call = _start(spec, args, kwargs)
        if call is None:
            return func(*args, **kwargs)
        token = call.enter()
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            call.exit(token)
            call.fail(exc)
            raise
        call.exit(token)
        return call.finish(result)

    return wrapper


def _wrap_async(func: Callable[P, Any], spec: _Spec) -> Callable[P, Any]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
        call = _start(spec, args, kwargs)
        if call is None:
            return await func(*args, **kwargs)
        token = call.enter()
        try:
            result = await func(*args, **kwargs)
        except BaseException as exc:  # includes asyncio.CancelledError
            call.exit(token)
            call.fail(exc)
            raise
        call.exit(token)
        return call.finish(result)

    return wrapper


def _open_stream(call: _Call | None) -> StreamState | None:
    if call is None:
        return None
    try:
        return call.open_stream(call.adapter)
    except Exception:
        record_internal_error("open-stream")
        return None


def _resume(call: _Call | None, step: Callable[..., Any], *args: Any) -> Any:
    """Run one step of the wrapped generator with its span active.

    A generator shares its caller's context, so the span is activated only
    while the generator body runs and never leaks to the code consuming it.
    """
    if call is None:
        return step(*args)
    token = call.enter()
    try:
        return step(*args)
    finally:
        call.exit(token)


async def _aresume(call: _Call | None, step: Callable[..., Any], *args: Any) -> Any:
    if call is None:
        return await step(*args)
    token = call.enter()
    try:
        return await step(*args)
    finally:
        call.exit(token)


def _wrap_generator(func: Callable[P, Generator[Any, Any, Any]], spec: _Spec) -> Callable[P, Any]:
    """Keeps the function a real generator function, which frameworks check for."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[Any, Any, Any]:
        call = _start(spec, args, kwargs)
        state = _open_stream(call)
        try:
            generator = func(*args, **kwargs)
            item = _resume(call, generator.__next__)
            while True:
                if state is not None:
                    state.add(item)
                try:
                    sent = yield item
                except GeneratorExit:
                    generator.close()
                    raise
                except BaseException as thrown:
                    item = _resume(call, generator.throw, thrown)
                else:
                    item = _resume(call, generator.send, sent)
        except StopIteration as stop:
            if state is not None:
                state.finish("completed")
            return stop.value
        except GeneratorExit:
            # close() and garbage collection look the same from inside a generator.
            if state is not None:
                state.finish("closed_early")
            raise
        except BaseException as exc:
            if state is not None:
                state.finish("error", exc)
            raise

    return wrapper


def _wrap_async_generator(
    func: Callable[P, AsyncGenerator[Any, Any]], spec: _Spec
) -> Callable[P, Any]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[Any, Any]:
        call = _start(spec, args, kwargs)
        state = _open_stream(call)
        try:
            generator = func(*args, **kwargs)
            item = await _aresume(call, generator.__anext__)
            while True:
                if state is not None:
                    state.add(item)
                try:
                    sent = yield item
                except GeneratorExit:
                    await generator.aclose()
                    raise
                except BaseException as thrown:
                    item = await _aresume(call, generator.athrow, thrown)
                else:
                    item = await _aresume(call, generator.asend, sent)
        except StopAsyncIteration:
            if state is not None:
                state.finish("completed")
            return
        except GeneratorExit:
            if state is not None:
                state.finish("closed_early")
            raise
        except BaseException as exc:
            if state is not None:
                state.finish("error", exc)
            raise

    return wrapper


def _names(value: Iterable[str] | None) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(value)


def _build_spec(func: Callable[..., Any], options: dict[str, Any]) -> _Spec:
    signature: inspect.Signature | None
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None
    skip_first = var_keyword = None
    if signature is not None:
        parameters = list(signature.parameters.values())
        if parameters and parameters[0].name in ("self", "cls"):
            skip_first = parameters[0].name
        for parameter in parameters:
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                var_keyword = parameter.name
    return _Spec(
        name=options["name"] or getattr(func, "__qualname__", None) or type(func).__name__,
        kind="llm",
        provider=options["provider"],
        operation=options["operation"],
        capture_args=_names(options["capture_args"]),
        ignore_args=_names(options["ignore_args"]) or frozenset(),
        metadata=dict(options["metadata"] or {}),
        session_id=options["session_id"],
        user_id=options["user_id"],
        stream=options["stream"],
        signature=signature,
        skip_first=skip_first,
        var_keyword=var_keyword,
    )


@overload
def trace(
    func: Callable[P, R],
    /,
    *,
    name: str | None = ...,
    provider: str | None = ...,
    operation: str | None = ...,
    capture_args: Iterable[str] | None = ...,
    ignore_args: Iterable[str] | None = ...,
    metadata: dict[str, Any] | None = ...,
    session_id: str | None = ...,
    user_id: str | None = ...,
    stream: bool | None = ...,
) -> Callable[P, R]: ...


@overload
def trace(
    *,
    name: str | None = ...,
    provider: str | None = ...,
    operation: str | None = ...,
    capture_args: Iterable[str] | None = ...,
    ignore_args: Iterable[str] | None = ...,
    metadata: dict[str, Any] | None = ...,
    session_id: str | None = ...,
    user_id: str | None = ...,
    stream: bool | None = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def trace(
    func: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    provider: str | None = None,
    operation: str | None = None,
    capture_args: Iterable[str] | None = None,
    ignore_args: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    stream: bool | None = None,
) -> Any:
    """Record every call of a function as one LLM span.

    Works bare (``@trace``), with arguments (``@trace(provider="groq")``) and as
    a plain wrapper (``create = trace(client.chat.completions.create)``). Sync
    functions, coroutine functions, generator functions and async generator
    functions are all supported, and the decorated function keeps its signature.

    Args:
        name: Span name. Defaults to the function's qualified name.
        provider: Provider name, e.g. ``"groq"``. When omitted, the provider is
            detected from the type of the returned object.
        operation: OpenTelemetry operation name such as ``"chat"`` or ``"embeddings"``.
        capture_args: Only these arguments are recorded as ``input``.
        ignore_args: These arguments are never recorded. ``self`` and ``cls``
            are always skipped.
        metadata: Free-form tags stored on every record from this function.
        session_id, user_id: Default identifiers; an enclosing ``span`` wins if
            these are omitted.
        stream: ``None`` wraps returned iterators automatically. ``True`` also
            treats a returned context manager as a stream. ``False`` never wraps.
    """
    options = {
        "name": name,
        "provider": provider,
        "operation": operation,
        "capture_args": capture_args,
        "ignore_args": ignore_args,
        "metadata": metadata,
        "session_id": session_id,
        "user_id": user_id,
        "stream": stream,
    }

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(target):
            raise TypeError(
                f"trace() expects a function, got {type(target).__name__}; "
                'to name a span use @trace(name="...")'
            )
        try:
            spec = _build_spec(target, options)
            if inspect.isasyncgenfunction(target):
                return _wrap_async_generator(target, spec)
            if inspect.isgeneratorfunction(target):
                return _wrap_generator(target, spec)
            if inspect.iscoroutinefunction(target):
                return _wrap_async(target, spec)
            return _wrap_sync(target, spec)
        except Exception:
            record_internal_error("decorate")
            return target

    return decorate if func is None else decorate(func)


class span:
    """Group work into a span. Usable with ``with`` and ``async with``.

    Spans nest: traced calls and spans opened inside become children. The
    identifiers and ``metadata`` are available on the object::

        with ll.span("rag", session_id="s1") as s:
            docs = retrieve(query)
            s.metadata["documents"] = len(docs)
    """

    def __init__(
        self,
        name: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        operation: str | None = None,
    ) -> None:
        self.name = name
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._session_id = session_id
        self._user_id = user_id
        self._operation = operation
        self._call: _Call | None = None
        self._token: Token[SpanContext | None] | None = None
        self._input: Any = None
        self._output: Any = None

    @property
    def trace_id(self) -> str | None:
        return self._call.ctx.trace_id if self._call is not None else None

    @property
    def span_id(self) -> str | None:
        return self._call.ctx.span_id if self._call is not None else None

    def set(self, *, input: Any = None, output: Any = None) -> None:
        """Attach an input and/or output value to this span, e.g. retrieved documents."""
        if input is not None:
            self._input = input
        if output is not None:
            self._output = output

    def __enter__(self) -> span:
        spec = _Spec(
            name=self.name,
            kind="span",
            operation=self._operation,
            session_id=self._session_id,
            user_id=self._user_id,
            metadata=self.metadata,  # shared on purpose: edits made inside the block are kept
        )
        self._call = _start(spec, (), {})
        if self._call is not None:
            self._token = self._call.enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        call = self._call
        if call is None:
            return
        call.exit(self._token)
        if not call.ctx.sampled:
            return
        try:
            if call.config.capture_content:
                if self._input is not None:
                    call.fields["input"] = call._jsonable(self._input)
                if self._output is not None:
                    call.fields["output"] = call._jsonable(self._output)
            if exc is not None:
                call.fail(exc)
            else:
                call.emit()
        except Exception:
            record_internal_error("span-exit")

    async def __aenter__(self) -> span:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)

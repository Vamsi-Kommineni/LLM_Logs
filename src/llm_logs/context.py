"""Trace context carried in ``contextvars``.

``contextvars`` follow ``asyncio`` tasks automatically: each task gets a copy of
the context it was created in, so concurrent tasks never see each other's spans.
Plain threads start with an empty context; pass ``contextvars.copy_context()``
to the thread (``ctx.run(fn)``) to continue a trace there.
"""

from __future__ import annotations

import contextlib
import random
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass

_TRACE_ID_MASK = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class SpanContext:
    trace_id: str
    span_id: str
    sampled: bool
    session_id: str | None = None
    user_id: str | None = None


_current: ContextVar[SpanContext | None] = ContextVar("llm_logs_span", default=None)


def current() -> SpanContext | None:
    return _current.get()


def activate(ctx: SpanContext) -> Token[SpanContext | None]:
    return _current.set(ctx)


def deactivate(token: Token[SpanContext | None]) -> None:
    # A ValueError means the token belongs to another context, e.g. a stream
    # started in one task and finished in another. Nothing to restore then.
    with contextlib.suppress(ValueError):
        _current.reset(token)


def new_trace_id() -> str:
    # The random module is re-seeded in the child after a fork, so forked
    # workers do not produce colliding IDs.
    return f"{random.getrandbits(128) or 1:032x}"


def new_span_id() -> str:
    return f"{random.getrandbits(64) or 1:016x}"


def is_sampled(trace_id: str, rate: float) -> bool:
    """Deterministic head sampling on the low 64 bits of the trace ID.

    Every process that sees the same trace ID reaches the same decision without
    coordination, and the decision is made once per trace, so a trace is either
    complete or absent.
    """
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return (int(trace_id, 16) & _TRACE_ID_MASK) < int(rate * (1 << 64))


def host_otel_parent() -> tuple[str, str] | None:
    """Return ``(trace_id, span_id)`` of the host's active OpenTelemetry span.

    The core never imports OpenTelemetry. If the application has not imported
    it, there cannot be an active span, so ``sys.modules`` is all we look at.
    """
    trace_api = sys.modules.get("opentelemetry.trace")
    if trace_api is None:
        return None
    try:
        span_context = trace_api.get_current_span().get_span_context()
        if not span_context.is_valid:
            return None
        return f"{span_context.trace_id:032x}", f"{span_context.span_id:016x}"
    except Exception:
        return None


def child_of_current(
    *, sample_rate: float, session_id: str | None, user_id: str | None
) -> tuple[SpanContext, str | None]:
    """Build the context for a new span. Returns ``(context, parent_span_id)``."""
    parent = _current.get()
    if parent is not None:
        return (
            SpanContext(
                trace_id=parent.trace_id,
                span_id=new_span_id(),
                sampled=parent.sampled,
                session_id=session_id if session_id is not None else parent.session_id,
                user_id=user_id if user_id is not None else parent.user_id,
            ),
            parent.span_id,
        )
    host = host_otel_parent()
    trace_id, parent_span_id = host if host is not None else (new_trace_id(), None)
    return (
        SpanContext(
            trace_id=trace_id,
            span_id=new_span_id(),
            sampled=is_sampled(trace_id, sample_rate),
            session_id=session_id,
            user_id=user_id,
        ),
        parent_span_id,
    )

"""Export records as OpenTelemetry spans. Needs the ``otel`` extra.

    pip install "llm-logs[otel]"

The exporter reads the standard ``OTEL_EXPORTER_OTLP_*`` environment variables,
so pointing it at a collector or a backend needs no code.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

try:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope
    from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'OtelSink needs OpenTelemetry. Install it with: pip install "llm-logs[otel]"'
    ) from exc

from llm_logs.capture import LIB_VERSION
from llm_logs.record import Record
from llm_logs.sinks import _otel_semconv as semconv

_SAMPLED = TraceFlags(TraceFlags.SAMPLED)


def _context(trace_id: str, span_id: str) -> SpanContext:
    return SpanContext(
        trace_id=int(trace_id, 16), span_id=int(span_id, 16), is_remote=False, trace_flags=_SAMPLED
    )


class OtelSink:
    """Sends each record to an OTLP endpoint as one span.

    Spans are built as finished ``ReadableSpan`` objects and handed straight to
    a ``SpanExporter``. The tracer API cannot be used here: it generates its own
    span IDs, and the records already carry the IDs that tie a trace together
    across the JSONL, SQLite and OpenTelemetry views of the same call.

    Args:
        exporter: Any ``SpanExporter``. Defaults to OTLP over HTTP, configured by
            the standard ``OTEL_EXPORTER_OTLP_*`` variables. The default is
            created lazily on the writer thread, and again after a fork.
        service_name: ``service.name`` of the resource. ``OTEL_SERVICE_NAME``
            and ``OTEL_RESOURCE_ATTRIBUTES`` are honoured as usual.
        capture_content: Send prompts and completions. **Off by default**, even
            when ``configure(capture_content=True)`` keeps them in local files:
            shipping content to another system is a separate privacy decision,
            and the conventions mark these attributes as opt-in.
    """

    def __init__(
        self,
        *,
        exporter: SpanExporter | None = None,
        service_name: str | None = None,
        capture_content: bool = False,
    ) -> None:
        self._given_exporter = exporter
        self._exporter: SpanExporter | None = exporter
        self._pid = os.getpid()
        self._capture_content = capture_content
        self._resource = Resource.create({"service.name": service_name} if service_name else {})
        self._scope = InstrumentationScope("llm_logs", LIB_VERSION)

    def write_batch(self, records: Sequence[Record]) -> None:
        if not records:
            return
        spans = [self.to_span(record) for record in records]
        result = self._current_exporter().export(spans)
        if result is not SpanExportResult.SUCCESS:
            raise RuntimeError("the OpenTelemetry exporter reported a failed export")

    def flush(self) -> None:
        if self._exporter is not None and self._pid == os.getpid():
            self._exporter.force_flush()

    def close(self) -> None:
        exporter, self._exporter = self._exporter, None
        if exporter is not None and self._pid == os.getpid():
            exporter.shutdown()

    def _current_exporter(self) -> SpanExporter:
        pid = os.getpid()
        if self._exporter is None or (self._pid != pid and self._given_exporter is None):
            # First use, or a forked child: network sessions do not survive a
            # fork, so the child builds its own and leaves the inherited one alone.
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            self._exporter = OTLPSpanExporter()
        self._pid = pid
        return self._exporter

    def to_span(self, record: Record) -> ReadableSpan:
        start_ns = int(record.start_time.timestamp() * 1_000_000_000)
        # The duration comes from a monotonic clock; wall-clock subtraction does not.
        end_ns = start_ns + int(record.duration_ms * 1_000_000)
        if record.status == "error":
            status = Status(StatusCode.ERROR, record.error_message or record.error_type)
        else:
            status = Status(StatusCode.UNSET)
        parent = _context(record.trace_id, record.parent_span_id) if record.parent_span_id else None
        return ReadableSpan(
            name=semconv.span_name(record),
            context=_context(record.trace_id, record.span_id),
            parent=parent,
            resource=self._resource,
            attributes=semconv.attributes(record, capture_content=self._capture_content),
            kind=SpanKind.CLIENT if record.kind == "llm" else SpanKind.INTERNAL,
            status=status,
            start_time=start_ns,
            end_time=end_ns,
            instrumentation_scope=self._scope,
        )

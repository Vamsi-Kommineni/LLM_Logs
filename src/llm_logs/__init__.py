"""llm_logs: local-first logging of LLM calls.

import llm_logs as ll

ll.configure(sinks=[ll.JsonlSink("logs/")])

@ll.trace
def ask(client, prompt): ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_logs import adapters, redact
from llm_logs.capture import LIB_VERSION as __version__
from llm_logs.capture import span, trace
from llm_logs.config import ConfigurationError, configure, flush, shutdown, stats
from llm_logs.record import Record
from llm_logs.redact import Redactor
from llm_logs.sinks import InMemorySink, JsonlSink, Sink, SqliteSink
from llm_logs.stats import StatsSnapshot

if TYPE_CHECKING:
    from llm_logs.sinks.otel import OtelSink

__all__ = [
    "ConfigurationError",
    "InMemorySink",
    "JsonlSink",
    "OtelSink",
    "Record",
    "Redactor",
    "Sink",
    "SqliteSink",
    "StatsSnapshot",
    "__version__",
    "adapters",
    "configure",
    "flush",
    "redact",
    "shutdown",
    "span",
    "stats",
    "trace",
]


def __getattr__(name: str) -> Any:
    # OpenTelemetry is only imported when OtelSink is actually used.
    if name == "OtelSink":
        from llm_logs.sinks.otel import OtelSink

        return OtelSink
    raise AttributeError(f"module 'llm_logs' has no attribute {name!r}")

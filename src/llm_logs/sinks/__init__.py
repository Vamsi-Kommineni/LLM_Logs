"""Sinks. ``OtelSink`` lives in ``llm_logs.sinks.otel`` and needs the ``otel`` extra."""

from llm_logs.sinks.base import InMemorySink, Sink
from llm_logs.sinks.jsonl import JsonlSink
from llm_logs.sinks.sqlite import SqliteSink

__all__ = ["InMemorySink", "JsonlSink", "Sink", "SqliteSink"]

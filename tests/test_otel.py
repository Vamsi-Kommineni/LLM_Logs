from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import groq.types.chat
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

import llm_logs as ll
from llm_logs.sinks import _otel_semconv as semconv
from tests.conftest import FakeStream
from tests.test_adapters import load

SECRET_PROMPT = "my account number is 12345"


def configure(exporter: SpanExporter, **sink_options: Any) -> None:
    ll.configure(
        sinks=[ll.OtelSink(exporter=exporter, service_name="test-app", **sink_options)],
        flush_interval=0.01,
    )


def spans_by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    assert ll.flush(5)
    return {span.name: span for span in exporter.get_finished_spans()}


def run_pipeline() -> groq.types.chat.ChatCompletion:
    response = groq.types.chat.ChatCompletion.model_validate(load("groq_chat"))

    def create(**kwargs: Any) -> groq.types.chat.ChatCompletion:
        return response

    traced = ll.trace(create, name="groq.create")
    with ll.span("rag_pipeline", session_id="s1", user_id="u42", metadata={"route": "/chat"}):
        with ll.span("retrieve", operation="retrieval"):
            pass
        traced(
            model=response.model,
            messages=[
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": SECRET_PROMPT},
            ],
            temperature=0.2,
            max_completion_tokens=256,
            stop=["END"],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
    return response


def test_attributes_ids_and_structure() -> None:
    exporter = InMemorySpanExporter()
    configure(exporter)
    response = run_pipeline()
    spans = spans_by_name(exporter)
    fixture = load("groq_chat")

    llm = spans[f"chat {response.model}"]
    pipeline, retrieve = spans["rag_pipeline"], spans["retrieve"]
    assert llm.kind is SpanKind.CLIENT and pipeline.kind is SpanKind.INTERNAL

    # parent/child structure, carried by the library's own IDs
    assert pipeline.parent is None
    assert llm.parent.span_id == pipeline.context.span_id == retrieve.parent.span_id
    assert {s.context.trace_id for s in spans.values()} == {pipeline.context.trace_id}

    attributes = dict(llm.attributes or {})
    usage = fixture["usage"]
    assert attributes[semconv.OPERATION_NAME] == "chat"
    assert attributes[semconv.PROVIDER_NAME] == "groq"
    assert attributes[semconv.REQUEST_MODEL] == attributes[semconv.RESPONSE_MODEL] == response.model
    assert attributes[semconv.RESPONSE_ID] == "fixture-id"
    assert list(attributes[semconv.RESPONSE_FINISH_REASONS]) == ["stop"]
    assert attributes[semconv.USAGE_INPUT_TOKENS] == usage["prompt_tokens"]
    assert attributes[semconv.USAGE_OUTPUT_TOKENS] == usage["completion_tokens"]
    assert attributes[semconv.USAGE_REASONING] == 8
    assert attributes["gen_ai.request.temperature"] == 0.2
    assert attributes["gen_ai.request.max_tokens"] == 256
    assert list(attributes[semconv.REQUEST_STOP_SEQUENCES]) == ["END"]
    assert attributes[semconv.CONVERSATION_ID] == "s1" and attributes[semconv.USER_ID] == "u42"
    assert attributes["llm_logs.name"] == "groq.create"
    assert attributes["llm_logs.provider.queue_time"] == usage["queue_time"]
    assert dict(pipeline.attributes or {})["llm_logs.metadata.route"] == "/chat"
    assert dict(retrieve.attributes or {})[semconv.OPERATION_NAME] == "retrieval"
    assert semconv.PROVIDER_NAME not in (pipeline.attributes or {})

    assert llm.status.status_code is StatusCode.UNSET
    assert llm.resource.attributes["service.name"] == "test-app"
    assert llm.instrumentation_scope.name == "llm_logs"
    assert llm.end_time > llm.start_time


def test_the_span_ids_match_the_other_sinks() -> None:
    exporter, memory = InMemorySpanExporter(), ll.InMemorySink()
    ll.configure(sinks=[ll.OtelSink(exporter=exporter), memory], flush_interval=0.01)
    ll.trace(lambda: "ok", name="call")()
    assert ll.flush(5)
    (span,), (record,) = exporter.get_finished_spans(), memory.records
    assert f"{span.context.trace_id:032x}" == record.trace_id
    assert f"{span.context.span_id:016x}" == record.span_id
    assert span.end_time - span.start_time == int(record.duration_ms * 1_000_000)


def test_content_is_not_exported_unless_asked_for() -> None:
    exporter, memory = InMemorySpanExporter(), ll.InMemorySink()
    ll.configure(sinks=[ll.OtelSink(exporter=exporter), memory], flush_interval=0.01)
    run_pipeline()
    assert ll.flush(5)
    for span in exporter.get_finished_spans():
        serialised = json.dumps(dict(span.attributes or {}), default=str)
        assert SECRET_PROMPT not in serialised
        assert not {semconv.INPUT_MESSAGES, semconv.OUTPUT_MESSAGES, semconv.TOOL_DEFINITIONS} & (
            set(span.attributes or {})
        )
    # the local sink, configured with the default capture_content=True, still has it
    assert any(SECRET_PROMPT in record.to_json() for record in memory.records)


def test_content_follows_the_message_format_when_enabled() -> None:
    exporter = InMemorySpanExporter()
    configure(exporter, capture_content=True)
    response = run_pipeline()
    attributes = dict(spans_by_name(exporter)[f"chat {response.model}"].attributes or {})
    message = load("groq_chat")["choices"][0]["message"]
    assert json.loads(attributes[semconv.INPUT_MESSAGES]) == [
        {"role": "system", "parts": [{"type": "text", "content": "You are terse."}]},
        {"role": "user", "parts": [{"type": "text", "content": SECRET_PROMPT}]},
    ]
    assert json.loads(attributes[semconv.OUTPUT_MESSAGES]) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": message["reasoning"]},
                {"type": "text", "content": message["content"]},
            ],
            "finish_reason": "stop",
        }
    ]
    assert json.loads(attributes[semconv.TOOL_DEFINITIONS])[0]["function"]["name"] == "get_weather"


def test_message_conversion_for_tools_blocks_and_plain_prompts() -> None:
    assert semconv.input_messages({"prompt": "hi", "client": "<groq.Groq>"}) != []
    assert semconv.input_messages({"prompt": "hi"}) == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    ]
    assert semconv.system_instructions({"system": "Be brief.", "messages": []}) == [
        {"type": "text", "content": "Be brief."}
    ]
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "rainy"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "and this?"}, {"type": "image_url"}],
            },
        ]
    }
    assert semconv.input_messages(conversation) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "c1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                }
            ],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "c1", "response": "rainy"}],
        },
        {
            "role": "user",
            "parts": [{"type": "text", "content": "and this?"}, {"type": "image_url"}],
        },
    ]
    anthropic_output = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
        ],
    }
    assert semconv.output_messages(anthropic_output, ["tool_use"]) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "hmm"},
                {"type": "text", "content": "checking"},
                {
                    "type": "tool_call",
                    "id": "t1",
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                },
            ],
            "finish_reason": "tool_use",
        }
    ]


def test_errors_and_streams() -> None:
    exporter = InMemorySpanExporter()
    configure(exporter)

    @ll.trace(provider="groq", name="failing")
    def failing(model: str) -> None:
        raise TimeoutError("upstream timed out")

    with pytest.raises(TimeoutError):
        failing(model="m")
    chunks = [
        groq.types.chat.ChatCompletionChunk.model_validate(c) for c in load("groq_chat_stream")
    ]
    list(ll.trace(lambda: FakeStream(chunks), name="streaming")())

    assert ll.flush(5)
    spans = {s.attributes["llm_logs.name"]: s for s in exporter.get_finished_spans()}
    failed, streamed = spans["failing"], spans["streaming"]
    assert failed.name == "chat m"
    assert failed.status.status_code is StatusCode.ERROR
    assert failed.status.description == "upstream timed out"
    assert failed.attributes[semconv.ERROR_TYPE] == "TimeoutError"
    assert streamed.attributes[semconv.REQUEST_STREAM] is True
    assert 0 <= streamed.attributes[semconv.RESPONSE_TIME_TO_FIRST_CHUNK] < 5, "seconds, not ms"
    assert streamed.attributes["llm_logs.stream_outcome"] == "completed"


def test_llm_spans_join_the_hosts_active_trace() -> None:
    """With OpenTelemetry already tracing the request, LLM spans nest under it."""
    exporter, memory = InMemorySpanExporter(), ll.InMemorySink()
    ll.configure(sinks=[ll.OtelSink(exporter=exporter), memory], flush_interval=0.01)
    tracer = TracerProvider().get_tracer("host-app")
    ask = ll.trace(lambda: "ok", name="ask")

    with tracer.start_as_current_span("POST /chat") as request_span:
        with ll.span("pipeline"):
            ask()
        host = request_span.get_span_context()
    ask()  # outside any host span: a trace of its own
    assert ll.flush(5)

    inner, pipeline, outside = memory.records
    assert pipeline.trace_id == f"{host.trace_id:032x}"
    assert pipeline.parent_span_id == f"{host.span_id:016x}"
    assert inner.parent_span_id == pipeline.span_id and inner.trace_id == pipeline.trace_id
    assert outside.trace_id != pipeline.trace_id and outside.parent_span_id is None
    assert trace_api.get_current_span().get_span_context().is_valid is False


def test_a_failed_export_is_counted_and_does_not_stop_other_sinks() -> None:
    class Failing(SpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            return SpanExportResult.FAILURE

        def shutdown(self) -> None:
            return None

    memory = ll.InMemorySink()
    ll.configure(sinks=[ll.OtelSink(exporter=Failing()), memory], flush_interval=0.01)
    ll.trace(lambda: "ok", name="call")()
    assert ll.flush(5)
    assert len(memory.records) == 1
    assert ll.stats().failed_by_sink == {"OtelSink": 1}


def test_the_default_exporter_honours_the_standard_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.invalid:4318")
    sink = ll.OtelSink()
    exporter = sink._current_exporter()
    assert exporter._endpoint == "http://collector.invalid:4318/v1/traces"  # type: ignore[attr-defined]
    sink.close()


def test_the_default_exporter_really_sends_otlp_over_http() -> None:
    """No mocks: a local HTTP server stands in for the collector and decodes the protobuf."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    received: list[tuple[str, ExportTraceServiceRequest]] = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "gzip":
                import gzip

                body = gzip.decompress(body)
            request = ExportTraceServiceRequest()
            request.ParseFromString(body)
            received.append((self.path, request))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
        memory = ll.InMemorySink()
        ll.configure(
            sinks=[
                ll.OtelSink(exporter=OTLPSpanExporter(endpoint=endpoint), service_name="wire-test"),
                memory,
            ],
            flush_interval=0.01,
        )
        with ll.span("pipeline", session_id="s1"):
            ll.trace(lambda **kwargs: "ok", name="call", provider="groq")(
                model="m", temperature=0.5
            )
        assert ll.shutdown(15)
    finally:
        server.shutdown()
        server.server_close()

    assert ll.stats().failed == 0
    assert {path for path, _ in received} == {"/v1/traces"}
    resource_spans = [rs for _, request in received for rs in request.resource_spans]
    service = {a.key: a.value.string_value for a in resource_spans[0].resource.attributes}
    assert service["service.name"] == "wire-test"
    spans = {s.name: s for rs in resource_spans for ss in rs.scope_spans for s in ss.spans}
    assert set(spans) == {"pipeline", "chat m"}
    by_name = {r.name: r for r in memory.records}
    assert spans["chat m"].trace_id.hex() == by_name["call"].trace_id
    assert spans["chat m"].span_id.hex() == by_name["call"].span_id
    assert spans["chat m"].parent_span_id == spans["pipeline"].span_id
    attributes = {a.key: a.value for a in spans["chat m"].attributes}
    assert attributes["gen_ai.provider.name"].string_value == "groq"
    assert attributes["gen_ai.request.temperature"].double_value == 0.5
    assert attributes["gen_ai.conversation.id"].string_value == "s1"

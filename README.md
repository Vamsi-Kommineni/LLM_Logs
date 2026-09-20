# llm-logs

Record every LLM call your Python application makes — prompts, completions, the parameters that were really used, latency, tokens, errors and trace context — without slowing the application down and without writing credentials to disk.

Records go to local **JSONL** files, a **SQLite** database you can query from the command line, and/or **OpenTelemetry**, so the same data can flow into Langfuse, Phoenix, SigNoz, Jaeger or any other OTLP backend later.

> **Status: alpha (0.1.0a1).** The API may still change. Not on PyPI yet; install from this repository (see below). Version 1 of this project, a small 2024 script, lives unchanged in [`legacy/`](legacy/).

## What it is, and what it is not

It **is** a small library (one runtime dependency: `pydantic`) built around three promises:

1. **It never blocks or breaks your application.** Capturing a call costs about a tenth of a millisecond. Everything slow happens on a background thread behind a bounded queue. If the queue is full, records are dropped and counted; your request is never made to wait. After `configure()`, nothing in the library raises into your code.
2. **It records what actually ran.** The model, temperature and token counts come from the call's own arguments and the provider's response, not from a config file that might say something else.
3. **It does not leak credentials.** Only an allow-list of data types is ever serialised. An SDK client object passed to a traced function is recorded as `"<groq.Groq>"`, never walked. Values under keys like `api_key` or `authorization` are blanked at capture time, and strings shaped like API keys are scrubbed before anything is written.

It is **not** a dashboard, a hosted service, an eval framework, a prompt manager or a gateway, and it ships no price table. If you already run an observability backend and want every SDK call instrumented automatically, that ecosystem's own OpenTelemetry instrumentors are a better fit. This library is for when you want useful logs with zero infrastructure, wrapped around the functions *you* choose, with a standard way out (OTLP, JSONL, CSV) when you outgrow local files.

## Install

```bash
# until the first PyPI release
pip install "llm-logs @ git+https://github.com/Vamsi-Kommineni/LLM_Logs"

# optional extras
pip install "llm-logs[groq]"      # or [openai], [anthropic]
pip install "llm-logs[otel]"      # OpenTelemetry sink
```

Python 3.11 or newer. The extras only install the provider SDKs for your convenience: the adapters never import them, so `import llm_logs` stays light either way.

## Quickstart

```python
import llm_logs as ll
from groq import Groq

ll.configure(sinks=[ll.JsonlSink("logs/"), ll.SqliteSink("logs/llm.db")])
client = Groq()  # reads GROQ_API_KEY by itself


@ll.trace
def ask(client, prompt: str, *, model: str, temperature: float = 0.0):
    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )


ask(client, "What is 2 + 2?", model="<a current model id>")
```

```console
$ llm-logs tail --db logs/llm.db
14:02:11 ok       412 ms        92→18 tok  openai/gpt-oss-20b   ask   4
```

Each call becomes one record. Trimmed:

```json
{
  "trace_id": "5b8e…", "span_id": "a41c…", "parent_span_id": null,
  "kind": "llm", "operation": "chat", "name": "ask",
  "provider": "groq", "model": "openai/gpt-oss-20b", "response_model": "openai/gpt-oss-20b",
  "params": {"temperature": 0.0},
  "input": {"client": "<groq._client.Groq>", "prompt": "What is 2 + 2?"},
  "output": {"role": "assistant", "content": "4", "reasoning": "Simple arithmetic."},
  "input_tokens": 92, "output_tokens": 18, "reasoning_output_tokens": 8,
  "duration_ms": 412.3, "finish_reasons": ["stop"], "status": "ok",
  "provider_extras": {"queue_time": 0.29, "total_time": 0.035, "service_tier": "on_demand"}
}
```

The provider was detected from the type of the returned object. Note what happened to `client`.

A runnable version with the four questions from the 2024 project is in [`examples/groq_quickstart.py`](examples/groq_quickstart.py). Provider model IDs are retired often, so the examples read the model from `GROQ_MODEL`.

## Tracing

`@ll.trace` works on sync functions, `async` functions, generator functions and async generator functions, bare or with arguments, and keeps the function's signature for type checkers. It can also wrap an SDK method directly, which gives the adapter the real request:

```python
create = ll.trace(client.chat.completions.create, name="groq.chat")
create(
    model=MODEL, messages=messages, temperature=0.2
)  # model and params recorded from these kwargs
```

| Option | Meaning |
| --- | --- |
| `name` | Span name. Default: the function's qualified name. |
| `provider` | `"groq"`, `"openai"`, `"anthropic"`, or your own. Default: detected from the response type. Set it when you call another server through the OpenAI SDK's `base_url`. |
| `capture_args` / `ignore_args` | Choose which arguments are recorded as `input`. `self` and `cls` are always skipped. |
| `metadata`, `session_id`, `user_id` | Tags stored on the record. |
| `stream` | `None`: returned iterators are wrapped automatically. `True`: also treat a returned context manager as a stream. `False`: never wrap. |

If the function raises, the record gets `status="error"` with the exception type and message, and the exception propagates unchanged. That includes `asyncio.CancelledError`: a client that disconnects mid-request shows up as `error_type="CancelledError"`, which `llm-logs stats` lists separately from provider failures.

### Spans

Group calls into a trace. Spans nest, work with `with` and `async with`, and pass `session_id` and `user_id` down to everything inside:

```python
with ll.span("rag_pipeline", session_id="s1", user_id="u42", metadata={"route": "/chat"}):
    with ll.span("retrieve", operation="retrieval") as s:
        docs = retrieve(question)
        s.set(output=docs)
        s.metadata["documents"] = len(docs)
    answer = ask(client, question, model=MODEL)  # a child of rag_pipeline
```

Context is carried in `contextvars`, so concurrent `asyncio` tasks never see each other's spans. A plain thread starts with an empty context; to continue a trace there, run the function through `contextvars.copy_context().run(...)`.

### Streaming

A call that returns a stream is recorded when the stream *ends*, with the full text, token counts, `time_to_first_chunk_ms` and a `stream_outcome`:

| `stream_outcome` | When |
| --- | --- |
| `completed` | The stream was read to the end. |
| `closed_early` | `close()`, `aclose()`, or leaving the `with` block before the end. |
| `error` | The stream raised while being read. What arrived before is kept. |
| `abandoned` | The caller dropped the stream without closing it. The record is written when the object is garbage collected. |

Exactly one record is written in every case. The wrapper is a transparent proxy: SDK attributes and helpers such as `stream.response` keep working. For a traced *generator function*, garbage collection and `close()` are indistinguishable from the inside, so both are reported as `closed_early`.

## Privacy and security

Logs of LLM traffic contain whatever your users typed. Treat the files as sensitive data.

- **Credentials are kept out by construction**, as described above. This does not depend on any configuration.
- **Redactors** run on the writer thread before any sink sees a record. `ll.redact.api_keys` is on by default; add your own:

  ```python
  ll.configure(
      sinks=[...],
      redactors=[
          ll.redact.api_keys,
          ll.redact.emails,
          ll.redact.regex(r"\b\d{16}\b", "[CARD]"),
          ll.redact.keys("iban"),
      ],
  )
  ```

  If a redactor raises, that record is discarded and counted. It is never written unredacted.
- **`capture_content=False`** keeps parameters, timings and token counts but no prompts or completions. `LLM_LOGS_CAPTURE_CONTENT=0` does the same without a deploy.
- **The OpenTelemetry sink does not export content unless you ask**, even when local files keep it: `OtelSink(capture_content=True)`. Sending prompts to another system is a separate decision.
- Binary data, data URLs and long base64 strings are replaced by placeholders such as `<data-url mime=image/png len=48211>`.
- Log files are created with owner-only permissions on POSIX systems.

To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Sinks

```python
ll.JsonlSink("logs/", max_file_bytes=100 * 2**20, retention_days=30)
ll.SqliteSink("logs/llm.db")
ll.OtelSink(service_name="my-app")  # needs the [otel] extra
ll.InMemorySink()  # for tests
```

- **JSONL**: one file per UTC day and per process (`llm_logs-2026-09-20-p4121.0.jsonl`), so worker processes never interleave writes. Files roll over at `max_file_bytes`, and `retention_days` deletes the sink's own old files so the logger cannot fill the disk unnoticed.
- **SQLite**: one table with indexes on trace, time, model and status; payloads as JSON columns. WAL mode lets you query while the application writes. SQLite allows one writer at a time across all processes, so with many busy workers prefer JSONL or OTel. Do not put the file on a network filesystem.
- **OpenTelemetry**: spans follow the [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai) and keep the same trace and span IDs as the other sinks. If your application already has an active OpenTelemetry span (for example from FastAPI instrumentation), LLM spans join that trace as children. Configure the destination with the standard `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` variables. [`examples/otel_jaeger/`](examples/otel_jaeger/) has a one-container backend to try it with.
- A sink that fails does not affect the others. Writing your own takes three methods: `write_batch(records)`, `flush()`, `close()`.

## Command line

Everything works on the SQLite file and opens it read-only.

```console
$ llm-logs stats --db logs/llm.db --since 24h
LLM calls     6,400   (12,800 other spans)
Errors        402   (6.3%)
              CancelledError 279, ConnectionError 123
Latency       p50 40 ms   p95 53 ms
First chunk   p50 20 ms   p95 31 ms
...
$ llm-logs tail --db logs/llm.db --errors -n 50
$ llm-logs export --db logs/llm.db --format jsonl --model my-model -o dataset.jsonl
```

`export` is the hand-off to eval tooling: a JSONL or CSV dataset of real inputs and outputs. Filters: `--since 30m|24h|7d|<date>`, `--model`, `--errors`, `--trace`.

## In a web service

Configure inside the worker process (in FastAPI, the `lifespan`), and call `ll.shutdown()` when it stops:

```python
@asynccontextmanager
async def lifespan(app):
    ll.configure(sinks=[ll.JsonlSink("logs/")], redactors=[ll.redact.api_keys, ll.redact.emails])
    yield
    ll.shutdown()
```

[`examples/fastapi_groq_rag/`](examples/fastapi_groq_rag/) is a complete retrieval-plus-streaming-chat service with a load test. Under four uvicorn workers, 6,400 streaming requests (64 at a time) with client hang-ups and injected provider failures produced 19,200 records with nothing dropped, nothing duplicated and every call attached to its request's span, while server memory stayed flat. An e-mail address and a key-shaped string planted in every question reached neither the JSONL files nor the database.

The library is safe with forking servers such as gunicorn: a forked child gets its own queue, writer thread and files, and never touches what it inherited.

## Configuration

```python
ll.configure(
    sinks=[...],  # required
    redactors=None,  # None = [ll.redact.api_keys]; [] = none
    sample_rate=1.0,  # decided once per trace, so a trace is never partial
    capture_content=True,
    max_payload_chars=20_000,  # per input and per output; both ends of long text are kept
    queue_size=10_000,
    batch_size=100,
    flush_interval=1.0,  # seconds
    unknown_objects="type",  # "repr" records repr(obj) instead; it can leak what objects print
    pricing=None,  # optional Record -> cost, runs on the writer thread
    enabled=True,
)
```

`configure()` validates its arguments and raises `ll.ConfigurationError`: a mistake should fail at startup, not silently log nothing. Without `configure()`, tracing is a pass-through and one warning is logged; the library never writes files nobody asked for.

| Environment variable | Effect |
| --- | --- |
| `LLM_LOGS_DISABLED=1` | Turns tracing into a pass-through. |
| `LLM_LOGS_SAMPLE_RATE=0.1` | Overrides `sample_rate`. |
| `LLM_LOGS_CAPTURE_CONTENT=0` | Overrides `capture_content`. |

`ll.flush(timeout)` waits until everything recorded so far is written. `ll.shutdown(timeout)` flushes, closes the sinks and turns tracing off; it also runs at interpreter exit. `ll.stats()` returns counters (`enqueued`, `dropped`, `written`, `failed`, per-sink breakdowns, `redaction_errors`, `internal_errors`) — worth exporting to your metrics. The library's own problems are reported, rate-limited, through the standard `logging` module under the logger name `llm_logs`.

## Performance

Measured with [`benchmarks/overhead.py`](benchmarks/overhead.py) on a desktop-class x86-64 Linux machine with Python 3.13: 3,000 interleaved traced and untraced calls per row, difference of medians.

| Scenario | Added per call (median) |
| --- | --- |
| 200 to 2,000-character prompt | about 0.10 ms |
| 20,000-character prompt (at the size limit) | about 0.26 ms |
| 2,000,000-character prompt (truncated) | about 0.26 ms |
| `capture_content=False` | about 0.05 ms |
| Trace sampled out | about 0.005 ms |
| Sink that needs a full second per batch | about 0.11 ms (no change) |

The cost stops growing at the size limit because the serialiser walks with a character budget instead of converting everything and cutting afterwards. The worst single calls (p99) reached 5 to 6 ms: when the writer thread is busy, the calling thread can wait up to one interpreter switch interval for the GIL. Your numbers will differ; run the script. A test in CI fails if the median overhead behind a one-second sink reaches 1 ms.

## Providers

Adapters for **Groq**, **OpenAI** (Chat Completions, the Responses API, embeddings) and **Anthropic** fill in the model, response ID, finish reasons and token counts, including cached and reasoning tokens. `input_tokens` always means the full prompt size, cached tokens included, for every provider. Anything OpenAI-compatible (vLLM, Ollama, OpenRouter, ...) works through the OpenAI adapter; pass `provider="..."` to record the real name.

Without a matching adapter a call is still recorded, with the returned value as `output`. Register your own adapter with `ll.adapters.register(...)`; the protocol is in [`adapters/base.py`](src/llm_logs/adapters/base.py).

The Groq adapter is tested against responses recorded from the live API. The OpenAI and Anthropic fixtures were written from the public API references and are validated against those SDKs' own response types in the tests, but have not been checked against live traffic.

## Limitations

- SDKs retry internally, so one record can cover several HTTP attempts and its duration includes them.
- A traced *sync* function that returns an awaitable is recorded when it returns, not when the awaitable resolves. Decorate the `async` function instead.
- Sampling is decided when a trace starts. There is no "keep all errors" mode, because that would produce partial traces.
- The OpenTelemetry GenAI conventions are still at Development status and their attribute names change. They are isolated in one module, [`sinks/_otel_semconv.py`](src/llm_logs/sinks/_otel_semconv.py), which records the date it was last checked against the spec.
- The Jaeger example has not been run end to end yet. The OTLP export itself is covered by a test that sends real OTLP over HTTP to a local server and decodes the protobuf.

## Development

```bash
uv sync
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest
```

The design, the reasons behind it and the bugs found along the way are in [docs/design.md](docs/design.md). To refresh the Groq fixtures: `GROQ_API_KEY=... uv run python scripts/record_fixtures.py`. Live provider tests are opt-in: `GROQ_API_KEY=... GROQ_MODEL=... uv run pytest -m live`.

## License

[Apache License 2.0](LICENSE)

## Citation

```
@misc{LLM_Logs,
  author = {Vamsi Kommineni},
  month = {04},
  title = {{LLM_Logs}},
  url = {https://github.com/Vamsi-Kommineni/LLM_Logs},
  year = {2024}
}
```

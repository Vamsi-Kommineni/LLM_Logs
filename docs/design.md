# Design

This document explains why `llm-logs` is built the way it is. The README says what the library does; this says why, what was tried, and what went wrong along the way. It replaces the implementation plan the first version was built from.

## The problem

Version 1 (2024, kept in `legacy/`) was a decorator that appended a JSON line to a file. It showed the idea and had every problem a first attempt has: it read the model name and temperature from a config file rather than from the call, lost keyword arguments, logged nothing when the call failed, wrote to disk on the request path, pulled in `torch` on import, and sat next to a tracked `.env`.

Version 2 keeps the idea and is built around three promises:

1. It never blocks or breaks the host application.
2. It records what actually ran.
3. It does not write credentials to disk.

Everything below follows from taking those literally.

## Architecture

```
caller code
   │  @trace / with span(...)
   ▼
capture  ── sampled out? pass straight through
   │        otherwise: JSON-safe snapshot, secrets scrubbed, truncated, timed
   │  put_nowait
   ▼
bounded queue ── full? drop the record and count it; never block
   │
   ▼
writer thread ── pricing → redact → batch → fan out
   ├── JsonlSink
   ├── SqliteSink
   └── OtelSink
```

| Module | Job |
| --- | --- |
| `capture.py` | `trace`, `span`, the four wrapper kinds, building the `Record` |
| `serialize.py` | `to_jsonable()`: bounded, allow-listed, credential-free conversion |
| `streams.py` | Transparent proxies that finalise a record when a stream ends |
| `context.py` | Trace context in `contextvars`, IDs, sampling, the OpenTelemetry bridge |
| `writer.py` | Queue, thread, batching, flush, shutdown |
| `config.py` | `configure()`, process-wide state, fork and exit hooks |
| `redact.py` | Redactors that run on the writer thread |
| `adapters/` | Provider knowledge, duck-typed, no SDK imports |
| `sinks/` | JSONL, SQLite, OpenTelemetry |
| `cli.py` | `llm-logs tail / stats / export`, standard library only |

## Decisions

### A queue and a thread, not a direct write

A log write on the request path makes your latency depend on your disk, your database lock, or your collector's network. The caller does a `put_nowait` and nothing else. Redaction, JSON encoding, SQLite transactions and OTLP export all happen on one daemon thread.

One thread is enough: the work is I/O bound, sinks then need no locking, and order is preserved.

### Drop instead of block

When the queue is full there are two options: make the caller wait, or lose a record. A logging library that makes a request wait has failed at the first promise, so records are dropped and counted (`ll.stats().dropped`). This is the same trade OpenTelemetry's batch processor makes. Drops are visible, so you can size the queue or fix the sink.

### "Never raises", with one exception

After `configure()` returns, every entry point catches its own failures, counts them (`internal_errors`) and logs them, rate-limited, under the `llm_logs` logger. The caller's function still runs and still returns or raises exactly what it would have.

`configure()` itself does raise. A logging setup that is wrong should fail at startup, not quietly record nothing in production. The same applies to `trace("name")` used by mistake for `trace(name="name")`, which fails when the module is imported. A typo in an *environment variable* is different: that is an operator's mistake at deploy time, so it is logged and ignored rather than allowed to take the application down.

User exceptions propagate unchanged, including `BaseException` subclasses. `asyncio.CancelledError` matters most: in an async web service, a client that disconnects cancels the task, and that call must still leave a record.

### An allow-list serialiser

This was the one real flaw in the first design, which said to fall back to walking an object's `__dict__`. SDK client objects keep the API key in an attribute, and a traced function very often receives the client (`def ask(client, prompt)`, or `self.client` on a method). A `__dict__` walk would have written the key into the log file.

So `to_jsonable()` walks only types it knows: primitives, mappings, sequences, sets, enums, dates, UUIDs, decimals, paths, dataclasses and Pydantic models. Everything else becomes a type name, `"<groq._client.Groq>"`. There is no `__dict__` walk and no `repr` call unless you opt in with `unknown_objects="repr"`. On top of that:

- Values under credential-like keys (`api_key`, `authorization`, `token`, `password`, ...) are blanked at capture time. The rule has to leave `max_tokens`, `input_tokens` and `eos_token` alone, which is why "token" gets special handling and its own tests.
- Dataclass and Pydantic fields declared with `repr=False` are skipped; that flag usually marks a secret.
- `self` and `cls` are never recorded.
- Strings shaped like API keys are scrubbed by a default redactor, which also covers error messages: an authentication error that echoes the key back is a real case.

Tests plant a key in a client object, in headers, in a model field and in an exception message, and assert it is in none of the bytes written.

### Bounded by budget, not by trimming afterwards

Converting a 2 MB prompt to JSON and then cutting it to 20 KB would make the cost depend on the payload. The walk instead carries a character budget, charges the exact JSON cost of everything it emits, and stops descending when the budget is spent. A property test (hypothesis) asserts that the compact JSON form never exceeds the limit, for arbitrary nested data including strings full of characters that JSON has to escape. The benchmark shows the effect: a 2,000,000-character prompt costs the same as a 20,000-character one.

Two refinements came from thinking about what is worth keeping in a chat log:

- **Both ends are kept.** A long string keeps its head and tail around a marker; a long list keeps its first and last items. The system prompt is at the head and the turn that produced the answer is at the tail.
- **Siblings share a tight budget fairly.** Without this, a 50,000-character system prompt eats the whole budget and the user's actual question is lost. Small items get what they need and large ones split the rest.

### Streams are finalised exactly once

A streamed call is not finished when the function returns. The returned stream is wrapped in a proxy that passes chunks through, accumulates what the record needs in bounded memory, and finalises on the first of: exhaustion, an error while iterating, `close()`/`aclose()`/leaving a `with` block, or garbage collection (`weakref.finalize`). Without the last one, a caller who breaks out of a loop and forgets the stream would leave no record at all.

The proxy delegates unknown attributes to the wrapped object, because SDK streams have helpers people use. Some helpers (Anthropic's `text_stream`) read the underlying stream directly and bypass the proxy; for those, the adapter reads the SDK's own final snapshot when the block ends.

Traced *generator functions* are wrapped differently: the wrapper is itself a generator function, because frameworks check for that (`inspect.isgeneratorfunction`) to decide whether to stream. The cost is that `close()` and garbage collection look the same from inside a generator, so both are reported as `closed_early`.

An exception raised by the *caller's* loop body while reading a stream is not an LLM error. The record says `closed_early` with `status="ok"`.

### Context in `contextvars`, sampling per trace

`contextvars` follow `asyncio` tasks, so concurrent requests cannot see each other's spans. A generator shares its caller's context, so a traced generator activates its span only while its own body runs and never leaks it to the consumer.

Sampling is decided once, when a trace starts, from the low 64 bits of the trace ID. Every span in the trace inherits the decision, so a trace is complete or absent, and any process that sees the same trace ID decides the same way without coordination. A sampled-out call costs about 5 µs.

### Process safety

- The writer thread starts on first use, not at import or `configure()`.
- After a fork, the child replaces the queue, the locks, the counters and the writer, and never touches what it inherited: an inherited lock may be held by a thread that does not exist in the child.
- Sinks open their resources lazily on the writer thread and reopen when the PID changes. The JSONL sink writes through an unbuffered file descriptor, one `write` per batch, so there is no userspace buffer a forked child could flush a second time. The SQLite sink parks an inherited connection and never uses or closes it.
- JSONL file names contain the PID, so workers never share a file.
- Python 3.12 warns when a multi-threaded process forks, and since 3.14 `multiprocessing` on Linux defaults to `forkserver`. Tests therefore set the start method explicitly and cover both `fork` and `spawn`, plus a raw `os.fork()` with the writer running, which is what gunicorn does.
- An `atexit` hook flushes with a timeout. The thread is a daemon, so a stuck sink cannot keep the process alive. Since 3.12 a thread cannot be started during interpreter shutdown; a late record is counted as dropped.

### OpenTelemetry instead of a dashboard

Building a UI would mean maintaining a worse version of what Langfuse, Phoenix and others already do. Speaking OTLP with the GenAI semantic conventions gets all of them at once.

Two things were not obvious:

- **The tracer API cannot be used.** It generates its own span IDs, and the records already carry the IDs that tie a trace together across JSONL, SQLite and OTel. The sink builds finished `ReadableSpan` objects with explicit contexts and timestamps and hands them to a `SpanExporter`.
- **The bridge.** If the host application already has an active OpenTelemetry span, a new root span adopts its trace ID and uses it as parent, so LLM spans appear inside the request trace instead of floating beside it. The core looks in `sys.modules` for OpenTelemetry and never imports it.

Content is not exported unless `OtelSink(capture_content=True)`, even when local sinks keep it. The conventions mark message content as opt-in, and sending prompts to another system is a different privacy decision from writing a local file.

The conventions are at Development status and have moved to their own repository. All attribute names live in `sinks/_otel_semconv.py`, written as literals: the released Python semconv package lagged the spec when this was written (`cache_creation` where the spec says `cache_write`). The module records the date it was last checked.

### Adapters never import an SDK

Adapters read attributes and keys, and recognise a response by the module name of its type. That keeps `import llm_logs` light, makes the adapters work on plain dicts from raw HTTP calls, and means one extractor serves Groq, OpenAI and every OpenAI-compatible server.

`input_tokens` always means the whole prompt including cached tokens, which is also what the OTel conventions ask for. Anthropic reports the uncached part separately, so its adapter adds the parts up.

Model IDs appear nowhere in `src/` or `tests/` outside recorded fixtures. While this was being written, Groq retired its Llama 3 models, and one of its own documentation pages still listed them as current weeks later. Examples read the model from an environment variable.

### The Record schema

One `Record` per span, Pydantic, `extra="forbid"` on construction so that typos in this library fail in tests, with `schema_version` and `lib_version` for later migrations. The CLI reads rows as plain dicts, not through the strict model, so it can read a database written by a newer version (there is a test for that).

Fields beyond the obvious, and why they exist:

| Field | Reason |
| --- | --- |
| `kind` (`llm` / `span`) | `span()` produces records that are not model calls |
| `model` and `response_model` | What was asked for and what the provider says it used can differ |
| `time_to_first_chunk_ms` | The latency users feel in a streaming UI |
| `cache_read_`, `cache_write_`, `reasoning_` tokens | Needed for correct cost, and for reasoning models |
| `stream_outcome` | Distinguishes a finished answer from a closed tab |
| `provider_extras` | Provider-specific fields (Groq's queue time), kept apart from user `metadata` |

`duration_ms` comes from `time.perf_counter`, not from subtracting wall-clock times.

## What testing turned up

- **SQLite, two processes starting at once.** The fork test failed about half the time: JSONL complete, SQLite missing exactly one batch. The library's own diagnostics (`failed_by_sink` and the rate-limited log) pointed at `PRAGMA journal_mode=WAL` raising "database is locked". SQLite does not apply the busy timeout to that statement, and the retry loop covered only the insert. Connection setup now sits inside the retried section, asks for the journal mode before trying to change it, and creates the schema in one transaction. 120 consecutive stress runs passed afterwards, and a deterministic test locks the database while the sink connects. The multi-process tests now also assert `failed == 0` in every process, because complete JSONL files had been hiding the problem.
- **The size estimator starved small messages.** With a tight budget, short chat messages came out as `"<dict len=2>"`: their share was computed from their content alone, which is less than the headroom a container needs to be opened.
- **The hot path was slower than assumed.** About 0.10 ms per call, not the "tens of microseconds" first written down. Profiling showed the serialiser at about two thirds of it. Caching the sensitive-key check and inlining leaf estimates helped a little; further work on a security-critical, property-tested walker was not worth the risk for a cost that is about 0.1% of a fast LLM call. The documentation was corrected instead.
- **A fixture the SDK rejected.** The hand-written OpenAI Responses fixture failed validation against the installed SDK, which requires `cache_write_tokens`. That field now feeds `cache_write_input_tokens`. This is what validating hand-written fixtures against the real SDK types is for.
- **Groq's SDK has no `stream_options` parameter**, although the API accepts it; the fixture recorder passes it through `extra_body`. With or without it, Groq sends usage on the last content chunk under both `usage` and `x_groq.usage`, and with it an extra chunk with empty `choices`. The accumulator handles all three.

## Verification

- 172 tests that need no network. They include the hypothesis properties, the overhead guard (median overhead behind a one-second sink under 1 ms), the credential-leak tests, four processes under both start methods, a fork with the writer running, the seven streaming cases, a real OTLP/HTTP export decoded from protobuf, and a wheel installed into a clean environment to check that only `pydantic` comes with it.
- Four opt-in live tests against Groq through the real SDK: sync, streaming through the SDK's own stream class, async streaming, and an authentication error recorded without the key.
- A soak run of the FastAPI example under four workers: 6,400 streaming requests with client hang-ups and injected failures, 19,200 records, nothing dropped or duplicated, flat memory, and planted PII absent from every file including the SQLite WAL.
- Not verified: the Jaeger example end to end, and the OpenAI and Anthropic adapters against live traffic (their fixtures are hand-written and validated against the SDK types).

## Deviations from the original plan

| Plan | What was done | Why |
| --- | --- | --- |
| `examples/otel_langfuse/` | `examples/otel_jaeger/` | Jaeger is one container with native OTLP. Langfuse is five with its own secrets. Any OTLP backend works through the standard environment variables |
| Hugging Face and LangChain adapters | Not built | LangChain integration is a callback handler, a different mechanism. The local Hugging Face pipeline was the outdated path |
| "Run for a week" dogfooding | A concurrent soak run | There is no external application, and a soak run tests more in less time |
| Python 3.10 | 3.11 and newer | 3.10 reaches end of life on 2026-10-31, before a first stable release |
| `pip install -e ".[dev]"` | `uv sync` with a PEP 735 dependency group | Development tools stay out of the published metadata |
| Rewrite git history to purge `.env` | Not done | The committed file only ever held a placeholder, and a scan of the full history found no key |

## Known limitations

See the README. The main ones: SDK-internal retries are invisible, there is no tail-based sampling, and the OTel attribute names will need updating as the conventions settle.

# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0a1]

A rewrite from scratch as an installable library. The 2024 script is kept unchanged in `legacy/`.

### Added

- `trace` decorator for sync, async, generator and async generator functions, usable bare, with arguments, or as a plain wrapper around an SDK method. Keeps the wrapped function's signature for type checkers (`py.typed` included).
- `span` context manager (sync and async) with nesting, `session_id` / `user_id` inheritance, mutable `metadata` and `set(input=..., output=...)`.
- `Record` schema (version 1) with requested and reported model, parameters as passed, token counts including cached and reasoning tokens, time to first chunk, finish reasons, stream outcome, provider extras and error details.
- Background writer: bounded queue, one daemon thread, batching by size and time, drop-and-count when full, `flush()`, idempotent `shutdown()`, flush at interpreter exit, and a fresh writer in forked children.
- Allow-list serialiser with a character budget: unknown objects become type names, credential-like keys are blanked, media is replaced by placeholders, long text and lists keep both ends, siblings share a tight budget fairly.
- Sinks: `JsonlSink` (per-day and per-process files, size cap, retention, owner-only permissions), `SqliteSink` (WAL, indexes, busy handling), `OtelSink` (GenAI semantic conventions, same IDs as the other sinks, joins an active host trace, content export off by default), `InMemorySink`.
- Redactors on the writer thread: `redact.api_keys` (on by default), `redact.emails`, `redact.regex(...)`, `redact.keys(...)`. A redactor that raises discards the record.
- Stream handling for iterators, async iterators and context-manager streams through transparent proxies; one record per stream with outcome `completed`, `closed_early`, `error` or `abandoned`.
- Adapters for Groq, OpenAI (Chat Completions, Responses, embeddings) and Anthropic, with provider detection from the response type and no SDK imports. `adapters.register()` for your own.
- Deterministic per-trace sampling, `capture_content` switch, and the environment overrides `LLM_LOGS_DISABLED`, `LLM_LOGS_SAMPLE_RATE`, `LLM_LOGS_CAPTURE_CONTENT`.
- `stats()` counters and rate-limited diagnostics under the `llm_logs` logger.
- `llm-logs` command line: `tail`, `stats` (error rate and types, p50/p95 latency and time to first chunk, tokens by model) and `export` to JSONL or CSV.
- Examples: Groq quickstart, a FastAPI retrieval-and-streaming service with a load test, and an OpenTelemetry setup with Jaeger. A benchmark script and a fixture recorder.

### Changed

- The project is now a package (`src/llm_logs`) with `pydantic` as its only runtime dependency. Importing it no longer loads `torch`, `transformers` or `langchain`.
- Model and parameters are taken from the call and the provider's response instead of `config.ini`.
- Requires Python 3.11 or newer.

### Removed

- Model loading (`load_llm`, `LLM_loader.py`) is no longer part of the library. The old code remains in `legacy/` for reference.

### Security

- `.env` is no longer tracked, environment files are ignored, and `.env.example` documents the variables. Secret scanning runs in pre-commit and CI.
- Credentials held by SDK client objects, headers, model fields or error messages are kept out of every sink. See "Privacy and security" in the README.

# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting instead: on this repository, go to **Security → Report a vulnerability**. You should get a first answer within a week.

Useful to include: the version (`python -c "import llm_logs; print(llm_logs.__version__)"`), a minimal reproduction, and what ended up where it should not have.

Things that count as vulnerabilities here:

- A credential (API key, token, password, authorization header) reaching any sink through this library.
- A way for logged data to escape redaction that the documentation says is applied.
- The library raising into, blocking, or crashing the host application after `configure()`.
- Path handling in a sink that lets a log file be written outside the configured directory.

## What this library stores

`llm-logs` records the inputs and outputs of LLM calls. With the default `capture_content=True`, that means prompts and completions: **whatever your users typed and whatever the model answered**. Treat the log directory and the SQLite file as sensitive data:

- Keep them out of version control (the project's own `.gitignore` excludes `logs/`, `*.jsonl` and `*.db`).
- Files are created with owner-only permissions on POSIX systems. Keep it that way in containers and on shared volumes.
- Set `retention_days` on `JsonlSink`, and decide how long the SQLite file is kept.
- Add redactors for the personal data your application handles; only `redact.api_keys` is on by default.
- Use `capture_content=False` (or `LLM_LOGS_CAPTURE_CONTENT=0`) where you must not store content at all. Timings, parameters and token counts are still recorded.
- `OtelSink` does not export content unless you pass `capture_content=True`. Before turning it on, check where your OTLP endpoint sends data and who can read it.

## What the library does to protect credentials

Only an allow-list of data types is serialised; any other object is recorded as its type name. Values under credential-like key names are blanked at capture time. Strings shaped like common API keys are scrubbed before any sink sees a record, including in error messages. These measures are covered by tests that plant a key and search the written bytes for it.

They are safeguards, not a guarantee. A secret under an unusual key name, inside free text, or returned by `repr()` when `unknown_objects="repr"` is enabled, can still be recorded. The safest secret is one that never enters a traced function's arguments.

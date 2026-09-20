# Project rules

- This is a logging library. After `configure()`, it must never raise into caller code and never block the caller.
- It must never record credentials. `to_jsonable()` walks an allow-list of types only. Do not add `__dict__` or `repr` fallbacks.
- Core package depends on pydantic only. Adapters are duck-typed and import no SDK. `sinks/otel.py` is the only module that imports OpenTelemetry, and it is loaded lazily.
- Python >= 3.11, full type hints, mypy strict on `src/`.
- Every behaviour change comes with a test. Before finishing run:
  `uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest`
- The overhead test (`tests/test_writer.py`) and the credential-leak tests (`tests/test_serialize.py`, `tests/test_capture.py`, `tests/test_redact.py`) guard the core promises. Never weaken or skip them to make a change pass.
- `docs/design.md` records the design decisions and the reasons. Keep to them, or change the document in the same commit and say why.
- Non-goals: no dashboard, no hosted backend, no evals, no prompt management, no gateway, no model loading, no bundled price table.
- Public API changes must be reflected in `README.md` and `CHANGELOG.md`.
- `legacy/` is the 2024 version, kept for the record. Do not modify or import it.

# Public repo hygiene

- Never read out, print, log or commit the contents of `.env`. Check it by key name only. When a script needs the key, load it into a subprocess environment.
- No absolute local paths, usernames, hostnames or email addresses in code, docs, fixtures, benchmarks or commit messages. Describe benchmark hardware generically.
- No model IDs in `src/` or `tests/` outside recorded fixtures. Examples and live tests read the model from an environment variable.
- Fixtures are scrubbed response bodies (`scripts/record_fixtures.py`). Never commit raw HTTP captures or headers.
- OpenTelemetry GenAI attribute names appear only in `sinks/_otel_semconv.py`. Check the current spec before editing it and update `SPEC_CHECKED`.
- GitHub Actions are pinned by commit SHA. Do not replace a SHA with a tag.

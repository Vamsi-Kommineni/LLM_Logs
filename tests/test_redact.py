from __future__ import annotations

from pathlib import Path

import pytest

import llm_logs as ll
from llm_logs import redact

EMAIL = "jane.doe@example.com"
KEYS = [
    "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
    "gsk_abcdefghijklmnopqrstuvwxyz0123456789",
    "hf_abcdefghijklmnopqrstuvwxyz012345",
    "AKIAIOSFODNN7EXAMPLE",  # the example key from AWS's own documentation
    "Bearer abcdefghijklmnopqrstuvwxyz.0123456789",
]


@ll.trace
def ask(prompt: str, account: dict[str, str]) -> str:
    return f"I wrote to {EMAIL} about card 4111111111111111"


def raw_bytes(directory: Path) -> bytes:
    return b"".join(path.read_bytes() for path in sorted(directory.glob("*.jsonl")))


def test_redaction_happens_before_bytes_reach_the_disk(tmp_path: Path) -> None:
    ll.configure(
        sinks=[ll.JsonlSink(tmp_path), ll.SqliteSink(tmp_path / "llm.db")],
        redactors=[
            redact.emails,
            redact.api_keys,
            redact.regex(r"\b\d{16}\b", "[CARD]"),
            redact.keys("iban"),
        ],
        flush_interval=0.01,
    )
    ask(f"mail {EMAIL}, keys: {' '.join(KEYS)}", {"iban": "DE89370400440532013000", "plan": "pro"})
    assert ll.flush(5)

    on_disk = raw_bytes(tmp_path) + (tmp_path / "llm.db").read_bytes()
    wal = tmp_path / "llm.db-wal"
    if wal.exists():
        on_disk += wal.read_bytes()
    for secret in [EMAIL, "4111111111111111", "DE89370400440532013000", *KEYS]:
        assert secret.encode() not in on_disk
    assert b"[EMAIL]" in on_disk and b"[CARD]" in on_disk and b'"plan":"pro"' in on_disk


def test_api_key_shapes_are_scrubbed_by_default(tmp_path: Path) -> None:
    ll.configure(sinks=[ll.JsonlSink(tmp_path)], flush_interval=0.01)

    @ll.trace
    def call(prompt: str) -> str:
        raise PermissionError(f"Incorrect API key provided: {KEYS[1]}")

    with pytest.raises(PermissionError):
        call(f"my key is {KEYS[0]}")
    assert ll.flush(5)
    on_disk = raw_bytes(tmp_path)
    assert KEYS[0].encode() not in on_disk and KEYS[1].encode() not in on_disk
    assert b"Incorrect API key provided: [REDACTED]" in on_disk


def test_a_redactor_that_raises_never_lets_the_record_through(tmp_path: Path) -> None:
    def broken(record: ll.Record) -> ll.Record:
        if "poison" in str(record.input):
            raise RuntimeError("redactor bug")
        return record

    ll.configure(sinks=[ll.JsonlSink(tmp_path)], redactors=[broken], flush_interval=0.01)
    ask("poison pill with secret-data-123", {})
    ask("harmless", {})
    assert ll.flush(5)
    on_disk = raw_bytes(tmp_path)
    assert b"secret-data-123" not in on_disk and b"harmless" in on_disk
    stats = ll.stats()
    assert (stats.redaction_errors, stats.written) == (1, 1)


def test_a_redactor_must_return_a_record(tmp_path: Path) -> None:
    ll.configure(
        sinks=[ll.JsonlSink(tmp_path)],
        redactors=[lambda record: None],  # type: ignore[list-item,return-value]
        flush_interval=0.01,
    )
    ask("anything", {})
    assert ll.flush(5)
    assert raw_bytes(tmp_path) == b"" and ll.stats().redaction_errors == 1


def test_redactors_do_not_mutate_the_original_record() -> None:
    record = ll.Record(
        lib_version="t",
        trace_id="0" * 31 + "1",
        span_id="0" * 15 + "1",
        kind="llm",
        name="n",
        start_time="2026-01-01T00:00:00Z",  # type: ignore[arg-type]
        end_time="2026-01-01T00:00:01Z",  # type: ignore[arg-type]
        duration_ms=1.0,
        input={"messages": [{"role": "user", "content": f"write to {EMAIL}"}]},
    )
    cleaned = redact.emails(record)
    assert cleaned.input == {"messages": [{"role": "user", "content": "write to [EMAIL]"}]}
    assert EMAIL in record.input["messages"][0]["content"]

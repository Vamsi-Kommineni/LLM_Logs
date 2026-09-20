from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import stat
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import llm_logs as ll
from llm_logs.sinks.sqlite import COLUMNS
from tests.conftest import query


def make_record(number: int = 0, **fields: Any) -> ll.Record:
    base: dict[str, Any] = {
        "lib_version": "test",
        "trace_id": f"{number + 1:032x}",
        "span_id": f"{number + 1:016x}",
        "kind": "llm",
        "name": "ask",
        "start_time": datetime(2026, 1, 1, tzinfo=UTC),
        "end_time": datetime(2026, 1, 1, tzinfo=UTC) + timedelta(milliseconds=5),
        "duration_ms": 5.0,
        "model": "model-a",
        "input": {"prompt": f"question {number} – ünïcödé"},
        "output": "answer",
        "input_tokens": 10,
        "output_tokens": 3,
        "finish_reasons": ["stop"],
        "params": {"temperature": 0.0},
        "metadata": {"n": number},
    }
    return ll.Record(**{**base, **fields})


# -------------------------------------------------------------------- jsonl


def test_jsonl_lines_round_trip_into_records(tmp_path: Path) -> None:
    sink = ll.JsonlSink(tmp_path)
    records = [make_record(n) for n in range(3)]
    sink.write_batch(records)
    sink.flush()
    sink.close()
    (path,) = tmp_path.glob("*.jsonl")
    assert f"-p{os.getpid()}." in path.name and path.name.startswith("llm_logs-")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [ll.Record.model_validate_json(line) for line in lines] == records


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_log_files_are_private_to_the_owner(tmp_path: Path) -> None:
    jsonl, sqlite = ll.JsonlSink(tmp_path), ll.SqliteSink(tmp_path / "llm.db")
    for sink in (jsonl, sqlite):
        sink.write_batch([make_record()])
        sink.close()
    for path in [*tmp_path.glob("*.jsonl"), tmp_path / "llm.db"]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name


def test_jsonl_rolls_over_at_the_size_cap_and_never_splits_a_batch(tmp_path: Path) -> None:
    sink = ll.JsonlSink(tmp_path, max_file_bytes=2_000)
    for number in range(30):
        sink.write_batch([make_record(number)])
    sink.close()
    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) > 3
    assert all(path.stat().st_size <= 2_000 for path in files)
    lines = [line for path in files for line in path.read_text(encoding="utf-8").splitlines()]
    assert sorted(json.loads(line)["metadata"]["n"] for line in lines) == list(range(30))


def test_jsonl_continues_the_sequence_after_a_restart(tmp_path: Path) -> None:
    for _ in range(2):
        sink = ll.JsonlSink(tmp_path, max_file_bytes=600)
        sink.write_batch([make_record()])
        sink.close()
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_jsonl_retention_only_touches_its_own_old_files(tmp_path: Path) -> None:
    old = tmp_path / "llm_logs-2020-01-01-p1.0.jsonl"
    recent = tmp_path / f"llm_logs-{datetime.now(UTC):%Y-%m-%d}-p1.0.jsonl"
    foreign = tmp_path / "other-2020-01-01.jsonl"
    for path in (old, recent, foreign):
        path.write_text("{}\n")
    sink = ll.JsonlSink(tmp_path, retention_days=7)
    sink.write_batch([make_record()])
    sink.close()
    assert not old.exists() and recent.exists() and foreign.exists()


def test_jsonl_without_pid_in_the_name(tmp_path: Path) -> None:
    sink = ll.JsonlSink(tmp_path, include_pid=False, prefix="app")
    sink.write_batch([make_record()])
    sink.close()
    (path,) = tmp_path.glob("*.jsonl")
    assert path.name == f"app-{datetime.now(UTC):%Y-%m-%d}.0.jsonl"


def test_jsonl_rejects_path_tricks_in_the_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ll.JsonlSink(tmp_path, prefix="../escape")


# ------------------------------------------------------------------- sqlite


def test_sqlite_schema_indexes_and_wal(tmp_path: Path) -> None:
    sink = ll.SqliteSink(tmp_path / "nested" / "llm.db")
    sink.write_batch([make_record(n) for n in range(5)])
    path = tmp_path / "nested" / "llm.db"
    assert [row[1] for row in query(path, "PRAGMA table_info(records)")] == ["id", *COLUMNS]
    indexes = {row[1] for row in query(path, "PRAGMA index_list(records)")}
    assert {f"idx_records_{name}" for name in ("trace_id", "start_time", "model", "status")} <= (
        indexes
    )
    assert query(path, "PRAGMA journal_mode") == [("wal",)]
    assert query(path, "SELECT COUNT(*) FROM records") == [(5,)]
    sink.close()


def test_sqlite_stores_payloads_as_json_and_scalars_as_columns(tmp_path: Path) -> None:
    sink = ll.SqliteSink(tmp_path / "llm.db")
    sink.write_batch([make_record(status="error", error_type="Timeout", streamed=True)])
    with contextlib.closing(sqlite3.connect(tmp_path / "llm.db")) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM records").fetchone()
    assert json.loads(row["input"]) == {"prompt": "question 0 – ünïcödé"}
    assert json.loads(row["finish_reasons"]) == ["stop"]
    assert (row["status"], row["error_type"], row["streamed"], row["truncated"]) == (
        "error",
        "Timeout",
        1,
        0,
    )
    assert row["provider_extras"] is None, "empty containers are stored as NULL"
    assert row["start_time"].startswith("2026-01-01T00:00:00")
    sink.close()


def test_sqlite_ignores_a_record_it_already_has(tmp_path: Path) -> None:
    sink = ll.SqliteSink(tmp_path / "llm.db")
    sink.write_batch([make_record(1), make_record(1), make_record(2)])
    sink.write_batch([make_record(2)])
    assert query(tmp_path / "llm.db", "SELECT COUNT(*) FROM records") == [(2,)]
    sink.close()


def test_sqlite_waits_for_a_competing_writer(tmp_path: Path) -> None:
    path = tmp_path / "llm.db"
    sink = ll.SqliteSink(path, busy_timeout=0.05)
    sink.write_batch([make_record(0)])

    other = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    other.execute("BEGIN IMMEDIATE")
    timer = threading.Timer(0.3, lambda: other.execute("COMMIT"))
    timer.start()
    sink.write_batch([make_record(1)])  # retried until the other writer commits
    timer.join()
    assert other.execute("SELECT COUNT(*) FROM records").fetchone() == (2,)
    sink.close()
    other.close()


def test_sqlite_survives_a_locked_database_while_connecting(tmp_path: Path) -> None:
    """Two workers opening the file at once: the WAL switch ignores the busy timeout."""
    path = tmp_path / "llm.db"
    other = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    other.execute("CREATE TABLE unrelated (x)")
    other.execute("BEGIN EXCLUSIVE")
    timer = threading.Timer(0.3, lambda: other.execute("COMMIT"))
    timer.start()
    sink = ll.SqliteSink(path, busy_timeout=0.01)
    sink.write_batch([make_record(0)])  # first connection happens while the file is locked
    timer.join()
    assert other.execute("SELECT COUNT(*) FROM records").fetchone() == (1,)
    assert other.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    sink.close()
    other.close()


def test_both_sinks_work_through_the_public_api(tmp_path: Path) -> None:
    ll.configure(
        sinks=[ll.JsonlSink(tmp_path / "logs"), ll.SqliteSink(tmp_path / "logs" / "llm.db")],
        flush_interval=0.01,
    )

    @ll.trace
    def ask(prompt: str) -> str:
        return "answer"

    with ll.span("pipeline", session_id="s1"):
        ask("q")
    assert ll.shutdown()
    lines = [
        json.loads(line)
        for path in (tmp_path / "logs").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    rows = query(tmp_path / "logs" / "llm.db", "SELECT name, session_id FROM records ORDER BY id")
    assert len(lines) == 2 and len(rows) == 2
    assert {row[1] for row in rows} == {"s1"}

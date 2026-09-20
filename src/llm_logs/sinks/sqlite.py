"""A single SQLite file, queryable with the ``llm-logs`` CLI or plain SQL."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_logs.record import Record

SCALAR_COLUMNS = (
    "schema_version",
    "lib_version",
    "trace_id",
    "span_id",
    "parent_span_id",
    "kind",
    "operation",
    "name",
    "session_id",
    "user_id",
    "start_time",
    "end_time",
    "duration_ms",
    "time_to_first_chunk_ms",
    "provider",
    "model",
    "response_model",
    "response_id",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
    "cost",
    "status",
    "error_type",
    "error_message",
    "streamed",
    "stream_outcome",
    "truncated",
)
JSON_COLUMNS = ("finish_reasons", "params", "input", "output", "provider_extras", "metadata")
COLUMNS = SCALAR_COLUMNS + JSON_COLUMNS

_TYPES = {
    "schema_version": "INTEGER",
    "duration_ms": "REAL",
    "time_to_first_chunk_ms": "REAL",
    "input_tokens": "INTEGER",
    "output_tokens": "INTEGER",
    "cache_read_input_tokens": "INTEGER",
    "cache_write_input_tokens": "INTEGER",
    "reasoning_output_tokens": "INTEGER",
    "cost": "REAL",
    "streamed": "INTEGER",
    "truncated": "INTEGER",
}
_INDEXED = ("trace_id", "start_time", "model", "status")
_BUSY_RETRIES = 7
_SCHEMA_VERSION = 1

_INSERT = (
    f"INSERT OR IGNORE INTO records ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in COLUMNS)})"
)

# Connections inherited through a fork are parked here and never used or closed
# in the child. SQLite does not support carrying a connection across fork().
_inherited: list[sqlite3.Connection] = []


def _schema() -> list[str]:
    columns = ", ".join(f'"{name}" {_TYPES.get(name, "TEXT")}' for name in COLUMNS)
    statements = [
        f"CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, {columns}, "
        "UNIQUE (trace_id, span_id))"
    ]
    statements += [
        f"CREATE INDEX IF NOT EXISTS idx_records_{name} ON records ({name})" for name in _INDEXED
    ]
    return statements


def row_from_record(record: Record) -> tuple[Any, ...]:
    data = record.model_dump(mode="json")
    row: list[Any] = []
    for name in SCALAR_COLUMNS:
        value = data[name]
        row.append(int(value) if isinstance(value, bool) else value)
    for name in JSON_COLUMNS:
        value = data[name]
        empty = value is None or value == {} or value == []
        row.append(None if empty else json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return tuple(row)


class SqliteSink:
    """Stores records in one table with indexes on trace, time, model and status.

    WAL mode lets readers (the CLI, a notebook) work while the application
    writes. SQLite still allows a single writer at a time across all processes,
    so each batch is one short transaction, guarded by a busy timeout and a few
    retries. With many worker processes prefer ``JsonlSink`` or ``OtelSink``.
    Do not put the file on a network filesystem: WAL needs shared memory.
    """

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout: float = 5.0) -> None:
        self._path = Path(path)
        self._busy_timeout = busy_timeout
        self._connection: sqlite3.Connection | None = None
        self._pid: int | None = None

    def write_batch(self, records: Sequence[Record]) -> None:
        if not records:
            return
        rows = [row_from_record(record) for record in records]
        delay = 0.05
        for attempt in range(_BUSY_RETRIES):
            try:
                self._insert(rows)
                return
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if attempt == _BUSY_RETRIES - 1 or not ("locked" in message or "busy" in message):
                    raise
                time.sleep(delay)
                delay *= 2

    def _insert(self, rows: list[tuple[Any, ...]]) -> None:
        # Connecting is inside the retried section on purpose: SQLite does not
        # apply the busy timeout to the switch into WAL mode, so two processes
        # that open the file at the same moment see "database is locked" there.
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.executemany(_INSERT, rows)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    def flush(self) -> None:
        """Nothing to do: every batch is committed when ``write_batch`` returns."""

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None and self._pid == os.getpid():
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._connection is not None and self._pid == pid:
            return self._connection
        if self._connection is not None:
            _inherited.append(self._connection)
            self._connection = None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            # Create the file ourselves so it is private from the first byte.
            os.close(os.open(self._path, os.O_WRONLY | os.O_CREAT, 0o600))
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout,
            isolation_level=None,  # explicit transactions only
            check_same_thread=False,
        )
        try:
            # WAL is a property of the file. Asking first avoids needing an
            # exclusive lock every time a worker starts.
            if connection.execute("PRAGMA journal_mode").fetchone() != ("wal",):
                connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            if connection.execute("PRAGMA user_version").fetchone() != (_SCHEMA_VERSION,):
                connection.execute("BEGIN IMMEDIATE")
                for statement in _schema():
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
                connection.execute("COMMIT")
        except BaseException:
            connection.close()  # never keep a half-initialised connection
            raise
        self._connection, self._pid = connection, pid
        return connection

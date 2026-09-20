"""The Sink protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from llm_logs.record import Record


@runtime_checkable
class Sink(Protocol):
    """Destination for batches of records.

    All three methods are called from the writer thread only, so a sink needs no
    locking of its own. A sink must open its resources lazily, inside
    ``write_batch``, and reopen them when ``os.getpid()`` changes: file handles
    and database connections must never be shared across a fork.

    A sink may raise. The writer catches it, counts the batch as failed for that
    sink, and carries on with the other sinks.
    """

    def write_batch(self, records: Sequence[Record]) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class InMemorySink:
    """Keeps records in a list. Meant for tests and interactive exploration."""

    def __init__(self) -> None:
        self.records: list[Record] = []
        self.flushed = 0
        self.closed = False

    def write_batch(self, records: Sequence[Record]) -> None:
        self.records.extend(records)

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed = True

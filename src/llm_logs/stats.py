"""Counters that make the library's own behaviour observable."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StatsSnapshot:
    """A point-in-time copy of the counters. Returned by ``llm_logs.stats()``."""

    enqueued: int = 0
    dropped: int = 0
    written: int = 0
    failed: int = 0
    redaction_errors: int = 0
    internal_errors: int = 0
    written_by_sink: dict[str, int] = field(default_factory=dict)
    failed_by_sink: dict[str, int] = field(default_factory=dict)


class Stats:
    """Thread-safe counters.

    - ``enqueued``: records accepted into the queue.
    - ``dropped``: records discarded because the queue was full or the writer
      could not run. Dropping is the designed response to overload.
    - ``written``: records delivered to at least one sink.
    - ``failed``: records that at least one sink failed to write.
    - ``redaction_errors``: records discarded because a redactor raised. They
      are never written unredacted.
    - ``internal_errors``: failures inside the library that were swallowed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enqueued = 0
        self._dropped = 0
        self._written = 0
        self._failed = 0
        self._redaction_errors = 0
        self._internal_errors = 0
        self._written_by_sink: dict[str, int] = {}
        self._failed_by_sink: dict[str, int] = {}

    def add_enqueued(self, n: int = 1) -> None:
        with self._lock:
            self._enqueued += n

    def add_dropped(self, n: int = 1) -> None:
        with self._lock:
            self._dropped += n

    def add_written(self, n: int) -> None:
        with self._lock:
            self._written += n

    def add_failed(self, n: int) -> None:
        with self._lock:
            self._failed += n

    def add_sink_written(self, sink: str, n: int) -> None:
        with self._lock:
            self._written_by_sink[sink] = self._written_by_sink.get(sink, 0) + n

    def add_sink_failed(self, sink: str, n: int) -> None:
        with self._lock:
            self._failed_by_sink[sink] = self._failed_by_sink.get(sink, 0) + n

    def add_redaction_error(self, n: int = 1) -> None:
        with self._lock:
            self._redaction_errors += n

    def add_internal_error(self, n: int = 1) -> None:
        with self._lock:
            self._internal_errors += n

    def snapshot(self) -> StatsSnapshot:
        with self._lock:
            return StatsSnapshot(
                enqueued=self._enqueued,
                dropped=self._dropped,
                written=self._written,
                failed=self._failed,
                redaction_errors=self._redaction_errors,
                internal_errors=self._internal_errors,
                written_by_sink=dict(self._written_by_sink),
                failed_by_sink=dict(self._failed_by_sink),
            )

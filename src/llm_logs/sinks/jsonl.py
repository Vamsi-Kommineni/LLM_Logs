"""Append-only JSON Lines files, one record per line."""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from llm_logs.record import Record

_DEFAULT_MAX_BYTES = 100 * 1024 * 1024


class JsonlSink:
    """Writes ``<prefix>-<YYYY-MM-DD>-p<pid>.<n>.jsonl`` files into a directory.

    - A new file starts every UTC day and whenever ``max_file_bytes`` is reached.
    - The PID is part of the name by default, so several worker processes never
      write to the same file.
    - Files are created readable by the owner only: they contain prompts.
    - ``retention_days`` deletes this sink's own files once they are older than that.

    The file is opened unbuffered and each batch is written with a single
    ``write`` call. Without a userspace buffer there is nothing a forked child
    could flush a second time, and a crash can lose at most the batch in flight.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        prefix: str = "llm_logs",
        include_pid: bool = True,
        max_file_bytes: int = _DEFAULT_MAX_BYTES,
        retention_days: int | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
            raise ValueError("prefix may only contain letters, digits, '_', '.' and '-'")
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")
        if retention_days is not None and retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        self._directory = Path(directory)
        self._prefix = prefix
        self._include_pid = include_pid
        self._max_file_bytes = max_file_bytes
        self._retention_days = retention_days
        self._fd: int | None = None
        self._pid: int | None = None
        self._day: str | None = None
        self._size = 0
        self._sequence = 0
        self.path: Path | None = None

    def write_batch(self, records: Sequence[Record]) -> None:
        if not records:
            return
        data = "".join(record.to_json() + "\n" for record in records).encode("utf-8")
        fd = self._current_fd(len(data))
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        self._size += len(data)

    def flush(self) -> None:
        if self._fd is not None and self._pid == os.getpid():
            os.fsync(self._fd)

    def close(self) -> None:
        self._release()

    def _release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            # Safe after a fork too: the descriptor table is per process and
            # there is no buffered data that could be written twice.
            with contextlib.suppress(OSError):
                os.close(fd)

    def _current_fd(self, incoming: int) -> int:
        pid = os.getpid()
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._fd is not None and self._pid == pid and self._day == day:
            if self._size == 0 or self._size + incoming <= self._max_file_bytes:
                return self._fd
            self._sequence += 1
        elif self._pid != pid or self._day != day:
            self._sequence = 0
        self._release()
        self._pid, self._day = pid, day
        self._directory.mkdir(parents=True, exist_ok=True)
        while True:
            path = self._path_for(day, pid, self._sequence)
            size = path.stat().st_size if path.exists() else 0
            if size == 0 or size + incoming <= self._max_file_bytes:
                break
            self._sequence += 1
        self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        self._size = size
        self.path = path
        self._apply_retention()
        return self._fd

    def _path_for(self, day: str, pid: int, sequence: int) -> Path:
        owner = f"-p{pid}" if self._include_pid else ""
        return self._directory / f"{self._prefix}-{day}{owner}.{sequence}.jsonl"

    def _apply_retention(self) -> None:
        if self._retention_days is None:
            return
        cutoff = (datetime.now(UTC) - timedelta(days=self._retention_days)).strftime("%Y-%m-%d")
        pattern = re.compile(
            rf"^{re.escape(self._prefix)}-(\d{{4}}-\d{{2}}-\d{{2}})(-p\d+)?\.\d+\.jsonl$"
        )
        for candidate in self._directory.iterdir():
            match = pattern.match(candidate.name)
            if match and match.group(1) < cutoff:
                with contextlib.suppress(OSError):
                    candidate.unlink()

"""The background writer: a bounded queue drained by one daemon thread.

Callers only ever do ``put_nowait``. Everything slow (pricing, redaction,
serialisation, disk, network) happens on the writer thread.

Lifecycle rules, because this is where subtle bugs hide:

- **Lazy start.** The thread starts on the first ``submit``. Since Python 3.12 a
  thread cannot be started while the interpreter is shutting down; in that case
  the record is counted as dropped.
- **Fork.** A ``Writer`` is never reused in a forked child. ``config`` replaces
  it with a fresh instance there, because this one's queue and locks may have
  been copied while another thread held them, and its thread does not exist in
  the child. Nothing here is drained, closed or otherwise touched after a fork.
- **Shutdown.** ``close`` is idempotent and callable from any thread. It asks
  the thread to drain, close the sinks and exit, and it never waits without a
  timeout. The thread is a daemon, so a stuck sink cannot keep the process alive.
- **Isolation.** A failing sink, redactor or pricing function affects only
  itself. The loop never dies from an exception.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from llm_logs import _internal
from llm_logs.record import Record
from llm_logs.sinks.base import Sink
from llm_logs.stats import Stats

_WAKE: Any = object()
_STOP: Any = object()


class Writer:
    def __init__(
        self,
        *,
        sinks: Sequence[Sink],
        redactors: Sequence[Callable[[Record], Record]],
        pricing: Callable[[Record], float | None] | None,
        queue_size: int,
        batch_size: int,
        flush_interval: float,
        stats: Stats,
    ) -> None:
        self._sinks = tuple(sinks)
        self._redactors = tuple(redactors)
        self._pricing = pricing
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._stats = stats
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._flush_requested = 0
        self._flush_completed = 0
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False

    # ------------------------------------------------------------ caller side

    def submit(self, record: Record) -> bool:
        """Hand a record to the writer. Never blocks, never raises."""
        try:
            if self._closed or (not self._started and not self._start()):
                self._stats.add_dropped()
                return False
            self._queue.put_nowait(record)
        except queue.Full:
            # Dropping is the designed answer to overload: the host application
            # is worth more than one log record.
            self._stats.add_dropped()
            return False
        except Exception:
            self._stats.add_internal_error()
            return False
        self._stats.add_enqueued()
        return True

    def flush(self, timeout: float) -> bool:
        """Block until everything submitted so far has reached the sinks.

        Returns False if that did not happen within ``timeout`` seconds.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            return self._queue.empty()
        if thread is threading.current_thread():
            return False
        with self._cond:
            self._flush_requested += 1
            target = self._flush_requested
        # A full queue means the thread is busy and will see the request anyway.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(_WAKE)
        with self._cond:
            return self._cond.wait_for(lambda: self._flush_completed >= target, timeout)

    def close(self, timeout: float) -> bool:
        """Drain, close the sinks and stop the thread. Idempotent."""
        with self._lock:
            already_closed = self._closed
            self._closed = True
            thread = self._thread
        if thread is None:
            if not already_closed:
                self._close_sinks()
            return True
        if thread is threading.current_thread():
            return False
        if not already_closed:
            try:
                self._queue.put(_STOP, timeout=timeout)
            except queue.Full:
                return False
        thread.join(timeout)
        return not thread.is_alive()

    def _start(self) -> bool:
        with self._lock:
            if self._started:
                return True
            if self._closed:
                return False
            thread = threading.Thread(target=self._run, name="llm-logs-writer", daemon=True)
            try:
                thread.start()
            except RuntimeError:
                return False  # interpreter is shutting down
            self._thread = thread
            self._started = True
            return True

    # ------------------------------------------------------------ thread side

    def _run(self) -> None:
        batch: list[Record] = []
        deadline = 0.0
        stop = False  # lives outside the try so an error cannot make us miss a stop
        while not stop:
            try:
                item = self._next_item(bool(batch), deadline)
                stop = item is _STOP
                if isinstance(item, Record):
                    if not batch:
                        deadline = time.monotonic() + self._flush_interval
                    batch.append(item)
                flush_target = self._pending_flush()
                due = (
                    stop
                    or flush_target > 0
                    or item is None
                    or len(batch) >= self._batch_size
                    or (bool(batch) and time.monotonic() >= deadline)
                )
                if not due:
                    continue
                if stop or flush_target:
                    stop = self._drain_into(batch) or stop
                if batch:
                    self._process(tuple(batch))
                    batch = []
                if flush_target:
                    self._flush_sinks()
                    self._complete_flush(flush_target)
            except Exception:
                batch = []
                self._stats.add_internal_error()
                _internal.warn("writer-loop", "llm_logs writer loop error", exc_info=True)
                time.sleep(0.05)
        self._close_sinks()
        with self._cond:
            self._flush_completed = self._flush_requested
            self._cond.notify_all()

    def _next_item(self, have_batch: bool, deadline: float) -> Any:
        try:
            if not have_batch:
                return self._queue.get()  # idle: sleep until there is work
            timeout = deadline - time.monotonic()
            if timeout > 0:
                return self._queue.get(timeout=timeout)
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _drain_into(self, batch: list[Record]) -> bool:
        """Move what is queued right now into ``batch``. Returns True if a stop was seen.

        Bounded by the queue size at entry, so a busy producer cannot keep a
        flush from finishing.
        """
        stop = False
        for _ in range(self._queue.qsize()):
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                stop = True
            elif isinstance(item, Record):
                batch.append(item)
                if len(batch) >= self._batch_size:
                    self._process(tuple(batch))
                    batch.clear()
        return stop

    def _pending_flush(self) -> int:
        with self._cond:
            if self._flush_requested > self._flush_completed:
                return self._flush_requested
            return 0

    def _complete_flush(self, target: int) -> None:
        with self._cond:
            self._flush_completed = max(self._flush_completed, target)
            self._cond.notify_all()

    def _process(self, batch: Sequence[Record]) -> None:
        ready: list[Record] = []
        for record in batch:
            prepared = self._prepare(record)
            if prepared is not None:
                ready.append(prepared)
        if not ready:
            return
        written = failed = False
        for sink in self._sinks:
            name = type(sink).__name__
            try:
                sink.write_batch(ready)
            except Exception:
                failed = True
                self._stats.add_sink_failed(name, len(ready))
                _internal.warn(f"sink-{name}", "llm_logs sink %s failed", name, exc_info=True)
            else:
                written = True
                self._stats.add_sink_written(name, len(ready))
        if written:
            self._stats.add_written(len(ready))
        if failed:
            self._stats.add_failed(len(ready))

    def _prepare(self, record: Record) -> Record | None:
        if self._pricing is not None and record.cost is None:
            try:
                cost = self._pricing(record)
                if cost is not None:
                    record = record.model_copy(update={"cost": float(cost)})
            except Exception:
                self._stats.add_internal_error()
                _internal.warn("pricing", "llm_logs pricing function failed", exc_info=True)
        try:
            for redactor in self._redactors:
                record = redactor(record)
                if not isinstance(record, Record):
                    raise TypeError("a redactor must return a Record")
        except Exception:
            # Fail closed: a record that could not be redacted is not written.
            self._stats.add_redaction_error()
            _internal.warn("redactor", "llm_logs redactor failed; record discarded", exc_info=True)
            return None
        return record

    def _flush_sinks(self) -> None:
        for sink in self._sinks:
            try:
                sink.flush()
            except Exception:
                name = type(sink).__name__
                _internal.warn(
                    f"flush-{name}", "llm_logs sink %s flush failed", name, exc_info=True
                )

    def _close_sinks(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                name = type(sink).__name__
                _internal.warn(
                    f"close-{name}", "llm_logs sink %s close failed", name, exc_info=True
                )

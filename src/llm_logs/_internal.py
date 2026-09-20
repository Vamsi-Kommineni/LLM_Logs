"""Package logger with rate limiting, shared by every module."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("llm_logs")
logger.addHandler(logging.NullHandler())

_INTERVAL_SECONDS = 60.0
_lock = threading.Lock()
_last_logged: dict[str, float] = {}
_suppressed: dict[str, int] = {}


def warn(key: str, message: str, *args: object, exc_info: bool = False) -> None:
    """Log a warning at most once per minute per ``key``.

    A sink that fails on every batch would otherwise write one line per batch
    into the host application's logs for as long as it stays broken.
    """
    try:
        now = time.monotonic()
        with _lock:
            last = _last_logged.get(key)
            if last is not None and now - last < _INTERVAL_SECONDS:
                _suppressed[key] = _suppressed.get(key, 0) + 1
                return
            _last_logged[key] = now
            skipped = _suppressed.pop(key, 0)
        if skipped:
            message += f" ({skipped} similar messages suppressed)"
        logger.warning(message, *args, exc_info=exc_info)
    except Exception:
        pass


def reset_rate_limits() -> None:
    """Forget what was logged. Used by tests and after a fork."""
    with _lock:
        _last_logged.clear()
        _suppressed.clear()

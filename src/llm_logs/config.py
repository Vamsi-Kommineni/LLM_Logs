"""Configuration and process-wide runtime state.

``configure()`` is the one place where this library raises: bad arguments fail
at startup instead of silently logging nothing in production. After it returns,
nothing in the library raises into caller code.
"""

from __future__ import annotations

import atexit
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llm_logs import _internal
from llm_logs.record import Record
from llm_logs.redact import DEFAULT_REDACTORS, Redactor
from llm_logs.serialize import MIN_BUDGET, UnknownPolicy
from llm_logs.sinks.base import Sink
from llm_logs.stats import Stats, StatsSnapshot
from llm_logs.writer import Writer

ENV_DISABLED = "LLM_LOGS_DISABLED"
ENV_SAMPLE_RATE = "LLM_LOGS_SAMPLE_RATE"
ENV_CAPTURE_CONTENT = "LLM_LOGS_CAPTURE_CONTENT"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}
_ATEXIT_TIMEOUT_SECONDS = 5.0

PricingFunction = Callable[[Record], "float | None"]


class ConfigurationError(ValueError):
    """Raised by ``configure()`` for invalid arguments."""


@dataclass(frozen=True, slots=True)
class Config:
    sinks: tuple[Sink, ...]
    redactors: tuple[Redactor, ...]
    sample_rate: float
    capture_content: bool
    max_payload_chars: int
    queue_size: int
    batch_size: int
    flush_interval: float
    unknown_objects: UnknownPolicy
    pricing: PricingFunction | None
    enabled: bool


class _Runtime:
    """Holds the active configuration and writer. One per process."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.config: Config | None = None
        self.writer: Writer | None = None
        self.stats = Stats()
        self.warned_unconfigured = False


_runtime = _Runtime()


def _new_writer(config: Config, stats: Stats) -> Writer:
    return Writer(
        sinks=config.sinks,
        redactors=config.redactors,
        pricing=config.pricing,
        queue_size=config.queue_size,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        stats=stats,
    )


def active() -> tuple[Config, Writer] | None:
    """Return the live config and writer, or None when tracing is off. Hot path."""
    rt = _runtime
    config, writer = rt.config, rt.writer
    if config is None or writer is None:
        if config is None and not rt.warned_unconfigured:
            rt.warned_unconfigured = True
            _internal.logger.warning(
                "llm_logs is used but configure() was never called; nothing will be recorded"
            )
        return None
    if not config.enabled:
        return None
    return config, writer


def record_internal_error(where: str) -> None:
    """Count and (rate-limited) log a failure that was swallowed to protect the caller."""
    try:
        _runtime.stats.add_internal_error()
        _internal.warn(f"internal-{where}", "llm_logs internal error in %s", where, exc_info=True)
    except Exception:
        pass


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    _internal.logger.warning("ignoring %s: not a boolean", name)
    return None


def _env_rate(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0.0 <= value <= 1.0:
        # A typo in an environment variable must not take the application down.
        _internal.logger.warning("ignoring %s: not a number between 0 and 1", name)
        return None
    return value


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def configure(
    *,
    sinks: Sequence[Sink],
    redactors: Sequence[Redactor] | None = None,
    sample_rate: float = 1.0,
    capture_content: bool = True,
    max_payload_chars: int = 20_000,
    queue_size: int = 10_000,
    batch_size: int = 100,
    flush_interval: float = 1.0,
    unknown_objects: UnknownPolicy = "type",
    pricing: PricingFunction | None = None,
    enabled: bool = True,
) -> None:
    """Set up sinks and behaviour. Call once at startup; calling again replaces the setup.

    Args:
        sinks: Where records go. At least one.
        redactors: Functions applied to each record on the writer thread before
            any sink. ``None`` selects the default (``redact.api_keys``); pass
            ``[]`` for none.
        sample_rate: Fraction of traces to keep, decided once per trace.
        capture_content: Record prompts and completions. When False, only
            metadata, parameters, timings and token counts are kept.
        max_payload_chars: Size limit for each of ``input`` and ``output``,
            measured on the compact JSON form.
        queue_size: Records that may wait for the writer. When full, new
            records are dropped and counted; the caller is never blocked.
        batch_size: Most records handed to a sink at once.
        flush_interval: Longest time in seconds a record waits before it is written.
        unknown_objects: ``"type"`` records objects of unknown type as a type
            name. ``"repr"`` records ``repr(obj)`` instead, which can leak
            whatever the object chooses to print.
        pricing: Optional ``Record -> cost``. Runs on the writer thread.
        enabled: Master switch.

    Environment variables override the arguments: ``LLM_LOGS_DISABLED``,
    ``LLM_LOGS_SAMPLE_RATE``, ``LLM_LOGS_CAPTURE_CONTENT``.

    Raises:
        ConfigurationError: if an argument is invalid.
    """
    if not isinstance(sinks, Sequence) or len(sinks) == 0:
        raise ConfigurationError("sinks must be a non-empty sequence")
    sink_list = tuple(sinks)
    for sink in sink_list:
        _check(isinstance(sink, Sink), f"{sink!r} does not implement write_batch/flush/close")
    redactor_list = DEFAULT_REDACTORS if redactors is None else tuple(redactors)
    for redactor in redactor_list:
        _check(callable(redactor), f"redactor {redactor!r} is not callable")
    _check(
        isinstance(sample_rate, (int, float)) and 0.0 <= sample_rate <= 1.0,
        "sample_rate must be between 0 and 1",
    )
    _check(isinstance(capture_content, bool), "capture_content must be a bool")
    _check(
        isinstance(max_payload_chars, int) and max_payload_chars >= MIN_BUDGET,
        f"max_payload_chars must be an int >= {MIN_BUDGET}",
    )
    _check(isinstance(queue_size, int) and queue_size >= 1, "queue_size must be an int >= 1")
    _check(isinstance(batch_size, int) and batch_size >= 1, "batch_size must be an int >= 1")
    _check(
        isinstance(flush_interval, (int, float)) and flush_interval > 0,
        "flush_interval must be a positive number",
    )
    _check(unknown_objects in ("type", "repr"), "unknown_objects must be 'type' or 'repr'")
    _check(pricing is None or callable(pricing), "pricing must be callable")
    _check(isinstance(enabled, bool), "enabled must be a bool")

    if _env_bool(ENV_DISABLED):
        enabled = False
    env_rate = _env_rate(ENV_SAMPLE_RATE)
    if env_rate is not None:
        sample_rate = env_rate
    env_content = _env_bool(ENV_CAPTURE_CONTENT)
    if env_content is not None:
        capture_content = env_content

    config = Config(
        sinks=sink_list,
        redactors=redactor_list,
        sample_rate=float(sample_rate),
        capture_content=capture_content,
        max_payload_chars=max_payload_chars,
        queue_size=queue_size,
        batch_size=batch_size,
        flush_interval=float(flush_interval),
        unknown_objects=unknown_objects,
        pricing=pricing,
        enabled=enabled,
    )
    with _runtime.lock:
        previous = _runtime.writer
        _runtime.stats = Stats()
        _runtime.config = config
        _runtime.writer = _new_writer(config, _runtime.stats)
    if previous is not None:
        previous.close(_ATEXIT_TIMEOUT_SECONDS)


def flush(timeout: float = 5.0) -> bool:
    """Wait until everything recorded so far has reached the sinks. Never raises."""
    try:
        writer = _runtime.writer
        return True if writer is None else writer.flush(timeout)
    except Exception:
        record_internal_error("flush")
        return False


def shutdown(timeout: float = 5.0) -> bool:
    """Flush, close the sinks and turn tracing off. Idempotent. Never raises."""
    try:
        with _runtime.lock:
            writer = _runtime.writer
            _runtime.writer = None
        return True if writer is None else writer.close(timeout)
    except Exception:
        record_internal_error("shutdown")
        return False


def stats() -> StatsSnapshot:
    """Counters for records enqueued, dropped, written and failed."""
    return _runtime.stats.snapshot()


def _reset_for_tests() -> None:
    shutdown(timeout=2.0)
    with _runtime.lock:
        _runtime.config = None
        _runtime.stats = Stats()
        _runtime.warned_unconfigured = False
    _internal.reset_rate_limits()


def _after_fork_in_child() -> None:
    """Give the child its own writer, locks and counters.

    The parent's writer thread does not exist here, and any lock may have been
    copied in the locked state. So everything is replaced and nothing inherited
    is drained, closed or otherwise touched. Sinks notice the new PID themselves.
    """
    rt = _runtime
    rt.lock = threading.Lock()
    rt.stats = Stats()
    _internal._lock = threading.Lock()
    config = rt.config
    if config is not None and rt.writer is not None:
        rt.writer = _new_writer(config, rt.stats)


def _at_exit() -> None:
    shutdown(_ATEXIT_TIMEOUT_SECONDS)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
atexit.register(_at_exit)

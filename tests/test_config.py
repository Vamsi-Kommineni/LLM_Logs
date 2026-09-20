from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from typing import Any

import pytest

import llm_logs as ll
from tests.conftest import flushed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sinks": []},
        {"sinks": [object()]},
        {"sinks": "logs/"},
        {"sample_rate": 1.5},
        {"sample_rate": "half"},
        {"max_payload_chars": 10},
        {"queue_size": 0},
        {"batch_size": 0},
        {"flush_interval": 0},
        {"capture_content": "yes"},
        {"redactors": ["not callable"]},
        {"unknown_objects": "dict"},
        {"pricing": 3},
    ],
)
def test_invalid_configuration_raises_at_startup(kwargs: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {"sinks": [ll.InMemorySink()], **kwargs}
    with pytest.raises(ll.ConfigurationError):
        ll.configure(**arguments)


def test_environment_variables_override_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    @ll.trace
    def ask(prompt: str) -> str:
        return "answer"

    sink = ll.InMemorySink()
    monkeypatch.setenv("LLM_LOGS_CAPTURE_CONTENT", "false")
    ll.configure(sinks=[sink], capture_content=True, flush_interval=0.01)
    ask("q")
    assert flushed(sink)[0].input is None

    monkeypatch.setenv("LLM_LOGS_SAMPLE_RATE", "0")
    ll.configure(sinks=[sink := ll.InMemorySink()], flush_interval=0.01)
    ask("q")
    assert flushed(sink) == []

    monkeypatch.delenv("LLM_LOGS_SAMPLE_RATE")
    monkeypatch.setenv("LLM_LOGS_DISABLED", "1")
    ll.configure(sinks=[sink := ll.InMemorySink()], flush_interval=0.01)
    assert ask("q") == "answer"
    assert flushed(sink) == [] and ll.stats().enqueued == 0


def test_a_typo_in_an_environment_variable_is_ignored_not_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="llm_logs")
    monkeypatch.setenv("LLM_LOGS_SAMPLE_RATE", "lots")
    monkeypatch.setenv("LLM_LOGS_CAPTURE_CONTENT", "maybe")
    ll.configure(sinks=[ll.InMemorySink()])
    assert sum("ignoring LLM_LOGS_" in message for message in caplog.messages) == 2


def test_reconfiguring_closes_the_previous_sinks() -> None:
    first, second = ll.InMemorySink(), ll.InMemorySink()

    @ll.trace
    def ask(prompt: str) -> str:
        return prompt

    ll.configure(sinks=[first], flush_interval=0.01)
    ask("one")
    ll.configure(sinks=[second], flush_interval=0.01)
    ask("two")
    assert [r.input["prompt"] for r in flushed(second)] == ["two"]
    assert [r.input["prompt"] for r in first.records] == ["one"] and first.closed


def test_pricing_runs_off_the_hot_path_and_cannot_break_anything() -> None:
    sink = ll.InMemorySink()

    def pricing(record: ll.Record) -> float | None:
        if record.name.endswith("broken"):
            raise ZeroDivisionError
        return 0.25

    ll.configure(sinks=[sink], pricing=pricing, flush_interval=0.01)
    ll.trace(lambda: 1, name="fine")()
    ll.trace(lambda: 1, name="broken")()
    fine, broken = flushed(sink)
    assert (fine.cost, broken.cost) == (0.25, None)


def test_importing_the_package_pulls_in_no_sdk() -> None:
    """The core must stay importable and light with no extras installed."""
    script = textwrap.dedent(
        """
        import sys
        import llm_logs
        llm_logs.configure(sinks=[llm_logs.InMemorySink()])
        llm_logs.trace(lambda: 1)()
        heavy = ("groq", "openai", "anthropic", "opentelemetry", "torch", "transformers",
                 "langchain", "httpx", "requests")
        loaded = sorted(m for m in heavy if m in sys.modules)
        print(",".join(loaded))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.strip() == "", f"unexpected imports: {result.stdout}"


def test_the_public_api_is_what_the_readme_says() -> None:
    for name in ("configure", "trace", "span", "flush", "shutdown", "stats", "redact"):
        assert hasattr(ll, name)
    for name in ("JsonlSink", "SqliteSink", "InMemorySink", "OtelSink", "Record"):
        assert name in ll.__all__

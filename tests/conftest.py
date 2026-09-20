from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

import llm_logs as ll
from llm_logs import config


@pytest.fixture(autouse=True)
def _clean_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (config.ENV_DISABLED, config.ENV_SAMPLE_RATE, config.ENV_CAPTURE_CONTENT):
        monkeypatch.delenv(name, raising=False)
    yield
    config._reset_for_tests()


@pytest.fixture
def sink() -> ll.InMemorySink:
    """An in-memory sink wired into a fast-flushing configuration."""
    memory = ll.InMemorySink()
    ll.configure(sinks=[memory], flush_interval=0.01, redactors=[])
    return memory


def flushed(sink: ll.InMemorySink) -> list[ll.Record]:
    assert ll.flush(5), "writer did not drain in time"
    return sink.records


class FakeStream:
    """Shaped like an SDK stream: iterator, context manager, ``close()``, extra attributes."""

    def __init__(self, chunks: list[Any], *, fail_at: int | None = None) -> None:
        self._chunks = list(chunks)
        self._index = 0
        self._fail_at = fail_at
        self.closed = False
        self.response = "raw-http-response"

    def __iter__(self) -> FakeStream:
        return self

    def __next__(self) -> Any:
        if self._fail_at is not None and self._index == self._fail_at:
            raise ConnectionError("stream broke")
        if self._index >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def helper(self) -> str:
        return "sdk helper still works"


class FakeLLM:
    """A provider stand-in with configurable latency, failure and streaming. No network."""

    def __init__(
        self,
        *,
        latency: float = 0.0,
        fail_with: BaseException | None = None,
        chunks: tuple[str, ...] = ("Hel", "lo", " world"),
    ) -> None:
        self.latency = latency
        self.fail_with = fail_with
        self.chunks = chunks
        self.api_key = "gsk_THISISAFAKEKEYTHATMUSTNEVERBELOGGED00"
        self.calls = 0

    def complete(self, prompt: str, **params: Any) -> dict[str, Any]:
        self.calls += 1
        if self.latency:
            time.sleep(self.latency)
        if self.fail_with is not None:
            raise self.fail_with
        return {"text": f"echo: {prompt}", "params": params}

    async def acomplete(self, prompt: str, **params: Any) -> dict[str, Any]:
        self.calls += 1
        await asyncio.sleep(self.latency)
        if self.fail_with is not None:
            raise self.fail_with
        return {"text": f"echo: {prompt}", "params": params}

    def stream(self, prompt: str, *, fail_at: int | None = None) -> FakeStream:
        return FakeStream(list(self.chunks), fail_at=fail_at)

    async def astream(self, prompt: str, *, fail_at: int | None = None) -> AsyncIterator[str]:
        for index, chunk in enumerate(self.chunks):
            if fail_at is not None and index == fail_at:
                raise ConnectionError("stream broke")
            await asyncio.sleep(0)
            yield chunk


def query(path: str | Path, sql: str) -> list[tuple[Any, ...]]:
    """Run one read-only query and close the connection again."""
    with contextlib.closing(sqlite3.connect(path)) as connection:
        return connection.execute(sql).fetchall()

"""Opt-in checks against the real Groq API. A few hundred tokens per run.

    GROQ_API_KEY=... GROQ_MODEL=<a current chat model> uv run pytest -m live

Skipped unless both variables are set, and never run in CI for pull requests.
The model comes from the environment because providers retire models often.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

import llm_logs as ll
from tests.conftest import flushed

MODEL = os.environ.get("GROQ_MODEL", "")
KEY = os.environ.get("GROQ_API_KEY", "")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not (KEY and MODEL), reason="GROQ_API_KEY and GROQ_MODEL are not both set"),
]

REQUEST: dict[str, Any] = {
    "messages": [{"role": "user", "content": "What is 2 + 2? Answer with the number only."}],
    "temperature": 0,
    "max_completion_tokens": 128,
    "reasoning_effort": "low",
}


def check(record: ll.Record, *, streamed: bool) -> None:
    assert record.status == "ok", record.error_message
    assert (record.provider, record.operation, record.model) == ("groq", "chat", MODEL)
    assert record.response_model and record.response_id
    assert record.input_tokens and record.output_tokens
    assert record.finish_reasons
    assert "4" in record.output["content"]
    assert record.params["temperature"] == 0 and record.params["max_completion_tokens"] == 128
    assert record.streamed is streamed
    assert "queue_time" in record.provider_extras
    assert KEY not in record.to_json()


def test_sync_call_through_the_real_sdk(sink: ll.InMemorySink) -> None:
    from groq import Groq

    client = Groq()

    @ll.trace
    def ask(client: Groq, **request: Any) -> Any:
        return client.chat.completions.create(model=MODEL, **request)

    ask(client, **REQUEST)
    (record,) = flushed(sink)
    check(record, streamed=False)
    assert record.input["client"].startswith("<groq."), "the client is a type name, nothing more"


def test_streaming_through_the_real_sdk_stream_class(sink: ll.InMemorySink) -> None:
    from groq import Groq

    create = ll.trace(Groq().chat.completions.create, name="groq.create")
    with create(model=MODEL, stream=True, **REQUEST) as stream:
        assert stream.response.status_code == 200, "SDK attributes pass through the proxy"
        text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    (record,) = flushed(sink)
    check(record, streamed=True)
    assert record.stream_outcome == "completed" and record.output["content"] == text
    assert record.time_to_first_chunk_ms and record.time_to_first_chunk_ms < record.duration_ms


async def test_async_streaming(sink: ll.InMemorySink) -> None:
    from groq import AsyncGroq

    client = AsyncGroq()
    create = ll.trace(client.chat.completions.create, name="groq.acreate")
    stream = await create(model=MODEL, stream=True, **REQUEST)
    async for _ in stream:
        pass
    await client.close()
    (record,) = flushed(sink)
    check(record, streamed=True)
    assert record.stream_outcome == "completed"


def test_an_authentication_error_is_recorded_without_the_key(sink: ll.InMemorySink) -> None:
    import groq

    bad_key = "gsk_" + "0" * 52
    create = ll.trace(groq.Groq(api_key=bad_key).chat.completions.create, name="groq.bad")
    with pytest.raises(groq.AuthenticationError):
        create(model=MODEL, **REQUEST)
    (record,) = flushed(sink)
    assert (record.status, record.error_type) == ("error", "AuthenticationError")
    assert bad_key not in record.to_json()

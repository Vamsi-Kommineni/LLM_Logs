"""A small retrieval-plus-chat service, traced end to end.

    pip install "llm-logs[groq,otel]" fastapi uvicorn
    uvicorn examples.fastapi_groq_rag.app:app --workers 4

Two providers:

- ``LLM_PROVIDER=fake`` (default): no network, no key, configurable latency and
  failures. This is what the load test uses for volume.
- ``LLM_PROVIDER=groq``: the real API. Needs ``GROQ_API_KEY`` and, if the default
  has been retired, ``GROQ_MODEL``.

Tracing is switched off entirely by ``LLM_LOGS_DISABLED=1``, without a code change.
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import llm_logs as ll

LOG_DIR = os.environ.get("LLM_LOGS_DIR", "logs")
PROVIDER = os.environ.get("LLM_PROVIDER", "fake")
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
FAIL_RATE = float(os.environ.get("FAKE_FAIL_RATE", "0.02"))

DOCUMENTS = {
    "refunds": "Refunds are issued within 14 days of purchase. Digital goods are not refundable.",
    "shipping": "Standard shipping takes 3 to 5 working days. Express shipping takes 1 day.",
    "support": "Support is available on weekdays from 9:00 to 17:00 by chat and email.",
}


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    user_id: str | None = None


def build_sinks() -> list[Any]:
    sinks: list[Any] = [ll.JsonlSink(LOG_DIR, retention_days=7)]
    if os.environ.get("USE_SQLITE", "1") == "1":
        sinks.append(ll.SqliteSink(os.path.join(LOG_DIR, "llm.db")))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        # Content stays out of the exported spans unless you opt in here.
        sinks.append(ll.OtelSink(service_name="rag-demo"))
    return sinks


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Configure inside the worker process, after uvicorn or gunicorn has forked.
    ll.configure(sinks=build_sinks(), redactors=[ll.redact.api_keys, ll.redact.emails])
    yield
    ll.shutdown()


app = FastAPI(lifespan=lifespan)


def retrieve(question: str) -> list[str]:
    with ll.span("retrieve", operation="retrieval") as span:
        words = question.lower()
        hits = [text for topic, text in DOCUMENTS.items() if topic[:-1] in words]
        span.set(input=question, output=hits)
        span.metadata["documents"] = len(hits)
        return hits or list(DOCUMENTS.values())[:1]


def build_messages(question: str, documents: list[str]) -> list[dict[str, str]]:
    context = "\n".join(f"- {doc}" for doc in documents)
    return [
        {"role": "system", "content": "Answer from the context only. Be brief."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


@ll.trace(provider="fake", operation="chat")
async def fake_stream(
    messages: list[dict[str, str]], *, model: str, temperature: float = 0.0
) -> AsyncIterator[str]:
    """Streams a canned answer. Sometimes fails, like a real provider."""
    await asyncio.sleep(random.uniform(0.005, 0.03))
    if random.random() < FAIL_RATE:
        raise ConnectionError("fake provider: upstream unavailable")
    context_line = messages[-1]["content"].splitlines()[1].removeprefix("- ")
    for word in f"According to our policy: {context_line}".split(" "):
        await asyncio.sleep(0.001)
        yield word + " "


_groq_create: Any = None


def groq_create() -> Any:
    """The SDK's own ``create`` method, traced. Built once per process."""
    global _groq_create
    if _groq_create is None:
        from groq import AsyncGroq

        client = AsyncGroq()  # reads GROQ_API_KEY by itself
        # Tracing the SDK call directly lets the Groq adapter see the real
        # request and the real chunks: model, parameters, token counts, time to
        # first chunk and finish reason all come from the provider.
        _groq_create = ll.trace(client.chat.completions.create, name="groq.chat.completions")
    return _groq_create


async def groq_stream(
    messages: list[dict[str, str]], *, model: str, temperature: float = 0.0
) -> AsyncIterator[str]:
    stream = await groq_create()(
        model=model, messages=messages, temperature=temperature, stream=True
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        # The span lives as long as the response is streamed, so the LLM call,
        # which runs while the client reads, is recorded as its child.
        async with ll.span(
            "rag_pipeline",
            session_id=request.session_id,
            user_id=request.user_id,
            metadata={"route": "/chat", "provider": PROVIDER},
        ):
            messages = build_messages(request.question, retrieve(request.question))
            generate = groq_stream if PROVIDER == "groq" else fake_stream
            async for piece in generate(messages, model=MODEL):
                yield piece

    return StreamingResponse(body(), media_type="text/plain")


@app.get("/stats")
async def stats() -> dict[str, Any]:
    snapshot = ll.stats()
    return {"pid": os.getpid(), **{k: getattr(snapshot, k) for k in snapshot.__dataclass_fields__}}

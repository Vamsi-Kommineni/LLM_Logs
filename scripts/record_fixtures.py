"""Re-record the Groq response fixtures used by the adapter tests.

    GROQ_API_KEY=... uv run python scripts/record_fixtures.py

Costs a few hundred tokens. Only response *bodies* are saved, never headers, and
every body goes through ``scrub`` so that no request, organisation or account
identifier lands in the repository. Review the diff before committing.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is 2 + 2? Answer with the number only."},
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

_IDENTIFIER_KEYS = {"id": "fixture-id", "system_fingerprint": "fp_fixture"}
_LOOKS_LIKE_ID = re.compile(r"\b(?:req|org|user|proj|chatcmpl|fc|call)[_-][A-Za-z0-9_-]{6,}")


def scrub(value: Any) -> Any:
    """Replace identifiers and timestamps; keep everything the adapters read."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _IDENTIFIER_KEYS and isinstance(item, str):
                out[key] = _IDENTIFIER_KEYS[key]
            elif key == "created" and isinstance(item, int):
                out[key] = 0
            else:
                out[key] = scrub(item)
        return out
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        return _LOOKS_LIKE_ID.sub("fixture-id", value)
    return value


def save(name: str, payload: Any) -> None:
    path = FIXTURES / f"{name}.json"
    path.write_text(json.dumps(scrub(payload), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(FIXTURES.parent.parent)}")


def main() -> int:
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set", file=sys.stderr)
        return 2
    from groq import Groq

    client = Groq()
    common: dict[str, Any] = {
        "model": MODEL,
        "temperature": 0,
        "max_completion_tokens": 256,
        "reasoning_effort": "low",
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)

    response = client.chat.completions.create(messages=MESSAGES, **common)
    save("groq_chat", response.model_dump(mode="json"))

    chunks = client.chat.completions.create(messages=MESSAGES, stream=True, **common)
    save("groq_chat_stream", [chunk.model_dump(mode="json") for chunk in chunks])

    # The SDK has no named parameter for this OpenAI-style option; the API accepts it.
    chunks = client.chat.completions.create(
        messages=MESSAGES,
        stream=True,
        extra_body={"stream_options": {"include_usage": True}},
        **common,
    )
    save("groq_chat_stream_include_usage", [chunk.model_dump(mode="json") for chunk in chunks])

    chunks = client.chat.completions.create(
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=TOOLS,
        tool_choice="required",
        stream=True,
        **common,
    )
    save("groq_tool_call_stream", [chunk.model_dump(mode="json") for chunk in chunks])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

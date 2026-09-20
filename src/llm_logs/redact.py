"""Redactors: functions that scrub a record before it reaches any sink.

Redactors run on the writer thread, never on the caller's thread. If a redactor
raises, the record is discarded and counted: it is never written unredacted.

Credential-like *key names* are handled earlier, at capture time, by
``serialize.to_jsonable`` and do not depend on anything configured here.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from llm_logs.record import Record
from llm_logs.serialize import REDACTED

Redactor = Callable[[Record], Record]

_PAYLOAD_FIELDS = ("input", "output", "params", "metadata", "provider_extras", "error_message")

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Shapes of widely used API credentials. A prompt with a pasted key, or an
# authentication error that echoes one back, should not reach disk.
_API_KEY = re.compile(
    r"""
    \b(?:
        sk-[A-Za-z0-9_-]{20,}                  # OpenAI, Anthropic and look-alikes
      | gsk_[A-Za-z0-9]{20,}                   # Groq
      | hf_[A-Za-z0-9]{20,}                    # Hugging Face
      | xai-[A-Za-z0-9]{20,}                   # xAI
      | AKIA[0-9A-Z]{16}                       # AWS access key ID
      | gh[pousr]_[A-Za-z0-9]{36,}             # GitHub tokens
      | github_pat_[A-Za-z0-9_]{40,}
      | AIza[0-9A-Za-z_-]{35}                  # Google API key
      | eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}   # JWT
    )
    | \bBearer\s+[A-Za-z0-9._~+/-]{20,}=*
    """,
    re.VERBOSE,
)


def _map(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    return value


def map_strings(record: Record, fn: Callable[[str], str]) -> Record:
    """Return a copy of ``record`` with ``fn`` applied to every string in its payload fields."""
    update = {name: _map(getattr(record, name), fn) for name in _PAYLOAD_FIELDS}
    return record.model_copy(update=update)


def regex(pattern: str | re.Pattern[str], replacement: str = REDACTED) -> Redactor:
    """Build a redactor that replaces every match of ``pattern``."""
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern

    def _sub(text: str) -> str:
        return compiled.sub(replacement, text)

    def redactor(record: Record) -> Record:
        return map_strings(record, _sub)

    return redactor


def keys(*names: str, replacement: str = REDACTED) -> Redactor:
    """Build a redactor that blanks the values stored under the given dict keys, at any depth."""
    wanted = {n.lower() for n in names}

    def _scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: replacement if isinstance(k, str) and k.lower() in wanted else _scrub(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [_scrub(v) for v in value]
        return value

    def redactor(record: Record) -> Record:
        update = {
            name: _scrub(getattr(record, name))
            for name in _PAYLOAD_FIELDS
            if name != "error_message"
        }
        return record.model_copy(update=update)

    return redactor


emails: Redactor = regex(_EMAIL, "[EMAIL]")
api_keys: Redactor = regex(_API_KEY, REDACTED)

DEFAULT_REDACTORS: tuple[Redactor, ...] = (api_keys,)

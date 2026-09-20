"""The Record schema: one Record per span."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

Kind = Literal["llm", "span"]
Status = Literal["ok", "error"]
StreamOutcome = Literal["completed", "closed_early", "abandoned", "error"]


class Record(BaseModel):
    """Everything known about one traced call.

    ``extra="forbid"`` applies to construction, so a typo in this library fails
    in tests. Readers (the CLI) load stored rows as plain dicts instead, which
    lets an older reader cope with rows written by a newer version.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    lib_version: str

    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    parent_span_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")

    kind: Kind
    operation: str | None = None
    name: str
    session_id: str | None = None
    user_id: str | None = None

    start_time: datetime
    end_time: datetime
    duration_ms: float
    time_to_first_chunk_ms: float | None = None

    provider: str | None = None
    # ``model`` is what was requested when the request shows it, otherwise it
    # falls back to the model the provider reported. ``response_model`` is
    # always the provider's own answer.
    model: str | None = None
    response_model: str | None = None
    response_id: str | None = None
    finish_reasons: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    input: Any = None
    output: Any = None

    # ``input_tokens`` is the total prompt size including cached tokens, for
    # every provider, so that numbers are comparable across adapters.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    cost: float | None = None

    status: Status = "ok"
    error_type: str | None = None
    error_message: str | None = None

    streamed: bool = False
    stream_outcome: StreamOutcome | None = None

    provider_extras: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False

    def to_json(self) -> str:
        """Serialise to one compact JSON line."""
        return self.model_dump_json()

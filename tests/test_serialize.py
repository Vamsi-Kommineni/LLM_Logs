from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import uuid
from collections import namedtuple
from pathlib import PurePosixPath
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, Field, SecretStr

from llm_logs.serialize import MIN_BUDGET, REDACTED, is_sensitive_key, json_size, to_jsonable

SECRET = "gsk_THISISAFAKEKEYTHATMUSTNEVERBELOGGED00"


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# ------------------------------------------------------------------ property

json_like = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats()  # includes nan and inf on purpose
    | st.text()
    | st.binary(max_size=64),
    lambda children: (
        st.lists(children, max_size=8)
        | st.dictionaries(st.text(max_size=12) | st.integers(), children, max_size=8)
        | st.tuples(children, children)
        | st.frozensets(st.text(max_size=8), max_size=4)
    ),
    max_leaves=60,
)


@settings(max_examples=400, deadline=None)
@given(value=json_like, limit=st.integers(min_value=MIN_BUDGET, max_value=3000))
def test_output_is_valid_json_and_within_the_limit(value: Any, limit: int) -> None:
    out, _ = to_jsonable(value, max_chars=limit)
    encoded = dumps(out)  # allow_nan=False: raises on NaN/Infinity
    assert len(encoded) <= limit
    assert json.loads(encoded) == out
    encoded.encode("utf-8")  # no lone surrogates


@settings(max_examples=200, deadline=None)
@given(
    text=st.text(
        alphabet=st.characters(codec="utf-8") | st.sampled_from(['"', "\\", "\n", "\x00", "\x1f"]),
        min_size=200,
        max_size=5000,
    ),
    limit=st.integers(min_value=MIN_BUDGET, max_value=1500),
)
def test_strings_full_of_escapes_stay_within_the_limit(text: str, limit: int) -> None:
    out, _ = to_jsonable({"messages": [{"role": "user", "content": text}]}, max_chars=limit)
    assert json_size(out) <= limit


@settings(max_examples=100, deadline=None)
@given(value=json_like)
def test_small_values_round_trip_unchanged_when_they_fit(value: Any) -> None:
    out, truncated = to_jsonable(value, max_chars=1_000_000)
    assert dumps(out)  # always valid
    if not truncated:
        assert json_size(out) <= 1_000_000


# --------------------------------------------------------------- credentials


class SdkClient:
    """Shaped like a provider SDK client: the key is an ordinary attribute."""

    def __init__(self) -> None:
        self.api_key = SECRET
        self._headers = {"Authorization": f"Bearer {SECRET}"}

    def __repr__(self) -> str:
        return f"SdkClient(api_key={self.api_key!r})"


def test_unknown_objects_become_type_names_and_are_never_walked() -> None:
    out, _ = to_jsonable({"client": SdkClient(), "prompt": "hi"})
    assert out == {"client": f"<{SdkClient.__module__}.SdkClient>", "prompt": "hi"}
    assert SECRET not in dumps(out)


def test_repr_is_only_used_when_asked_for() -> None:
    out, _ = to_jsonable(SdkClient(), unknown="repr")
    assert out.startswith("SdkClient(")


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "apiKey",
        "X-Api-Key",
        "Authorization",
        "token",
        "access_token",
        "refreshToken",
        "hf_token",
        "client_secret",
        "password",
        "Cookie",
        "aws_credentials",
    ],
)
def test_sensitive_keys_are_scrubbed(key: str) -> None:
    assert is_sensitive_key(key)
    out, _ = to_jsonable({"outer": [{key: SECRET}]})
    assert out == {"outer": [{key: REDACTED}]}


@pytest.mark.parametrize(
    "key",
    [
        "max_tokens",
        "input_tokens",
        "completion_tokens",
        "eos_token",
        "pad_token",
        "stop_token",
        "time_to_first_token",
        "prompt",
        "keywords",
        "monkey",
    ],
)
def test_ordinary_llm_vocabulary_is_not_scrubbed(key: str) -> None:
    assert not is_sensitive_key(key)


def test_secret_fields_of_models_and_dataclasses_are_not_recorded() -> None:
    class Settings(BaseModel):
        name: str
        api_key: str
        vault: SecretStr
        hidden: str = Field(default="h", repr=False)

    @dataclasses.dataclass
    class Conn:
        host: str
        password: str
        internal: str = dataclasses.field(default="x", repr=False)

    out, _ = to_jsonable(
        [Settings(name="n", api_key=SECRET, vault=SecretStr(SECRET)), Conn("db", SECRET)]
    )
    assert out == [
        {"name": "n", "api_key": REDACTED, "vault": "<pydantic.types.SecretStr>"},
        {"host": "db", "password": REDACTED},
    ]
    assert SECRET not in dumps(out)


# ------------------------------------------------------------------- shapes


class Colour(enum.Enum):
    RED = "red"


class Level(enum.IntEnum):
    HIGH = 3


Point = namedtuple("Point", "x y")


def test_supported_types() -> None:
    @dataclasses.dataclass
    class Doc:
        title: str
        score: float
        tags: list[str]
        note: str | None = None

    class Msg(BaseModel):
        model_config = {"extra": "allow"}
        role: str
        content: str | None = None

    value = {
        "doc": Doc("t", 0.5, ["a"]),
        "msg": Msg(role="user", content="hi", x_extra=1),  # type: ignore[call-arg]
        "when": dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC),
        "id": uuid.UUID(int=1),
        "path": PurePosixPath("data/file.txt"),
        "colour": Colour.RED,
        "level": Level.HIGH,
        "point": Point(1, 2),
        "set": frozenset({"only"}),
        "bytes": b"\x00" * 10,
        "delta": dt.timedelta(seconds=1.5),
        "big": 2**70,
        "nan": float("nan"),
        7: "int key",
    }
    out, truncated = to_jsonable(value)
    assert not truncated
    assert out == {
        "doc": {"title": "t", "score": 0.5, "tags": ["a"]},
        "msg": {"role": "user", "content": "hi", "x_extra": 1},
        "when": "2026-01-02T03:04:05+00:00",
        "id": "00000000-0000-0000-0000-000000000001",
        "path": "data/file.txt",
        "colour": "red",
        "level": 3,
        "point": {"x": 1, "y": 2},
        "set": ["only"],
        "bytes": "<bytes len=10>",
        "delta": 1.5,
        "big": str(2**70),
        "nan": "nan",
        "7": "int key",
    }


def test_media_never_enters_a_record() -> None:
    data_url = "data:image/png;base64," + "A" * 5000
    out, _ = to_jsonable({"image_url": data_url, "blob": "QUJD" * 1000, "text": "word " * 500})
    assert out["image_url"] == f"<data-url mime=image/png len={len(data_url)}>"
    assert out["blob"] == "<base64 len=4000>"
    assert out["text"].startswith("word word")


def test_cycles_terminate() -> None:
    loop: dict[str, Any] = {"name": "loop"}
    loop["self"] = loop
    items: list[Any] = [1]
    items.append(items)
    out, truncated = to_jsonable({"loop": loop, "items": items})
    assert truncated
    assert out == {"loop": {"name": "loop", "self": "<cycle>"}, "items": [1, "<cycle>"]}


def test_depth_is_limited() -> None:
    nested: Any = "leaf"
    for _ in range(100):
        nested = [nested]
    out, truncated = to_jsonable(nested, max_chars=10_000)
    assert truncated
    assert "<max depth reached>" in dumps(out)


def test_lone_surrogates_are_replaced() -> None:
    out, _ = to_jsonable({"text": "ok \ud800 end"})
    dumps(out).encode("utf-8")


def test_objects_that_misbehave_cannot_break_the_walk() -> None:
    class Evil(dict):  # type: ignore[type-arg]
        def items(self) -> Any:
            raise RuntimeError("boom")

    out, truncated = to_jsonable({"evil": Evil(a=1), "fine": 1})
    assert truncated
    assert out["fine"] == 1
    assert out["evil"].startswith("<unserializable ")


# --------------------------------------------------------------- truncation


def test_long_strings_keep_head_and_tail() -> None:
    text = "HEAD " + "lorem ipsum " * 1_000 + "TAIL"
    out, truncated = to_jsonable(text, max_chars=500)
    assert truncated
    assert out.startswith("HEAD") and out.endswith("TAIL")
    assert "...[truncated " in out
    assert json_size(out) <= 500


def test_long_lists_keep_first_and_last_items() -> None:
    out, truncated = to_jsonable(list(range(10_000)), max_chars=2_000)
    assert truncated
    assert out[0] == 0 and out[-1] == 9_999
    assert any(isinstance(item, str) and "items omitted" in item for item in out)
    assert json_size(out) <= 2_000


def test_a_huge_system_prompt_does_not_crowd_out_the_last_turn() -> None:
    messages = [
        {"role": "system", "content": "system rule. " * 4_000},
        {"role": "user", "content": "What is the refund policy?"},
        {"role": "assistant", "content": "long answer. " * 4_000},
        {"role": "user", "content": "And for digital goods?"},
    ]
    out, truncated = to_jsonable({"messages": messages}, max_chars=4_000)
    assert truncated
    assert json_size(out) <= 4_000
    contents = [m["content"] for m in out["messages"]]
    assert contents[1] == "What is the refund policy?"
    assert contents[3] == "And for digital goods?"
    assert contents[0].startswith("system rule.") and contents[2].endswith("long answer. ")


def test_cost_depends_on_the_limit_not_on_the_payload() -> None:
    import time

    huge = {"messages": [{"role": "user", "content": "x" * 200} for _ in range(200_000)]}
    started = time.perf_counter()
    out, truncated = to_jsonable(huge, max_chars=20_000)
    elapsed = time.perf_counter() - started
    assert truncated and json_size(out) <= 20_000
    assert elapsed < 0.25, f"walk took {elapsed:.3f}s; it must not scale with the payload"

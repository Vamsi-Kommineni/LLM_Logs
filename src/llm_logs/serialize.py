"""Turn arbitrary Python values into bounded, JSON-safe, credential-free data.

This runs on the caller's thread, so it has three jobs and does nothing else:

1. **Never leak credentials.** Only an allow-list of types is walked. Anything
   else becomes a type-name placeholder such as ``"<groq.Groq>"``. There is no
   ``__dict__`` walk and no ``repr`` fallback, because SDK client objects keep
   the API key in an attribute and a traced function often receives the client.
   Values stored under credential-like key names are replaced as well.
2. **Stay bounded.** The walk carries a character budget and stops descending
   when it is spent, so the cost depends on the limit, not on the payload.
3. **Always produce valid JSON**: no NaN, no lone surrogates, no cycles.

Guarantee, enforced by a property test::

    len(json.dumps(out, ensure_ascii=False, separators=(",", ":"))) <= max_chars
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import decimal
import enum
import functools
import itertools
import json
import math
import pathlib
import re
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, Literal

from pydantic import BaseModel

UnknownPolicy = Literal["type", "repr"]

MIN_BUDGET = 256
REDACTED = "[REDACTED]"

_MIN_NODE = 32  # every walk() call is guaranteed at least this much budget
_MARKER_RESERVE = 41  # room kept in every container for one omission marker
_MARKER_MAX = 40  # longest string truncation marker
# Below this a container is summarised instead of opened: it needs room for its
# brackets, its omission marker and at least one child.
_CONTAINER_MIN = 2 + _MARKER_RESERVE + _MIN_NODE + 1
_MAX_DEPTH = 16
_MAX_ITEMS = 200
_MAX_KEY_CHARS = 128
_MEDIA_MIN_CHARS = 1024
_REPR_MAX_CHARS = 200

_encode_str = json.JSONEncoder(ensure_ascii=False).encode

_DATA_URL = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?[^,]{0,100};base64,", re.IGNORECASE)
_BASE64_RUN = re.compile(r"^[A-Za-z0-9+/_=-]{256}")

# Credential-like key names. Matching happens on a lower-cased key with "-"
# folded to "_". "token" needs care: "max_tokens" and "eos_token" are ordinary
# LLM vocabulary and must survive.
_SENSITIVE_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "authorization",
    "cookie",
    "credential",
    "private_key",
    "privatekey",
)
_SENSITIVE_EXACT = frozenset({"token", "auth", "bearer", "key", "pwd", "pass"})
_BENIGN_TOKEN_PREFIXES = (
    "eos",
    "bos",
    "pad",
    "unk",
    "sep",
    "cls",
    "mask",
    "stop",
    "next",
    "first",
    "last",
    "max",
    "min",
    "per",
    "time_to_first",
)


@functools.lru_cache(maxsize=4096)
def is_sensitive_key(key: str) -> bool:
    """True if a value stored under ``key`` should never be recorded.

    Cached: payloads repeat the same few keys ("role", "content", ...) endlessly.
    """
    k = key.lower().replace("-", "_")
    if k in _SENSITIVE_EXACT:
        return True
    if any(fragment in k for fragment in _SENSITIVE_FRAGMENTS):
        return True
    if k.endswith("token"):
        prefix = k[: -len("token")].rstrip("_")
        return not any(prefix == p or prefix.endswith("_" + p) for p in _BENIGN_TOKEN_PREFIXES)
    return False


def json_size(value: Any) -> int:
    """Length of the compact JSON form. This is the measure the limit refers to."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def to_jsonable(
    obj: Any, *, max_chars: int = 20_000, unknown: UnknownPolicy = "type"
) -> tuple[Any, bool]:
    """Return ``(json_safe_value, truncated)``. Never raises."""
    walker = _Walker(max(max_chars, MIN_BUDGET), unknown)
    try:
        return walker.walk(obj, 0), walker.truncated
    except Exception:  # pragma: no cover - walk() already guards every node
        return "<unserializable>", True


def type_name(obj: Any) -> str:
    t = type(obj)
    return f"<{t.__module__}.{t.__qualname__}>"


def _clean(s: str) -> str:
    """Replace lone surrogates, which cannot be encoded as UTF-8."""
    if s.isascii():
        return s
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return s.encode("utf-8", "replace").decode("utf-8")
    return s


def _media_placeholder(s: str) -> str | None:
    match = _DATA_URL.match(s)
    if match:
        return f"<data-url mime={match.group('mime') or 'unknown'} len={len(s)}>"
    if _BASE64_RUN.match(s):
        return f"<base64 len={len(s)}>"
    return None


def _is_scalar(v: Any) -> bool:
    return v is None or type(v) in (bool, int, float)


def _estimate(obj: Any) -> int:
    """Cheap one-level size guess used to share a tight budget between siblings.

    Containers include the headroom they need to be opened at all
    (``_CONTAINER_MIN``). Guessing high is harmless: a child only spends what it
    really costs and the rest of its allowance goes back to the pool.
    """
    try:
        t = type(obj)
        if t is str:
            return len(obj) + 2
        if _is_scalar(obj):
            return 8
        nested = 64 + _CONTAINER_MIN
        values: Any
        if isinstance(obj, BaseModel):
            values = obj.__dict__
        elif t is dict:
            values = obj
        elif t in (list, tuple):
            total = _CONTAINER_MIN
            for i, v in enumerate(obj):
                if i >= 32:
                    return total * len(obj) // 32
                cost = (len(v) + 2) if type(v) is str else 8 if _is_scalar(v) else nested
                total += 1 + max(cost, _MIN_NODE)
            return total
        else:
            return nested
        total = _CONTAINER_MIN
        for i, (k, v) in enumerate(values.items()):
            if i >= 32:
                return total * len(values) // 32
            total += (len(k) if type(k) is str else 8) + 4
            cost = (len(v) + 2) if type(v) is str else 8 if _is_scalar(v) else nested
            total += max(cost, _MIN_NODE)
        return total
    except Exception:
        return 64 + _CONTAINER_MIN


def _water_fill(estimates: list[int], budget: int, floor: int) -> list[int] | None:
    """Split ``budget`` fairly: small items get what they need, large ones share the rest.

    Returns ``None`` when everything is expected to fit and no limits are needed.
    """
    n = len(estimates)
    if sum(estimates) <= budget:
        return None
    allowances = [0] * n
    left = budget
    for rank, i in enumerate(sorted(range(n), key=estimates.__getitem__)):
        share = left // (n - rank)
        allowance = max(min(estimates[i], share), floor)
        allowances[i] = allowance
        left = max(left - allowance, 0)
    return allowances


class _Walker:
    __slots__ = ("_path", "_unknown", "remaining", "truncated")

    def __init__(self, budget: int, unknown: UnknownPolicy) -> None:
        self.remaining = budget
        self.truncated = False
        self._unknown = unknown
        self._path: set[int] = set()

    # Invariant: walk() is only called while remaining >= _MIN_NODE, and it
    # returns a value whose exact JSON cost has been deducted from remaining.
    def walk(self, obj: Any, depth: int) -> Any:
        before = self.remaining
        try:
            return self._dispatch(obj, depth)
        except Exception:
            # The partial result is discarded, so its budget comes back.
            self.remaining = before
            self.truncated = True
            return self._text(f"<unserializable {type_name(obj)[1:-1]}>")

    def _walk_within(self, obj: Any, depth: int, allowance: int) -> Any:
        if allowance >= self.remaining:
            return self.walk(obj, depth)
        held_back = self.remaining - allowance
        self.remaining = allowance
        try:
            return self.walk(obj, depth)
        finally:
            self.remaining += held_back

    def _dispatch(self, obj: Any, depth: int) -> Any:
        if obj is None:
            self.remaining -= 4
            return None
        t = type(obj)
        if t is str:
            return self._text(obj)
        if t is bool:
            self.remaining -= 4 if obj else 5
            return obj
        if t is int:
            return self._int(obj)
        if t is float:
            return self._float(obj)
        if depth >= _MAX_DEPTH:
            self.truncated = True
            return self._text("<max depth reached>")
        if t is dict:
            return self._mapping(obj, depth)
        if t is list or t is tuple:
            return self._sequence(obj, depth)

        if isinstance(obj, enum.Enum):
            return self.walk(obj.value, depth + 1)
        if isinstance(obj, str):
            return self._text(str.__str__(obj))
        if isinstance(obj, bool):  # pragma: no cover - bool cannot be subclassed
            return self.walk(bool(obj), depth)
        if isinstance(obj, int):
            return self._int(int(obj))
        if isinstance(obj, float):
            return self._float(float(obj))
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return self._text(f"<bytes len={len(obj)}>")
        if isinstance(obj, BaseModel):
            return self._pairs(self._model_pairs(obj), 0, id(obj), depth)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return self._pairs(self._dataclass_pairs(obj), 0, id(obj), depth)
        if isinstance(obj, Mapping):
            return self._mapping(obj, depth)
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return self._pairs(list(zip(obj._fields, obj, strict=False)), 0, id(obj), depth)
        if isinstance(obj, (Sequence, AbstractSet, deque)):
            return self._sequence(obj, depth)
        if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
            return self._text(obj.isoformat())
        if isinstance(obj, _dt.timedelta):
            return self._float(obj.total_seconds())
        if isinstance(obj, (uuid.UUID, decimal.Decimal, pathlib.PurePath)):
            return self._text(str(obj))
        return self._opaque(obj)

    # ---------------------------------------------------------------- scalars

    def _int(self, i: int) -> Any:
        if i.bit_length() > 64:
            # Too large for most JSON readers, and str() of a huge int is slow.
            if i.bit_length() > 512:
                self.truncated = True
                return self._text(f"<int bits={i.bit_length()}>")
            return self._text(str(i))
        self.remaining -= len(str(i))
        return i

    def _float(self, f: float) -> Any:
        if not math.isfinite(f):
            return self._text("nan" if math.isnan(f) else "inf" if f > 0 else "-inf")
        self.remaining -= len(repr(f))
        return f

    def _opaque(self, obj: Any) -> str:
        if self._unknown == "repr":
            try:
                text = repr(obj)
            except Exception:
                text = type_name(obj)
            if len(text) > _REPR_MAX_CHARS:
                text = text[:_REPR_MAX_CHARS] + "..."
            return self._text(text)
        return self._text(type_name(obj))

    def _text(self, s: str) -> str:
        n = len(s)
        if n >= _MEDIA_MIN_CHARS:
            placeholder = _media_placeholder(s)
            if placeholder is not None:
                s, n = placeholder, len(placeholder)
        if n + 2 <= self.remaining:
            s = _clean(s)
            cost = len(_encode_str(s))
            if cost <= self.remaining:
                self.remaining -= cost
                return s
        self.truncated = True
        avail = min(n, self.remaining - 2)
        while avail > 0:
            keep = avail - _MARKER_MAX
            if keep >= 16:
                # Keep both ends: in a chat, the system prompt sits at the head
                # and the turn that produced the answer sits at the tail.
                head = (keep + 1) // 2
                tail = keep - head
                candidate = s[:head] + f"...[truncated {n - keep} chars]..." + s[n - tail :]
            else:
                candidate = s[:avail]
            candidate = _clean(candidate)
            cost = len(_encode_str(candidate))
            if cost <= self.remaining:
                self.remaining -= cost
                return candidate
            avail = min(avail - 1, avail * self.remaining // cost)
        self.remaining -= 2
        return ""

    # ------------------------------------------------------------- containers

    def _summary(self, kind: str, size: int) -> str:
        self.truncated = True
        return self._text(f"<{kind} len={size}>")

    def _item_limit(self, per_item: int) -> int:
        usable = self.remaining - 2 - _MARKER_RESERVE
        return max(min(_MAX_ITEMS, usable // per_item), 0)

    def _sequence(self, obj: Any, depth: int) -> Any:
        oid = id(obj)
        if oid in self._path:
            self.truncated = True
            return self._text("<cycle>")
        total = len(obj)
        if total and self.remaining < _CONTAINER_MIN:
            return self._summary("list", total)
        limit = self._item_limit(_MIN_NODE + 1)
        if total <= limit:
            children, split, omitted = list(obj), 0, 0
        elif isinstance(obj, (list, tuple)):
            head = (limit + 1) // 2
            tail = limit - head
            children = list(obj[:head]) + (list(obj[total - tail :]) if tail else [])
            split, omitted = head, total - limit
        else:
            children = list(itertools.islice(obj, limit))
            split, omitted = limit, total - limit

        self._path.add(oid)
        try:
            return self._emit_list(children, split, omitted, depth)
        finally:
            self._path.discard(oid)

    def _emit_list(self, children: list[Any], split: int, omitted: int, depth: int) -> list[Any]:
        self.remaining -= 2
        out: list[Any] = []
        allowances = _water_fill(
            [
                len(c) + 3 if type(c) is str else 9 if _is_scalar(c) else _estimate(c) + 1
                for c in children
            ],
            self.remaining - _MARKER_RESERVE,
            _MIN_NODE + 1,
        )
        marker_pending = omitted > 0
        for index, child in enumerate(children):
            if marker_pending and index == split:
                out.append(self._list_marker(omitted, bool(out)))
                marker_pending = False
            if self.remaining - _MARKER_RESERVE < _MIN_NODE + 1:
                self.truncated = True
                if marker_pending or omitted == 0:
                    out.append(self._list_marker(omitted + len(children) - index, bool(out)))
                return out
            if out:
                self.remaining -= 1
            # A child may never spend the room reserved for this list's marker.
            allowance = self.remaining - _MARKER_RESERVE
            if allowances is not None:
                allowance = min(allowance, max(allowances[index] - 1, _MIN_NODE))
            out.append(self._walk_within(child, depth + 1, allowance))
        if marker_pending:
            out.append(self._list_marker(omitted, bool(out)))
        return out

    def _list_marker(self, omitted: int, needs_comma: bool) -> str:
        self.truncated = True
        marker = f"...[{omitted} items omitted]..."
        self.remaining -= len(marker) + 2 + (1 if needs_comma else 0)
        return marker

    def _mapping(self, obj: Any, depth: int) -> Any:
        total = len(obj)
        limit = self._item_limit(_MIN_NODE + 4)
        if total <= limit:
            pairs, omitted = list(obj.items()), 0
        else:
            head = (limit + 1) // 2
            tail = limit - head
            pairs = list(itertools.islice(obj.items(), head))
            if tail and isinstance(obj, dict):
                pairs += list(itertools.islice(reversed(obj.items()), tail))[::-1]
            omitted = total - len(pairs)
        return self._pairs(pairs, omitted, id(obj), depth)

    def _model_pairs(self, obj: BaseModel) -> list[tuple[Any, Any]]:
        values = obj.__dict__
        pairs: list[tuple[Any, Any]] = []
        for name, info in type(obj).model_fields.items():
            if info.exclude or info.repr is False:
                continue  # fields hidden from repr or dumps are usually secrets
            value = values.get(name)
            if value is not None:
                pairs.append((name, value))
        extra = obj.model_extra
        if extra:
            pairs.extend((k, v) for k, v in extra.items() if v is not None)
        return pairs

    def _dataclass_pairs(self, obj: Any) -> list[tuple[Any, Any]]:
        pairs: list[tuple[Any, Any]] = []
        for f in dataclasses.fields(obj):
            if not f.repr:
                continue
            value = getattr(obj, f.name, None)
            if value is not None:
                pairs.append((f.name, value))
        return pairs

    def _pairs(self, pairs: list[tuple[Any, Any]], omitted: int, oid: int, depth: int) -> Any:
        if oid in self._path:
            self.truncated = True
            return self._text("<cycle>")
        if (pairs or omitted) and self.remaining < _CONTAINER_MIN:
            return self._summary("dict", len(pairs) + omitted)
        self._path.add(oid)
        try:
            return self._emit_dict(pairs, omitted, depth)
        finally:
            self._path.discard(oid)

    def _emit_dict(self, pairs: list[tuple[Any, Any]], omitted: int, depth: int) -> dict[str, Any]:
        self.remaining -= 2
        keyed = [(self._key(k), v) for k, v in pairs]
        allowances = _water_fill(
            [
                len(k)
                + (len(v) + 6 if type(v) is str else 12 if _is_scalar(v) else _estimate(v) + 4)
                for k, v in keyed
            ],
            self.remaining - _MARKER_RESERVE,
            _MIN_NODE + 4,
        )
        out: dict[str, Any] = {}
        for index, (key, value) in enumerate(keyed):
            key_cost = len(_encode_str(key)) + 1 + (1 if out else 0)
            if self.remaining - _MARKER_RESERVE < key_cost + _MIN_NODE:
                omitted += len(keyed) - index
                break
            self.remaining -= key_cost
            if is_sensitive_key(key):
                self.remaining -= len(REDACTED) + 2
                out[key] = REDACTED
                continue
            allowance = self.remaining - _MARKER_RESERVE
            if allowances is not None:
                allowance = min(allowance, max(allowances[index] - key_cost, _MIN_NODE))
            out[key] = self._walk_within(value, depth + 1, allowance)
        if omitted:
            self.truncated = True
            note = f"{omitted} keys omitted"
            self.remaining -= 5 + 1 + len(note) + 2 + (1 if out else 0)
            out["..."] = note
        return out

    def _key(self, key: Any) -> str:
        if type(key) is str:
            text = key
        elif isinstance(key, enum.Enum):
            text = str(key.value)
        elif isinstance(key, (str, int, float, bool)) or key is None:
            text = str(key)
        else:
            text = type_name(key)
        if len(text) > _MAX_KEY_CHARS:
            self.truncated = True
            text = text[:_MAX_KEY_CHARS]
        return _clean(text)

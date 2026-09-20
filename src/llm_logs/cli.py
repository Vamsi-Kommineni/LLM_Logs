"""``llm-logs``: look at what the SQLite sink has stored. Standard library only.

    llm-logs tail  --db logs/llm.db -n 20 [--follow] [--errors] [--model NAME]
    llm-logs stats --db logs/llm.db [--since 24h]
    llm-logs export --db logs/llm.db --format jsonl|csv [-o FILE]

Rows are read as plain dictionaries rather than through the ``Record`` model, so
this tool can still read a database written by a newer version of the library.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import sqlite3
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import IO, Any

DEFAULT_DB = "logs/llm.db"
_JSON_COLUMNS = ("finish_reasons", "params", "input", "output", "provider_extras", "metadata")
_DURATION = re.compile(r"^(\d+)([smhdw])$")
_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _connect(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise SystemExit(f"llm-logs: no database at {path} (use --db)")
    # Read-only: looking at logs must never be able to change them.
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    return connection


def _since(value: str | None) -> str | None:
    """Turn ``90m`` / ``24h`` / ``7d`` or an ISO timestamp into an ISO timestamp."""
    if not value:
        return None
    match = _DURATION.match(value)
    if match:
        delta = timedelta(seconds=int(match.group(1)) * _SECONDS[match.group(2)])
        return (datetime.now(UTC) - delta).isoformat()
    try:
        return datetime.fromisoformat(value).astimezone(UTC).isoformat()
    except ValueError:
        message = f"llm-logs: cannot read --since {value!r}; use 30m, 24h, 7d or a date"
        raise SystemExit(message) from None


def _where(args: argparse.Namespace) -> tuple[str, list[Any]]:
    clauses, values = [], []
    since = _since(getattr(args, "since", None))
    if since:
        clauses.append("start_time >= ?")
        values.append(since)
    if getattr(args, "model", None):
        clauses.append("COALESCE(model, response_model) = ?")
        values.append(args.model)
    if getattr(args, "errors", False):
        clauses.append("status = 'error'")
    if getattr(args, "trace", None):
        clauses.append("trace_id = ?")
        values.append(args.trace)
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), values


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for name in _JSON_COLUMNS:
        if isinstance(data.get(name), str):
            with contextlib.suppress(ValueError):  # leave a malformed cell as text
                data[name] = json.loads(data[name])
    for name in ("streamed", "truncated"):
        if data.get(name) is not None:
            data[name] = bool(data[name])
    return data


def _short(value: Any, width: int) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _preview(data: dict[str, Any], width: int) -> str:
    output = data.get("output")
    if isinstance(output, dict) and isinstance(output.get("content"), str):
        output = output["content"]
    if data.get("status") == "error":
        return _short(f"{data.get('error_type')}: {data.get('error_message')}", width)
    return _short(output, width)


def _format_line(data: dict[str, Any], width: int) -> str:
    tokens = ""
    if data.get("input_tokens") is not None or data.get("output_tokens") is not None:
        tokens = f"{data.get('input_tokens') or 0}→{data.get('output_tokens') or 0} tok"
    mark = "ERR" if data.get("status") == "error" else "ok "
    when = str(data.get("start_time", ""))[11:19]
    model = data.get("model") or data.get("response_model") or "-"
    head = (
        f"{when} {mark} {data.get('duration_ms') or 0:8.0f} ms  {tokens:>16}  "
        f"{_short(model, 28):<28} {_short(data.get('name'), 24):<24} "
    )
    return head + _preview(data, max(width - len(head), 20))


def _width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 160


def cmd_tail(args: argparse.Namespace, out: IO[str]) -> int:
    with contextlib.closing(_connect(args.db)) as connection:
        return _tail(connection, args, out)


def _tail(connection: sqlite3.Connection, args: argparse.Namespace, out: IO[str]) -> int:
    where, values = _where(args)
    rows = connection.execute(
        f"SELECT * FROM (SELECT * FROM records{where} ORDER BY id DESC LIMIT ?) ORDER BY id",
        [*values, args.lines],
    ).fetchall()
    last_id = 0
    for row in rows:
        last_id = row["id"]
        print(_format_line(_decode(row), _width()), file=out)
    while args.follow:  # pragma: no cover - interactive
        time.sleep(1.0)
        glue = " AND " if where else " WHERE "
        for row in connection.execute(
            f"SELECT * FROM records{where}{glue}id > ? ORDER BY id", [*values, last_id]
        ):
            last_id = row["id"]
            print(_format_line(_decode(row), _width()), file=out, flush=True)
    return 0


def _percentile(sorted_values: Sequence[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def collect_stats(connection: sqlite3.Connection, where: str, values: list[Any]) -> dict[str, Any]:
    glue = " AND " if where else " WHERE "
    calls = f"{where}{glue}kind = 'llm'"
    total, errors, first, last = connection.execute(
        f"SELECT COUNT(*), SUM(status = 'error'), MIN(start_time), MAX(start_time) "
        f"FROM records{calls}",
        values,
    ).fetchone()
    durations = [
        row[0]
        for row in connection.execute(
            f"SELECT duration_ms FROM records{calls} ORDER BY duration_ms", values
        )
    ]
    first_chunks = [
        row[0]
        for row in connection.execute(
            f"SELECT time_to_first_chunk_ms FROM records{calls} "
            "AND time_to_first_chunk_ms IS NOT NULL ORDER BY time_to_first_chunk_ms",
            values,
        )
    ]
    models = [
        dict(row)
        for row in connection.execute(
            "SELECT COALESCE(model, response_model, '(unknown)') AS model, COUNT(*) AS calls, "
            "SUM(status = 'error') AS errors, SUM(input_tokens) AS input_tokens, "
            "SUM(output_tokens) AS output_tokens, SUM(cache_read_input_tokens) AS cached_tokens, "
            "SUM(reasoning_output_tokens) AS reasoning_tokens, SUM(cost) AS cost "
            f"FROM records{calls} GROUP BY 1 ORDER BY calls DESC",
            values,
        )
    ]
    return {
        "calls": total or 0,
        "errors": errors or 0,
        "error_rate": (errors or 0) / total if total else 0.0,
        "first": first,
        "last": last,
        "latency_ms": {"p50": _percentile(durations, 0.5), "p95": _percentile(durations, 0.95)},
        "time_to_first_chunk_ms": {
            "p50": _percentile(first_chunks, 0.5),
            "p95": _percentile(first_chunks, 0.95),
        },
        "spans": connection.execute(
            f"SELECT COUNT(*) FROM records{where}{glue}kind = 'span'", values
        ).fetchone()[0],
        "truncated": connection.execute(
            f"SELECT COUNT(*) FROM records{calls} AND truncated = 1", values
        ).fetchone()[0],
        # A client that hangs up shows as CancelledError; a provider outage does not.
        "errors_by_type": dict(
            connection.execute(
                f"SELECT error_type, COUNT(*) FROM records{calls} AND status = 'error' "
                "GROUP BY 1 ORDER BY 2 DESC",
                values,
            ).fetchall()
        ),
        "models": models,
    }


def cmd_stats(args: argparse.Namespace, out: IO[str]) -> int:
    where, values = _where(args)
    with contextlib.closing(_connect(args.db)) as connection:
        stats = collect_stats(connection, where, values)
    if args.json:
        print(json.dumps(stats, indent=2), file=out)
        return 0
    print(f"LLM calls     {stats['calls']:,}   ({stats['spans']:,} other spans)", file=out)
    print(f"Errors        {stats['errors']:,}   ({stats['error_rate']:.1%})", file=out)
    if stats["errors_by_type"]:
        kinds = ", ".join(f"{name} {count:,}" for name, count in stats["errors_by_type"].items())
        print(f"              {kinds}", file=out)
    print(f"Period        {stats['first'] or '-'}  to  {stats['last'] or '-'}", file=out)
    latency, first_chunk = stats["latency_ms"], stats["time_to_first_chunk_ms"]
    print(f"Latency       p50 {_ms(latency['p50'])} ms   p95 {_ms(latency['p95'])} ms", file=out)
    print(
        f"First chunk   p50 {_ms(first_chunk['p50'])} ms   p95 {_ms(first_chunk['p95'])} ms",
        file=out,
    )
    print(f"Truncated     {stats['truncated']:,}", file=out)
    if stats["models"]:
        header = f"\n{'model':<36}{'calls':>8}{'errors':>8}{'input tok':>12}{'output tok':>12}"
        print(header + f"{'cached':>10}{'reasoning':>11}{'cost':>10}", file=out)
        for m in stats["models"]:
            cost = "-" if m["cost"] is None else f"{m['cost']:.4f}"
            print(
                f"{_short(m['model'], 35):<36}{m['calls']:>8,}{m['errors'] or 0:>8,}"
                f"{m['input_tokens'] or 0:>12,}{m['output_tokens'] or 0:>12,}"
                f"{m['cached_tokens'] or 0:>10,}{m['reasoning_tokens'] or 0:>11,}{cost:>10}",
                file=out,
            )
    return 0


def _rows(connection: sqlite3.Connection, where: str, values: list[Any]) -> Iterator[sqlite3.Row]:
    yield from connection.execute(f"SELECT * FROM records{where} ORDER BY id", values)


def cmd_export(args: argparse.Namespace, out: IO[str]) -> int:
    with contextlib.closing(_connect(args.db)) as connection:
        return _export(connection, args, out)


def _export(connection: sqlite3.Connection, args: argparse.Namespace, out: IO[str]) -> int:
    where, values = _where(args)
    target = open(args.output, "w", encoding="utf-8", newline="") if args.output else out  # noqa: SIM115
    try:
        count = 0
        if args.format == "jsonl":
            for row in _rows(connection, where, values):
                data = _decode(row)
                data.pop("id", None)
                target.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
        else:
            writer: Any = None
            for row in _rows(connection, where, values):
                data = dict(row)  # JSON columns stay as JSON text inside their CSV cell
                data.pop("id", None)
                if writer is None:
                    writer = csv.DictWriter(target, fieldnames=list(data))
                    writer.writeheader()
                writer.writerow(data)
                count += 1
    finally:
        if args.output:
            target.close()
    if args.output:
        print(f"exported {count} records to {args.output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-logs", description=(__doc__ or "").split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--db", default=DEFAULT_DB, help=f"SQLite file (default: {DEFAULT_DB})")
        sub.add_argument("--since", help="only records newer than this: 30m, 24h, 7d or a date")
        sub.add_argument("--model", help="only this model")
        sub.add_argument("--errors", action="store_true", help="only failed calls")
        sub.add_argument("--trace", help="only this trace ID")

    tail = commands.add_parser("tail", help="show the most recent records")
    common(tail)
    tail.add_argument("-n", "--lines", type=int, default=20)
    tail.add_argument("-f", "--follow", action="store_true", help="keep printing new records")
    tail.set_defaults(handler=cmd_tail)

    stats = commands.add_parser("stats", help="calls, error rate, latency and tokens by model")
    common(stats)
    stats.add_argument("--json", action="store_true", help="machine-readable output")
    stats.set_defaults(handler=cmd_stats)

    export = commands.add_parser("export", help="dump records, e.g. as a dataset for evals")
    common(export)
    export.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    export.add_argument("-o", "--output", help="write to this file instead of stdout")
    export.set_defaults(handler=cmd_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args, sys.stdout))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130
    except BrokenPipeError:  # pragma: no cover - e.g. piped into head
        return 0
    except sqlite3.Error as exc:
        print(f"llm-logs: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

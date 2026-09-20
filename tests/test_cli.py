from __future__ import annotations

import contextlib
import csv
import io
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import llm_logs as ll
from llm_logs import cli
from tests.test_sinks import make_record


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = tmp_path / "llm.db"
    sink = ll.SqliteSink(path)
    now = datetime.now(UTC)
    records = []
    for n in range(20):
        records.append(
            make_record(
                n,
                model="model-a" if n % 2 else "model-b",
                start_time=now - timedelta(minutes=n),
                end_time=now - timedelta(minutes=n),
                duration_ms=100.0 * (n + 1),
                time_to_first_chunk_ms=10.0 * (n + 1) if n % 2 else None,
                streamed=bool(n % 2),
                input_tokens=100,
                output_tokens=10,
                reasoning_output_tokens=4,
                cost=0.001,
                status="error" if n == 3 else "ok",
                error_type="RateLimitError" if n == 3 else None,
                error_message="slow down" if n == 3 else None,
                output={"role": "assistant", "content": f"answer {n}"},
            )
        )
    records.append(make_record(99, kind="span", name="pipeline", model=None, input_tokens=None))
    sink.write_batch(records)
    sink.close()
    return str(path)


def run(*argv: str) -> str:
    out = io.StringIO()
    args = cli.build_parser().parse_args(argv)
    assert args.handler(args, out) == 0
    return out.getvalue()


def test_tail_shows_the_latest_records_oldest_first(db: str) -> None:
    lines = run("tail", "--db", db, "-n", "5").splitlines()
    assert len(lines) == 5
    assert "answer 19" in lines[-2] and "pipeline" in lines[-1], "in the order they were written"
    assert all(" ok " in line or "ERR" in line for line in lines)
    assert "100→10 tok" in lines[0]


def test_tail_filters(db: str) -> None:
    (line,) = run("tail", "--db", db, "--errors").splitlines()
    assert "ERR" in line and "RateLimitError: slow down" in line
    only_a = run("tail", "--db", db, "--model", "model-a", "-n", "100").splitlines()
    assert len(only_a) == 10 and all("model-a" in line for line in only_a)
    assert len(run("tail", "--db", db, "--since", "5m", "-n", "100").splitlines()) <= 7


def test_stats_numbers(db: str) -> None:
    stats = json.loads(run("stats", "--db", db, "--json"))
    assert (stats["calls"], stats["spans"], stats["errors"]) == (20, 1, 1)
    assert stats["error_rate"] == pytest.approx(0.05)
    assert stats["errors_by_type"] == {"RateLimitError": 1}
    assert stats["latency_ms"]["p50"] == pytest.approx(1050.0)
    assert stats["latency_ms"]["p95"] == pytest.approx(1905.0)
    assert stats["time_to_first_chunk_ms"]["p50"] == pytest.approx(110.0)
    by_model = {m["model"]: m for m in stats["models"]}
    assert by_model["model-a"]["calls"] == 10 and by_model["model-a"]["errors"] == 1
    assert by_model["model-b"]["input_tokens"] == 1000
    assert by_model["model-b"]["reasoning_tokens"] == 40
    assert by_model["model-a"]["cost"] == pytest.approx(0.01)


def test_stats_table_is_readable(db: str) -> None:
    text = run("stats", "--db", db)
    assert "LLM calls     20" in text and "(5.0%)" in text and "RateLimitError 1" in text
    assert "p50 1,050 ms" in text and "model-a" in text and "reasoning" in text


def test_export_jsonl_and_csv(db: str, tmp_path: Path) -> None:
    lines = run("export", "--db", db, "--format", "jsonl", "--model", "model-b").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 10 and "id" not in rows[0]
    assert rows[0]["input"] == {"prompt": rows[0]["input"]["prompt"]}, "JSON columns are decoded"
    assert isinstance(rows[0]["streamed"], bool)

    target = tmp_path / "out.csv"
    run("export", "--db", db, "--format", "csv", "-o", str(target))
    with target.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 21 and table[0]["trace_id"]
    assert json.loads(table[0]["output"])["content"].startswith("answer")


def test_the_cli_opens_the_database_read_only(db: str) -> None:
    with (
        contextlib.closing(cli._connect(db)) as connection,
        pytest.raises(sqlite3.OperationalError),
    ):
        connection.execute("DELETE FROM records")


def test_helpful_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="no database"):
        cli.main(["stats", "--db", str(tmp_path / "missing.db")])
    (tmp_path / "bad.db").write_text("not sqlite")
    assert cli.main(["stats", "--db", str(tmp_path / "bad.db")]) == 1
    assert "llm-logs:" in capsys.readouterr().err


def test_it_reads_rows_written_by_a_newer_schema(db: str) -> None:
    connection = sqlite3.connect(db)
    connection.execute("ALTER TABLE records ADD COLUMN field_from_the_future TEXT")
    connection.commit()
    connection.close()
    assert json.loads(run("stats", "--db", db, "--json"))["calls"] == 20
    assert "field_from_the_future" in json.loads(run("export", "--db", db).splitlines()[0])

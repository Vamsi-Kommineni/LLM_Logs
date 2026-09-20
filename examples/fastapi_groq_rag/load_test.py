"""Soak the demo service and check what the logger did under load.

    # terminal 1
    LLM_LOGS_DIR=/tmp/rag-logs uvicorn examples.fastapi_groq_rag.app:app --workers 4
    # terminal 2
    python -m examples.fastapi_groq_rag.load_test --requests 4000 --log-dir /tmp/rag-logs

Sends concurrent streaming requests. Some clients hang up mid-stream, the fake
provider fails now and then, and every question carries an e-mail address and
something shaped like an API key, so redaction is checked on real traffic.
Afterwards the log files are read back and verified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

import httpx

EMAIL = "jane.doe@example.com"
FAKE_KEY = "gsk_" + "a1B2" * 12
TOPICS = ("refunds", "shipping", "support")


async def one_request(client: httpx.AsyncClient, number: int, hang_up_rate: float) -> str:
    question = (
        f"Request {number}: what about {random.choice(TOPICS)}? "
        f"Mail me at {EMAIL}, my key is {FAKE_KEY}."
    )
    payload = {"question": question, "session_id": f"s{number % 50}", "user_id": f"u{number % 7}"}
    try:
        async with client.stream("POST", "/chat", json=payload) as response:
            if random.random() < hang_up_rate:
                async for _ in response.aiter_bytes():
                    return "hung_up"  # leave after the first bytes, like a closed browser tab
            await response.aread()
            return "ok" if response.status_code == 200 else f"http_{response.status_code}"
    except httpx.HTTPError as exc:
        return type(exc).__name__


async def run(args: argparse.Namespace) -> Counter[str]:
    limits = httpx.Limits(max_connections=args.concurrency)
    outcomes: Counter[str] = Counter()
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, limits=limits, timeout=60) as client:

        async def guarded(number: int) -> None:
            async with semaphore:
                outcomes[await one_request(client, number, args.hang_up_rate)] += 1

        await asyncio.gather(*(guarded(n) for n in range(args.requests)))
        seen: dict[int, dict[str, int]] = {}
        for _ in range(60):
            # A kept-alive connection stays with one worker, so ask on fresh ones.
            data = (await client.get("/stats", headers={"Connection": "close"})).json()
            seen[data["pid"]] = data
    print(f"\nworkers seen: {len(seen)}")
    for pid, data in sorted(seen.items()):
        print(f"  worker {pid}: " + ", ".join(f"{k}={v}" for k, v in data.items() if k != "pid"))
    return outcomes


def verify(log_dir: Path, requests: int) -> bool:
    files = sorted(log_dir.glob("*.jsonl"))
    records = [json.loads(line) for path in files for line in path.read_text().splitlines()]
    raw = b"".join(path.read_bytes() for path in files)
    calls = [r for r in records if r["kind"] == "llm"]
    pipelines = [r for r in records if r["name"] == "rag_pipeline"]
    by_span = {r["span_id"]: r for r in records}

    print(f"\nlog files: {len(files)}   records: {len(records):,}   bytes: {len(raw):,}")
    print("llm calls by outcome:", dict(Counter(r["stream_outcome"] for r in calls)))
    print("llm calls by status: ", dict(Counter(r["status"] for r in calls)))
    print("llm errors by type:  ", dict(Counter(r["error_type"] for r in calls if r["error_type"])))
    print("largest record:", max((len(json.dumps(r)) for r in records), default=0), "chars")

    checks = {
        "every line is valid JSON": True,  # json.loads above would have raised
        "one pipeline span per request": len(pipelines) == requests,
        "one llm call per request": len(calls) == requests,
        "no duplicate span ids": len(by_span) == len(records),
        "every llm call has its pipeline as parent": all(
            by_span.get(r["parent_span_id"], {}).get("name") == "rag_pipeline" for r in calls
        ),
        "e-mail address never reached the disk": EMAIL.encode() not in raw,
        "key-shaped string never reached the disk": FAKE_KEY.encode() not in raw,
        "redaction markers are present": b"[EMAIL]" in raw and b"[REDACTED]" in raw,
    }
    database = log_dir / "llm.db"
    if database.exists():
        with closing(sqlite3.connect(database)) as connection:
            rows = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        checks["sqlite has every record the files have"] = rows == len(records)
        for suffix in ("", "-wal"):
            path = Path(str(database) + suffix)
            if path.exists():
                blob = path.read_bytes()
                checks[f"no secrets in llm.db{suffix}"] = (
                    EMAIL.encode() not in blob and FAKE_KEY.encode() not in blob
                )
    print()
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--hang-up-rate", type=float, default=0.05)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    outcomes = asyncio.run(run(args))
    elapsed = time.perf_counter() - started
    print(f"\n{args.requests:,} requests in {elapsed:.1f} s ({args.requests / elapsed:,.0f}/s)")
    print("client outcomes:", dict(outcomes))
    time.sleep(2.5)  # let the writers' flush interval pass
    return 0 if verify(args.log_dir, args.requests) else 1


if __name__ == "__main__":
    raise SystemExit(main())

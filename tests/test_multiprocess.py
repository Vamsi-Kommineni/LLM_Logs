"""Process safety: several workers, and a fork after the writer thread has started."""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import llm_logs as ll
from tests.conftest import query

PER_PROCESS = 200
PROCESSES = 4

pytestmark = pytest.mark.slow


def worker(directory: str, number: int) -> None:
    ll.configure(
        sinks=[ll.JsonlSink(directory), ll.SqliteSink(Path(directory) / "llm.db")],
        batch_size=20,
        flush_interval=0.01,
    )

    @ll.trace
    def ask(prompt: str) -> str:
        return f"answer from {number}"

    for call in range(PER_PROCESS):
        ask(f"worker {number} call {call}")
    assert ll.shutdown(30)
    stats = ll.stats()
    assert (stats.dropped, stats.failed, stats.written) == (0, 0, PER_PROCESS), stats


def start_methods() -> list[str]:
    available = multiprocessing.get_all_start_methods()
    return [method for method in ("fork", "spawn") if method in available]


@pytest.mark.parametrize("method", start_methods())
def test_four_processes_lose_nothing_and_never_interleave(tmp_path: Path, method: str) -> None:
    # The start method is explicit: since Python 3.14 the default on Linux is
    # forkserver, and both fork and spawn have to work.
    context = multiprocessing.get_context(method)
    processes = [
        context.Process(target=worker, args=(str(tmp_path), number)) for number in range(PROCESSES)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(120)
        assert process.exitcode == 0

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == PROCESSES, "one file per process"
    prompts: list[str] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)  # every line parses: no interleaved writes
            prompts.append(record["input"]["prompt"])
    expected = {f"worker {n} call {c}" for n in range(PROCESSES) for c in range(PER_PROCESS)}
    assert len(prompts) == PROCESSES * PER_PROCESS and set(prompts) == expected

    assert query(tmp_path / "llm.db", "SELECT COUNT(*) FROM records") == [
        (PROCESSES * PER_PROCESS,)
    ]


FORK_SCRIPT = """
    import json, os, sys
    import llm_logs as ll

    directory, before = sys.argv[1], int(sys.argv[2])
    ll.configure(
        sinks=[ll.JsonlSink(directory), ll.SqliteSink(os.path.join(directory, "llm.db"))],
        flush_interval=0.01, batch_size=7,
    )

    @ll.trace
    def ask(prompt):
        return "ok"

    for n in range(before):
        ask(f"parent-before-{n}")       # starts the writer thread in the parent

    pid = os.fork()
    if pid == 0:
        for n in range(50):
            ask(f"child-{n}")
        ok = ll.shutdown(20)
        stats = ll.stats()
        clean = (stats.enqueued, stats.written, stats.dropped, stats.failed) == (50, 50, 0, 0)
        os._exit(0 if ok and clean else 3)

    for n in range(50):
        ask(f"parent-after-{n}")
    _, status = os.waitpid(pid, 0)
    ok = ll.shutdown(20) and ll.stats().failed == 0 and ll.stats().dropped == 0
    print(json.dumps({"child_exit": os.waitstatus_to_exitcode(status), "parent_ok": ok,
                      "parent_pid": os.getpid(), "child_pid": pid}))
"""


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs os.fork")
@pytest.mark.parametrize("before", [40, 0], ids=["writer-running", "writer-not-started"])
def test_a_forked_child_gets_its_own_writer_and_file(tmp_path: Path, before: int) -> None:
    """The gunicorn case: configure in the parent, fork, keep tracing on both sides."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(FORK_SCRIPT), str(tmp_path), str(before)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["child_exit"] == 0 and outcome["parent_ok"]

    by_pid: dict[str, list[str]] = {}
    for path in tmp_path.glob("*.jsonl"):
        pid = path.name.split("-p")[1].split(".")[0]
        lines = path.read_text(encoding="utf-8").splitlines()
        by_pid.setdefault(pid, []).extend(json.loads(line)["input"]["prompt"] for line in lines)

    parent = sorted(by_pid[str(outcome["parent_pid"])])
    child = sorted(by_pid[str(outcome["child_pid"])])
    expected_parent = [f"parent-before-{n}" for n in range(before)] + [
        f"parent-after-{n}" for n in range(50)
    ]
    assert parent == sorted(expected_parent), "nothing lost, nothing written twice"
    assert child == sorted(f"child-{n}" for n in range(50))

    assert query(tmp_path / "llm.db", "SELECT COUNT(*) FROM records") == [(before + 100,)]

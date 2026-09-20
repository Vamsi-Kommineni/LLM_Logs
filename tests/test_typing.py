"""The decorator must be invisible to type checkers: same parameters, same return type."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

GOOD = """
    import llm_logs as ll

    @ll.trace
    def ask(prompt: str, temperature: float = 0.0) -> str:
        return prompt

    @ll.trace(provider="groq", ignore_args=["client"])
    async def ask_async(client: object, prompt: str) -> int:
        return 1

    async def main() -> None:
        text: str = ask("hi", temperature=0.5)
        number: int = await ask_async(object(), "hi")
        with ll.span("s") as s:
            s.metadata["k"] = 1
"""

BAD = """
    import llm_logs as ll

    @ll.trace
    def ask(prompt: str) -> str:
        return prompt

    @ll.trace(name="other")
    def other(count: int) -> int:
        return count

    ask(42)
    other("three")
    wrong: int = ask("hi")
"""


def run_mypy(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "snippet.py"
    path.write_text(textwrap.dedent(source))
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-incremental", str(path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.slow
def test_mypy_accepts_correct_use_of_decorated_functions(tmp_path: Path) -> None:
    result = run_mypy(tmp_path, GOOD)
    assert result.returncode == 0, result.stdout


@pytest.mark.slow
def test_mypy_rejects_wrong_argument_and_return_types(tmp_path: Path) -> None:
    result = run_mypy(tmp_path, BAD)
    assert result.returncode != 0
    assert result.stdout.count("error:") == 3, result.stdout
    assert 'Argument 1 to "ask" has incompatible type "int"' in result.stdout
    assert 'Argument 1 to "other" has incompatible type "str"' in result.stdout

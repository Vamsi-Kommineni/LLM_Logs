"""The published wheel: what it contains and what installing it drags in."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UV = shutil.which("uv")

# pydantic and the packages pydantic itself depends on. Nothing else.
ALLOWED = {
    "llm-logs",
    "pydantic",
    "pydantic-core",
    "annotated-types",
    "typing-extensions",
    "typing-inspection",
}

pytestmark = [pytest.mark.slow, pytest.mark.skipif(UV is None, reason="uv is not installed")]


def run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, cwd=cwd)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    assert UV is not None
    dist = tmp_path_factory.mktemp("dist")
    run(UV, "build", "--wheel", "--out-dir", str(dist), str(ROOT))
    (path,) = dist.glob("*.whl")
    return path


def test_the_wheel_is_typed_and_contains_no_strays(wheel: Path) -> None:
    names = zipfile.ZipFile(wheel).namelist()
    assert "llm_logs/py.typed" in names
    assert any(name.endswith("licenses/LICENSE") for name in names)
    top_level = {name.split("/")[0] for name in names}
    assert top_level == {"llm_logs", next(n for n in top_level if n.endswith(".dist-info"))}
    assert not [n for n in names if "legacy" in n or "tests" in n or n.endswith(".env")]


def test_a_clean_install_brings_in_pydantic_only(wheel: Path, tmp_path: Path) -> None:
    assert UV is not None
    venv = tmp_path / "venv"
    run(UV, "venv", "--quiet", "--python", sys.executable, str(venv))
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    run(UV, "pip", "install", "--quiet", "--python", str(python), str(wheel))

    installed = run(
        str(python),
        "-c",
        "import importlib.metadata as m;"
        "names = (d.metadata['Name'].lower().replace('_', '-') for d in m.distributions());"
        "print('\\n'.join(sorted(names)))",
    ).split()
    assert set(installed) <= ALLOWED, f"unexpected dependencies: {set(installed) - ALLOWED}"

    out = run(
        str(python),
        "-c",
        "import llm_logs as ll; s = ll.InMemorySink(); ll.configure(sinks=[s]);"
        "ll.trace(lambda: 'ok')(); assert ll.shutdown(); print(len(s.records), ll.__version__)",
        cwd=tmp_path,
    )
    assert out.split()[0] == "1"

    script = venv / ("Scripts/llm-logs.exe" if sys.platform == "win32" else "bin/llm-logs")
    assert script.exists(), "the console script is installed"

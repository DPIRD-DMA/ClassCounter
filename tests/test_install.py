"""Pytest wrapper that runs each install-isolation script via ``uv run --script``.

Each script declares its own inline dependencies and gets an isolated
environment, so we can verify that partial installs (e.g. ``classcounter[geo]``)
work correctly without the other optional extras.

Run only these tests::

    uv run pytest -m install

Skip them (they're slower due to env creation)::

    uv run pytest -m "not install"
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

INSTALL_DIR = Path(__file__).parent / "install"

SCRIPTS = sorted(INSTALL_DIR.glob("test_*.py"))

pytestmark = pytest.mark.install


@pytest.mark.parametrize(
    "script",
    SCRIPTS,
    ids=[s.stem for s in SCRIPTS],
)
def test_install_script(script: Path):
    result = subprocess.run(
        ["uv", "run", "--script", str(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        # Show both stdout (pytest output) and stderr (uv/build output)
        msg = f"{script.name} failed (exit {result.returncode})\n"
        if result.stdout:
            msg += f"--- stdout ---\n{result.stdout}\n"
        if result.stderr:
            msg += f"--- stderr ---\n{result.stderr}\n"
        pytest.fail(msg)

"""Run every script under examples/ as an executable form of the docs.

A public-API rename or signature change fails here instead of leaving a stale
guide behind. Each script prints only and finishes in seconds.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("0*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(script):
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"{script.name} failed:\n{proc.stdout}\n{proc.stderr}"
    )

#!/usr/bin/env python3

import glob
import subprocess
import sys

import pytest

EXAMPLES = sorted(glob.glob("examples/*.py"))


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_runs(example):
    """Test that examples run without errors."""
    result = subprocess.run([sys.executable, str(example)], text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    for ex in EXAMPLES:
        test_example_runs(ex)

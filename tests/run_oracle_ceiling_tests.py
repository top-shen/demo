"""Run Oracle ceiling tests without pytest, PyTorch, SciPy, or PyYAML."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import test_oracle_ceiling


def main():
    tests = [
        function
        for name, function in inspect.getmembers(
            test_oracle_ceiling, inspect.isfunction
        )
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} Oracle ceiling tests passed")


if __name__ == "__main__":
    main()

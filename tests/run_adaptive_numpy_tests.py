"""Run dependency-light adaptive-controller tests without pytest/PyTorch."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_adaptive_numpy


def main():
    tests = [
        function
        for name, function in inspect.getmembers(test_adaptive_numpy, inspect.isfunction)
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} adaptive NumPy tests passed")


if __name__ == "__main__":
    main()

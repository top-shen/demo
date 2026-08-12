"""Run reference-dependence tests without requiring pytest."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_reference_dependence


def main():
    tests = [
        function
        for name, function in inspect.getmembers(test_reference_dependence, inspect.isfunction)
        if name.startswith("test_")
    ]
    for test in tests:
        if "tmp_path" in inspect.signature(test).parameters:
            with tempfile.TemporaryDirectory(prefix="reference_dependence_test_") as folder:
                test(Path(folder))
        else:
            test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} reference-dependence tests passed")


if __name__ == "__main__":
    main()

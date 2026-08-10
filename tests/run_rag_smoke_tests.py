"""Run the RI-VerbalTS unit tests without requiring pytest."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_retrieval
import test_adaptive_controller


def main():
    tests = []
    for module in (test_retrieval, test_adaptive_controller):
        tests.extend(
            function
            for name, function in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_")
        )
    for test in tests:
        parameters = inspect.signature(test).parameters
        if "tmp_path" in parameters:
            with tempfile.TemporaryDirectory(prefix="ri_verbalts_test_") as folder:
                test(Path(folder))
        else:
            test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} RI-VerbalTS smoke tests passed")


if __name__ == "__main__":
    main()

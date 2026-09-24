"""Run an untouched release test suite against reference, native or compatibility APIs.

Example:
    python scripts/check_release.py ../reference-release --backend compat -q

Run each backend in its own Python process. Import aliases below belong only to
this test harness; importing compatibility never switches a user's active API.
This is a diagnostic: upstream tests for removed class identities and leaves are
expected to fail with the constructor-based compatibility layer.
"""

import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument(
    "--backend",
    choices=("reference", "native", "compat"),
    default="compat",
)
options, pytest_args = parser.parse_known_args()
reference = options.reference.resolve()
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((reference if options.backend == "reference" else repo) / "src"))

# Preload JAX before coverage instrumentation on Python 3.14.
import jax  # noqa: F401 -- initialise MLIR before coverage
import dLux

print(f"dLux imported from {dLux.__file__}; backend={options.backend}", flush=True)
if options.backend == "compat":
    from dLux import compat
    from dLux.utils import compat as utilities

    # Unmodified upstream tests import dLux, dLux.layers and dLux.utils. Route
    # those imports to the explicitly retained API in this isolated process.
    for name, module in list(sys.modules.items()):
        if name.startswith("dLux.compat."):
            sys.modules[name.replace("dLux.compat", "dLux", 1)] = module
        elif name.startswith("dLux.utils.compat."):
            sys.modules[name.replace("dLux.utils.compat", "dLux.utils", 1)] = module
    sys.modules["dLux"] = compat
    sys.modules["dLux.utils"] = utilities

import pytest

os.chdir(reference)
# This upstream test runs pip install and removes build directories. Package
# installation is a separate environment check, never part of API comparison.
arguments = [
    str(reference / "tests"),
    "--import-mode=importlib",
    f"--ignore={reference / 'tests/test_installation.py'}",
    *pytest_args,
]
raise SystemExit(pytest.main(arguments))

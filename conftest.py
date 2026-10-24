"""Makes ``fbpredict`` importable when pytest is run from the project root.

``pyproject.toml`` already sets ``pythonpath = ["."]``; this is the belt-and-
braces version so the tests also work with an older pytest.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

"""Shared paths for the capacity scripts — run from anywhere."""
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("CAPACITY_OUT", Path(tempfile.gettempdir()) / "afterglow-capacity"))
OUT.mkdir(parents=True, exist_ok=True)
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

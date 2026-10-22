"""Compatibility import shim; canonical retrieval helpers live under src/rag."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.hybrid import *  # noqa: F401,F403
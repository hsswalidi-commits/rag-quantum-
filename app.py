"""Legacy command-line entrypoint. Serve the API with uvicorn main:app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.core import main

if __name__ == "__main__":
    main()
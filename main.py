"""Thin wrapper so `python main.py` still works without installation."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from deep_research.cli import main  # noqa: E402

if __name__ == "__main__":
    main()

import sys
from pathlib import Path

# Allow tests to import the package without installation
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

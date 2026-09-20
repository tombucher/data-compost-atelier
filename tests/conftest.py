import sys
from pathlib import Path

# Rend `atelier` importable sans installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

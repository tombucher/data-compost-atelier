#!/usr/bin/env python3
"""Point d'entrée de l'atelier.

    venv/bin/python demarrer.py ~/Images/pavots

Lancer ce fichier place la racine du dépôt sur le chemin d'import, ce qui
évite d'avoir à installer le paquet ou à régler PYTHONPATH.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atelier.server import main  # noqa: E402

if __name__ == "__main__":
    main()

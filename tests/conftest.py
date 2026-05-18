import sys
from pathlib import Path

# Permet `from main import load_data` depuis le dossier tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

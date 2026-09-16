import sys
from pathlib import Path

# The experiment scripts import each other as plain modules, so make the
# assignment folder importable from the tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

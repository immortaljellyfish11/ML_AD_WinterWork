from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architectures.common.runner import run_design


if __name__ == "__main__":
    run_design(Path(__file__).resolve().parent)


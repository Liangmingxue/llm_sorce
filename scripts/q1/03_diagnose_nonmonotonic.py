from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.diagnose_nonmonotonic import diagnose_nonmonotonic


if __name__ == "__main__":
    diagnose_nonmonotonic(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        standardized_dir=ROOT / "artifacts" / "q1" / "standardized",
        output_dir=ROOT / "artifacts" / "q1" / "diagnostics",
    )

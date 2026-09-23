from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.diagnose_unique_words_length import diagnose_unique_words_length


if __name__ == "__main__":
    diagnose_unique_words_length(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        standardized_dir=ROOT / "artifacts" / "q1" / "standardized",
        output_dir=ROOT / "artifacts" / "q1" / "diagnostics",
    )

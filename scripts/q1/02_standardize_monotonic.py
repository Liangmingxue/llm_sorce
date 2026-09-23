from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.standardize_monotonic import standardize_monotonic


if __name__ == "__main__":
    standardize_monotonic(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        config_path=ROOT / "configs" / "q1" / "quality_direction.yaml",
        output_dir=ROOT / "artifacts" / "q1" / "standardized",
    )

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.compute_indicator_weights import compute_weights


if __name__ == "__main__":
    compute_weights(
        transformed_a1=ROOT / "artifacts" / "q1" / "transformed" / "a1_quality_22_scores.csv.gz",
        config_path=ROOT / "configs" / "q1" / "quality_transform.yaml",
        output_dir=ROOT / "artifacts" / "q1" / "model",
    )

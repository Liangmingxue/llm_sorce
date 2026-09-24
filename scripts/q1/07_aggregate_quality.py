from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.aggregate_quality import run_quality_aggregation


if __name__ == "__main__":
    run_quality_aggregation(
        transformed_dir=ROOT / "artifacts" / "q1" / "transformed",
        weights_path=(
            ROOT
            / "artifacts"
            / "q1"
            / "model"
            / "hierarchical_group_critic_weights.csv"
        ),
        config_path=ROOT / "configs" / "q1" / "quality_aggregation.yaml",
        output_dir=ROOT / "artifacts" / "q1" / "quality",
    )

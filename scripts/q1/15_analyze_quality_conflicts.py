from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.analyze_quality_conflicts import run_conflict_analysis


if __name__ == "__main__":
    run_conflict_analysis(
        transformed_dir=ROOT / "artifacts" / "q1" / "transformed",
        quality_dir=ROOT / "artifacts" / "q1" / "quality",
        weights_path=ROOT / "artifacts" / "q1" / "model" / "hierarchical_group_critic_weights.csv",
        output_dir=ROOT / "artifacts" / "q1" / "conflict_analysis",
    )

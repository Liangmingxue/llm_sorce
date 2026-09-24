from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.analyze_ilr_epsilon import run_ilr_epsilon_sensitivity


if __name__ == "__main__":
    run_ilr_epsilon_sensitivity(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        regmix_cv_dir=ROOT / "artifacts" / "q1" / "regmix_cv",
        config_path=ROOT / "configs" / "q1" / "ilr_epsilon_sensitivity.yaml",
        output_dir=ROOT / "artifacts" / "q1" / "ilr_epsilon_sensitivity",
    )

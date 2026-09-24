from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.fit_regmix_cv import run_regmix_nested_cv


if __name__ == "__main__":
    run_regmix_nested_cv(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        config_path=ROOT / "configs" / "q1" / "regmix_nested_cv.yaml",
        output_dir=ROOT / "artifacts" / "q1" / "regmix_cv",
    )

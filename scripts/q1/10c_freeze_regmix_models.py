from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.freeze_regmix_models import run_freeze_regmix_models


if __name__ == "__main__":
    run_freeze_regmix_models(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        regmix_cv_dir=ROOT / "artifacts" / "q1" / "regmix_cv",
        config_path=ROOT / "configs" / "q1" / "regmix_final.yaml",
        output_dir=ROOT / "artifacts" / "q1" / "regmix_final",
    )

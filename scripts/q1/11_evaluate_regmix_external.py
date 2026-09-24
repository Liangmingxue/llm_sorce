from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.evaluate_regmix_external import run_external_regmix_test


if __name__ == "__main__":
    run_external_regmix_test(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        frozen_dir=ROOT / "artifacts" / "q1" / "regmix_final",
        output_dir=ROOT / "artifacts" / "q1" / "regmix_external_test",
    )

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.analyze_regmix_extrapolation import run_regmix_extrapolation


if __name__ == "__main__":
    run_regmix_extrapolation(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        frozen_dir=ROOT / "artifacts" / "q1" / "regmix_final",
        external_test_dir=ROOT / "artifacts" / "q1" / "regmix_external_test",
        output_dir=ROOT / "artifacts" / "q1" / "regmix_extrapolation",
    )

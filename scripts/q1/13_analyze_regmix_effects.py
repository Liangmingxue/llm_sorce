from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.analyze_regmix_effects import run_regmix_effect_analysis


if __name__ == "__main__":
    run_regmix_effect_analysis(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        frozen_dir=ROOT / "artifacts" / "q1" / "regmix_final",
        external_test_dir=ROOT / "artifacts" / "q1" / "regmix_external_test",
        extrapolation_dir=ROOT / "artifacts" / "q1" / "regmix_extrapolation",
        output_dir=ROOT / "artifacts" / "q1" / "regmix_effects",
    )

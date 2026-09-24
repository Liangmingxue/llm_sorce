from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.optimize_regmix_support_aware import run_regmix_optimization


if __name__ == "__main__":
    run_regmix_optimization(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        frozen_dir=ROOT / "artifacts" / "q1" / "regmix_final",
        effects_dir=ROOT / "artifacts" / "q1" / "regmix_effects",
        output_dir=ROOT / "artifacts" / "q1" / "regmix_optimization",
    )

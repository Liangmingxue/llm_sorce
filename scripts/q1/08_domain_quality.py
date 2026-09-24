from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.analyze_domain_quality import run_domain_quality_analysis


if __name__ == "__main__":
    run_domain_quality_analysis(
        quality_dir=ROOT / "artifacts" / "q1" / "quality",
        config_path=ROOT / "configs" / "q1" / "domain_quality.yaml",
        output_dir=ROOT / "artifacts" / "q1" / "domain_quality",
    )

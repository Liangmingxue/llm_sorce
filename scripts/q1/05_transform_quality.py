from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.transform_quality import run_quality_transform


if __name__ == "__main__":
    run_quality_transform(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        config_path=ROOT / "configs" / "q1" / "quality_transform.yaml",
        output_dir=ROOT / "artifacts" / "q1" / "transformed",
    )

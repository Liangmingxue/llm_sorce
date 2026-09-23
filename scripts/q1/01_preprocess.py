from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.preprocess import run_preprocessing


if __name__ == "__main__":
    run_preprocessing(
        data_root=ROOT / "real_attachments",
        output_dir=ROOT / "artifacts" / "q1" / "preprocessed",
    )

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.audit_regmix import run_regmix_audit


if __name__ == "__main__":
    run_regmix_audit(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        config_path=ROOT / "configs" / "q1" / "regmix_audit.yaml",
        output_dir=ROOT / "artifacts" / "q1" / "regmix_audit",
    )

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml


def _load_direction_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mid_ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Map values to the empirical CDF of the reference sample using mid-ranks.
    Missing/non-finite values remain NaN.
    """
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    if ref.size == 0:
        raise ValueError("Reference distribution has no finite values.")

    out = np.full(len(values), np.nan, dtype=float)
    vals = np.asarray(values, dtype=float)
    mask = np.isfinite(vals)
    v = vals[mask]

    left = np.searchsorted(ref, v, side="left")
    right = np.searchsorted(ref, v, side="right")
    out[mask] = (left + right) / (2.0 * ref.size)
    return out


def _transform_field(
    reference: pd.Series,
    target: pd.Series,
    direction: str,
) -> np.ndarray:
    u = _mid_ecdf(
        pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(target, errors="coerce").to_numpy(dtype=float),
    )
    if direction == "higher_is_better":
        return u
    if direction == "lower_is_better":
        return 1.0 - u
    raise ValueError(f"Unsupported monotonic direction: {direction}")


def standardize_monotonic(
    preprocessed_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, pd.DataFrame] = {
        "A1": pd.read_csv(preprocessed_dir / "a1_quality_clean.csv.gz"),
        "A2": pd.read_csv(preprocessed_dir / "a2_quality_clean.csv.gz"),
        "A3": pd.read_csv(preprocessed_dir / "a3_quality_clean.csv.gz"),
    }

    cfg = _load_direction_config(config_path)
    monotonic_rules: dict[str, str] = {}

    for field, meta in cfg.get("positive", {}).items():
        monotonic_rules[field] = meta["rule"]
    for field, meta in cfg.get("negative", {}).items():
        monotonic_rules[field] = meta["rule"]

    if not monotonic_rules:
        raise ValueError("No positive/negative fields found in direction config.")

    # Use A1 as the common reference scale:
    # A1 covers seven source domains, while A2/A3 are domain-specific expansions.
    reference = frames["A1"]

    audit_rows = []

    for field, direction in monotonic_rules.items():
        if field not in reference.columns:
            raise ValueError(f"Configured field not found in A1: {field}")

        for dataset, df in frames.items():
            if field not in df.columns:
                raise ValueError(f"Configured field not found in {dataset}: {field}")

            score_col = f"u_{field}"
            df[score_col] = _transform_field(reference[field], df[field], direction)

            s = pd.to_numeric(df[score_col], errors="coerce")
            audit_rows.append(
                {
                    "dataset": dataset,
                    "field": field,
                    "direction": direction,
                    "rows": len(df),
                    "missing": int(s.isna().sum()),
                    "min": float(s.min()) if s.notna().any() else np.nan,
                    "median": float(s.median()) if s.notna().any() else np.nan,
                    "max": float(s.max()) if s.notna().any() else np.nan,
                }
            )

    keep_base = ["id", "sub_path", "source_dataset", "source_domain"]
    score_cols = [f"u_{field}" for field in monotonic_rules]

    for dataset, df in frames.items():
        out = df[keep_base + score_cols].copy()
        out.to_csv(
            output_dir / f"{dataset.lower()}_monotonic_scores.csv.gz",
            index=False,
            compression="gzip",
        )

    pd.DataFrame(audit_rows).to_csv(
        output_dir / "monotonic_transform_audit.csv",
        index=False,
    )

    rule_rows = [
        {"field": field, "direction": direction, "reference_dataset": "A1"}
        for field, direction in monotonic_rules.items()
    ]
    pd.DataFrame(rule_rows).to_csv(
        output_dir / "monotonic_transform_rules.csv",
        index=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardize monotonic Q1 quality indicators to higher-is-better scores."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("artifacts/q1/preprocessed"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1/quality_direction.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/standardized"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    standardize_monotonic(
        preprocessed_dir=args.preprocessed_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )

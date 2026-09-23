from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


NON_MONOTONIC_FIELDS = [
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_lines_uppercase_letter_fraction",
    "rps_doc_mean_word_length",
]

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _finite_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.where(np.isfinite(s), np.nan)


def _distribution_rows(
    dataset: str,
    df: pd.DataFrame,
    fields: Iterable[str],
) -> list[dict]:
    rows: list[dict] = []
    for field in fields:
        s = _finite_numeric(df[field])
        finite = s.dropna()
        q = finite.quantile(QUANTILES)
        row = {
            "dataset": dataset,
            "field": field,
            "rows": int(len(df)),
            "missing": int(s.isna().sum()),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)),
        }
        for level, value in q.items():
            row[f"q{int(round(level * 100)):02d}"] = float(value)
        rows.append(row)
    return rows


def _domain_quantile_rows(
    df: pd.DataFrame,
    fields: Iterable[str],
) -> list[dict]:
    rows: list[dict] = []
    for domain, g in df.groupby("source_domain", dropna=False):
        domain_name = "<missing>" if pd.isna(domain) else str(domain)
        for field in fields:
            s = _finite_numeric(g[field])
            finite = s.dropna()
            q = finite.quantile(QUANTILES)
            row = {
                "source_domain": domain_name,
                "field": field,
                "rows": int(len(g)),
                "missing": int(s.isna().sum()),
                "min": float(finite.min()),
                "max": float(finite.max()),
            }
            for level, value in q.items():
                row[f"q{int(round(level * 100)):02d}"] = float(value)
            rows.append(row)
    return rows


def _attach_anchor(
    raw_a1: pd.DataFrame,
    standardized_a1: pd.DataFrame,
) -> pd.DataFrame:
    score_cols = [c for c in standardized_a1.columns if c.startswith("u_")]
    if len(score_cols) != 9:
        raise ValueError(
            f"Expected 9 monotonic score columns, found {len(score_cols)}: {score_cols}"
        )

    anchor = standardized_a1[["id"] + score_cols].copy()
    anchor["monotonic_anchor"] = anchor[score_cols].median(axis=1, skipna=True)

    merged = raw_a1.merge(
        anchor[["id", "monotonic_anchor"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if merged["monotonic_anchor"].isna().any():
        raise ValueError("Some A1 records could not be matched to monotonic scores.")
    return merged


def _decile_profile_for_group(
    g: pd.DataFrame,
    field: str,
    domain: str,
) -> list[dict]:
    x = _finite_numeric(g[field])
    a = _finite_numeric(g["monotonic_anchor"])
    mask = x.notna() & a.notna()
    x = x[mask]
    a = a[mask]

    if len(x) == 0:
        return []

    pct = x.rank(method="average", pct=True)
    decile = np.ceil(pct * 10.0).clip(1, 10).astype(int)

    temp = pd.DataFrame(
        {
            "x": x.to_numpy(),
            "anchor": a.to_numpy(),
            "decile": decile.to_numpy(),
        }
    )

    rows: list[dict] = []
    for d, b in temp.groupby("decile"):
        rows.append(
            {
                "field": field,
                "source_domain": domain,
                "decile": int(d),
                "n": int(len(b)),
                "raw_min": float(b["x"].min()),
                "raw_median": float(b["x"].median()),
                "raw_max": float(b["x"].max()),
                "anchor_mean": float(b["anchor"].mean()),
                "anchor_median": float(b["anchor"].median()),
            }
        )
    return rows


def _anchor_decile_rows(
    a1: pd.DataFrame,
    fields: Iterable[str],
) -> list[dict]:
    rows: list[dict] = []

    for field in fields:
        rows.extend(_decile_profile_for_group(a1, field, "ALL"))

        for domain, g in a1.groupby("source_domain", dropna=False):
            domain_name = "<missing>" if pd.isna(domain) else str(domain)
            rows.extend(_decile_profile_for_group(g, field, domain_name))

    return rows


def diagnose_nonmonotonic(
    preprocessed_dir: Path,
    standardized_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frames: Dict[str, pd.DataFrame] = {
        "A1": pd.read_csv(preprocessed_dir / "a1_quality_clean.csv.gz"),
        "A2": pd.read_csv(preprocessed_dir / "a2_quality_clean.csv.gz"),
        "A3": pd.read_csv(preprocessed_dir / "a3_quality_clean.csv.gz"),
    }

    for dataset, df in raw_frames.items():
        missing_fields = [f for f in NON_MONOTONIC_FIELDS if f not in df.columns]
        if missing_fields:
            raise ValueError(f"{dataset} is missing fields: {missing_fields}")

    distribution_rows: list[dict] = []
    for dataset, df in raw_frames.items():
        distribution_rows.extend(
            _distribution_rows(dataset, df, NON_MONOTONIC_FIELDS)
        )

    pd.DataFrame(distribution_rows).to_csv(
        output_dir / "nonmonotonic_distribution_audit.csv",
        index=False,
    )

    pd.DataFrame(
        _domain_quantile_rows(raw_frames["A1"], NON_MONOTONIC_FIELDS)
    ).to_csv(
        output_dir / "nonmonotonic_a1_domain_quantiles.csv",
        index=False,
    )

    standardized_a1 = pd.read_csv(
        standardized_dir / "a1_monotonic_scores.csv.gz"
    )
    a1_with_anchor = _attach_anchor(raw_frames["A1"], standardized_a1)

    pd.DataFrame(
        _anchor_decile_rows(a1_with_anchor, NON_MONOTONIC_FIELDS)
    ).to_csv(
        output_dir / "nonmonotonic_a1_anchor_deciles.csv",
        index=False,
    )

    print("Non-monotonic diagnostics completed.")
    print(f"Output directory: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose Q1 non-monotonic quality indicators before defining transforms."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("artifacts/q1/preprocessed"),
    )
    parser.add_argument(
        "--standardized-dir",
        type=Path,
        default=Path("artifacts/q1/standardized"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/diagnostics"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    diagnose_nonmonotonic(
        preprocessed_dir=args.preprocessed_dir,
        standardized_dir=args.standardized_dir,
        output_dir=args.output_dir,
    )

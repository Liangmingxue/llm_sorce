from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


UNIQUE_FIELD = "rps_doc_frac_unique_words"
LENGTH_FIELD = "rps_doc_word_count"


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(x[mask].corr(y[mask], method="spearman"))


def _partial_spearman(r_xy: float, r_xz: float, r_yz: float) -> float:
    denom = np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.nan
    return float((r_xy - r_xz * r_yz) / denom)


def diagnose_unique_words_length(
    preprocessed_dir: Path,
    standardized_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(preprocessed_dir / "a1_quality_clean.csv.gz")
    scores = pd.read_csv(standardized_dir / "a1_monotonic_scores.csv.gz")

    score_cols = [c for c in scores.columns if c.startswith("u_")]
    if len(score_cols) != 9:
        raise ValueError(f"Expected 9 monotonic score columns, got {len(score_cols)}")

    scores = scores[["id"] + score_cols].copy()
    scores["monotonic_anchor"] = scores[score_cols].median(axis=1, skipna=True)

    data = raw.merge(
        scores[["id", "monotonic_anchor"]],
        on="id",
        how="left",
        validate="one_to_one",
    )

    rows = []
    for domain, g in data.groupby("source_domain", dropna=False):
        r_unique_anchor = _spearman(g[UNIQUE_FIELD], g["monotonic_anchor"])
        r_unique_length = _spearman(g[UNIQUE_FIELD], g[LENGTH_FIELD])
        r_anchor_length = _spearman(g["monotonic_anchor"], g[LENGTH_FIELD])

        rows.append(
            {
                "source_domain": domain,
                "n": len(g),
                "rho_unique_anchor": r_unique_anchor,
                "rho_unique_wordcount": r_unique_length,
                "rho_anchor_wordcount": r_anchor_length,
                "partial_rho_unique_anchor_given_wordcount": _partial_spearman(
                    r_unique_anchor,
                    r_unique_length,
                    r_anchor_length,
                ),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(
        output_dir / "unique_words_length_partial_spearman.csv",
        index=False,
    )

    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    diagnose_unique_words_length(
        preprocessed_dir=ROOT / "artifacts" / "q1" / "preprocessed",
        standardized_dir=ROOT / "artifacts" / "q1" / "standardized",
        output_dir=ROOT / "artifacts" / "q1" / "diagnostics",
    )

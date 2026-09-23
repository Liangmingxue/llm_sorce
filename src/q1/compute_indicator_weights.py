from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _load_groups(config_path: Path) -> dict[str, list[str]]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    groups: dict[str, list[str]] = {}
    for group_name, spec in cfg["groups"].items():
        groups[group_name] = list(spec["fields"])

    flat = [field for fields in groups.values() for field in fields]
    if len(flat) != 22 or len(set(flat)) != 22:
        raise ValueError(
            f"Expected exactly 22 unique indicators across groups, got "
            f"{len(flat)} entries / {len(set(flat))} unique."
        )
    return groups


def _score_col(field: str) -> str:
    return f"u_{field}"


def _critic_components(
    df: pd.DataFrame,
    fields: list[str],
) -> pd.DataFrame:
    cols = [_score_col(f) for f in fields]
    x = df[cols].apply(pd.to_numeric, errors="coerce")

    std = x.std(axis=0, ddof=1)
    corr = x.corr(method="spearman", min_periods=20)

    rows = []
    for field in fields:
        col = _score_col(field)
        peers = [_score_col(k) for k in fields if k != field]

        if peers:
            r = corr.loc[col, peers].abs()
            valid = r.dropna()
            conflict = float((1.0 - valid).sum()) if len(valid) else 0.0
            mean_abs_corr = float(valid.mean()) if len(valid) else np.nan
        else:
            conflict = 1.0
            mean_abs_corr = np.nan

        sigma = float(std[col]) if np.isfinite(std[col]) else 0.0
        info = sigma * conflict

        rows.append(
            {
                "field": field,
                "std": sigma,
                "within_group_mean_abs_spearman": mean_abs_corr,
                "conflict": conflict,
                "critic_information": info,
            }
        )

    return pd.DataFrame(rows)


def _normalize_with_fallback(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(v.sum())
    if total > 0:
        return v / total
    return pd.Series(
        np.full(len(v), 1.0 / len(v), dtype=float),
        index=v.index,
    )


def compute_weights(
    transformed_a1: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(transformed_a1)
    groups = _load_groups(config_path)

    required_cols = {
        "source_domain",
        *[_score_col(f) for fields in groups.values() for f in fields],
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"A1 transformed table is missing columns: {missing}")

    all_fields = [f for fields in groups.values() for f in fields]
    all_cols = [_score_col(f) for f in all_fields]

    # Full 22x22 Spearman matrix on A1, useful for redundancy diagnostics.
    pooled_corr = (
        df[all_cols]
        .apply(pd.to_numeric, errors="coerce")
        .corr(method="spearman", min_periods=20)
    )
    pooled_corr.index = all_fields
    pooled_corr.columns = all_fields
    pooled_corr.to_csv(output_dir / "indicator_spearman_a1.csv")

    # Pair list sorted by absolute correlation for easier inspection.
    pair_rows = []
    for i, a in enumerate(all_fields):
        for b in all_fields[i + 1 :]:
            rho = pooled_corr.loc[a, b]
            pair_rows.append(
                {
                    "field_a": a,
                    "field_b": b,
                    "spearman": float(rho) if np.isfinite(rho) else np.nan,
                    "abs_spearman": float(abs(rho)) if np.isfinite(rho) else np.nan,
                }
            )
    (
        pd.DataFrame(pair_rows)
        .sort_values("abs_spearman", ascending=False, na_position="last")
        .to_csv(output_dir / "indicator_correlation_pairs_a1.csv", index=False)
    )

    domains = [str(x) for x in df["source_domain"].dropna().unique()]
    domain_rows = []
    pooled_rows = []

    for group_name, fields in groups.items():
        pooled = _critic_components(df, fields)
        pooled.insert(0, "group", group_name)
        pooled_rows.append(pooled)

        for domain in domains:
            g = df[df["source_domain"] == domain]
            comp = _critic_components(g, fields)
            comp.insert(0, "domain", domain)
            comp.insert(0, "group", group_name)
            domain_rows.append(comp)

    domain_diag = pd.concat(domain_rows, ignore_index=True)
    domain_diag.to_csv(
        output_dir / "critic_domain_diagnostics.csv",
        index=False,
    )

    pooled_diag = pd.concat(pooled_rows, ignore_index=True).rename(
        columns={
            "std": "pooled_std",
            "within_group_mean_abs_spearman": "pooled_within_group_mean_abs_spearman",
            "conflict": "pooled_conflict",
            "critic_information": "pooled_critic_information",
        }
    )

    # Domain-balanced CRITIC:
    # each of the 7 source domains contributes equally, so large domains do not
    # dominate the indicator weights merely because they contain more A1 rows.
    balanced = (
        domain_diag.groupby(["group", "field"], as_index=False)
        .agg(
            domain_balanced_std=("std", "mean"),
            domain_balanced_mean_abs_spearman=(
                "within_group_mean_abs_spearman",
                "mean",
            ),
            domain_balanced_conflict=("conflict", "mean"),
            domain_balanced_critic_information=("critic_information", "mean"),
        )
    )

    diagnostics = balanced.merge(
        pooled_diag[
            [
                "group",
                "field",
                "pooled_std",
                "pooled_within_group_mean_abs_spearman",
                "pooled_conflict",
                "pooled_critic_information",
            ]
        ],
        on=["group", "field"],
        how="left",
        validate="one_to_one",
    )

    n_groups = len(groups)
    equal_group_weight = 1.0 / n_groups

    weight_parts = []
    for group_name, g in diagnostics.groupby("group", sort=False):
        g = g.copy()
        g["within_group_weight"] = _normalize_with_fallback(
            g["domain_balanced_critic_information"]
        ).to_numpy()
        g["group_weight"] = equal_group_weight
        g["preliminary_weight"] = (
            g["within_group_weight"] * g["group_weight"]
        )
        weight_parts.append(g)

    weights = pd.concat(weight_parts, ignore_index=True)

    # Preserve configured group/field order rather than alphabetical order.
    order = {
        (group_name, field): rank
        for rank, (group_name, field) in enumerate(
            (g, f)
            for g, fields in groups.items()
            for f in fields
        )
    }
    weights["_order"] = [
        order[(g, f)] for g, f in zip(weights["group"], weights["field"])
    ]
    weights = weights.sort_values("_order").drop(columns="_order")

    weights.to_csv(
        output_dir / "preliminary_group_balanced_critic_weights.csv",
        index=False,
    )

    group_summary = (
        weights.groupby("group", as_index=False)
        .agg(
            indicators=("field", "count"),
            assigned_group_weight=("group_weight", "first"),
            resulting_weight_sum=("preliminary_weight", "sum"),
        )
    )
    group_summary.to_csv(output_dir / "group_weight_summary.csv", index=False)

    print("Indicator-weight diagnostics completed.")
    print(f"Output directory: {output_dir}")
    print(f"A1 rows: {len(df)}")
    print(f"Domains: {len(domains)} ({', '.join(domains)})")
    print(f"Indicators: {len(all_fields)}")
    print(f"Groups: {n_groups}; preliminary group weight = {equal_group_weight:.4f} each")
    print(f"Preliminary indicator weights sum = {weights['preliminary_weight'].sum():.12f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Q1 indicator-correlation diagnostics and preliminary group-balanced CRITIC weights."
    )
    parser.add_argument(
        "--transformed-a1",
        type=Path,
        default=Path("artifacts/q1/transformed/a1_quality_22_scores.csv.gz"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1/quality_transform.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/model"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    compute_weights(
        transformed_a1=args.transformed_a1,
        config_path=args.config,
        output_dir=args.output_dir,
    )

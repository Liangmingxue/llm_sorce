from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DATASET_FILES = {
    "train_1m": "regmix_train_1m_clean.csv",
    "test_1m": "regmix_test_1m_clean.csv",
    "test_60m": "regmix_test_60m_clean.csv",
    "test_1B": "regmix_test_1B_clean.csv",
    "est_10b": "regmix_est_10b_clean.csv",
    "est_70b": "regmix_est_70b_clean.csv",
}


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid RegMix audit config: {path}")
    return cfg


def _schema(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    p_cols = sorted(
        c for c in df.columns
        if c.startswith("p_") and not c.startswith("p_raw_")
    )
    p_raw_cols = sorted(c for c in df.columns if c.startswith("p_raw_"))
    loss_cols = sorted(c for c in df.columns if c.startswith("loss_"))
    return p_cols, p_raw_cols, loss_cols


def _recipe_set(df: pd.DataFrame, cols: list[str], decimals: int) -> set[tuple[float, ...]]:
    arr = np.round(df[cols].to_numpy(dtype=float), decimals)
    return {tuple(row) for row in arr}


def run_regmix_audit(
    preprocessed_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(config_path)

    dfs = {
        name: pd.read_csv(preprocessed_dir / filename)
        for name, filename in DATASET_FILES.items()
    }

    p_cols, p_raw_cols, loss_cols = _schema(dfs["train_1m"])

    expected_p = int(cfg.get("expected_mixture_columns", 17))
    expected_loss = int(cfg.get("expected_loss_columns", 13))
    tol = float(cfg.get("simplex_tolerance", 1e-12))
    norm_decimals = int(cfg.get("normalized_recipe_decimals", 12))
    raw_decimals = int(cfg.get("raw_recipe_decimals", 6))

    if len(p_cols) != expected_p:
        raise ValueError(f"Expected {expected_p} normalized p columns, got {len(p_cols)}")
    if len(p_raw_cols) != expected_p:
        raise ValueError(f"Expected {expected_p} raw p columns, got {len(p_raw_cols)}")
    if len(loss_cols) != expected_loss:
        raise ValueError(f"Expected {expected_loss} loss columns, got {len(loss_cols)}")

    audit_rows = []
    norm_sets: dict[str, set[tuple[float, ...]]] = {}
    raw_sets: dict[str, set[tuple[float, ...]]] = {}

    for name, df in dfs.items():
        pc, prc, lc = _schema(df)
        if pc != p_cols or prc != p_raw_cols or lc != loss_cols:
            raise ValueError(f"Schema mismatch in {name}")

        X = df[p_cols].to_numpy(dtype=float)
        simplex_error = np.abs(X.sum(axis=1) - 1.0)
        centered_rank = int(np.linalg.matrix_rank(X - X.mean(axis=0, keepdims=True)))

        norm_sets[name] = _recipe_set(df, p_cols, norm_decimals)
        raw_sets[name] = _recipe_set(df, p_raw_cols, raw_decimals)

        audit_rows.append(
            {
                "dataset": name,
                "rows": int(len(df)),
                "split": ",".join(sorted(df["split"].astype(str).unique())),
                "scale": ",".join(sorted(df["scale"].astype(str).unique())),
                "p_cols": len(pc),
                "loss_cols": len(lc),
                "missing_total": int(df[p_cols + loss_cols].isna().sum().sum()),
                "simplex_max_error": float(simplex_error.max()),
                "centered_p_rank": centered_rank,
                "unique_norm_recipes": len(norm_sets[name]),
                "duplicate_norm_recipes": int(len(df) - len(norm_sets[name])),
                "unique_raw_recipes": len(raw_sets[name]),
                "duplicate_raw_recipes": int(len(df) - len(raw_sets[name])),
                "min_p": float(X.min()),
                "max_p": float(X.max()),
                "loss_min": float(df[loss_cols].min().min()),
                "loss_max": float(df[loss_cols].max().max()),
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "regmix_modeling_audit.csv", index=False)

    names = list(dfs)
    norm_overlap = pd.DataFrame(index=names, columns=names, dtype=int)
    raw_overlap = pd.DataFrame(index=names, columns=names, dtype=int)

    for a in names:
        for b in names:
            norm_overlap.loc[a, b] = len(norm_sets[a] & norm_sets[b])
            raw_overlap.loc[a, b] = len(raw_sets[a] & raw_sets[b])

    norm_overlap.index.name = "dataset"
    raw_overlap.index.name = "dataset"
    norm_overlap.to_csv(output_dir / "recipe_overlap_normalized.csv")
    raw_overlap.to_csv(output_dir / "recipe_overlap_raw.csv")

    train = dfs["train_1m"]

    coverage_rows = []
    for col in p_cols:
        s = pd.to_numeric(train[col], errors="coerce")
        coverage_rows.append(
            {
                "domain": col.removeprefix("p_"),
                "min": float(s.min()),
                "P05": float(s.quantile(0.05)),
                "mean": float(s.mean()),
                "P95": float(s.quantile(0.95)),
                "max": float(s.max()),
                "std": float(s.std(ddof=1)),
                "zero_fraction": float(np.isclose(s, 0.0, atol=1e-15).mean()),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "train_1m_mixture_coverage.csv", index=False
    )

    loss_rows = []
    for col in loss_cols:
        s = pd.to_numeric(train[col], errors="coerce")
        loss_rows.append(
            {
                "target": col,
                "min": float(s.min()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "max": float(s.max()),
                "cv": float(s.std(ddof=1) / abs(s.mean())) if s.mean() != 0 else np.nan,
            }
        )
    pd.DataFrame(loss_rows).to_csv(
        output_dir / "train_1m_loss_variation.csv", index=False
    )

    zero_fractions = {
        col.removeprefix("p_"): float(np.isclose(train[col], 0.0, atol=1e-15).mean())
        for col in p_cols
    }

    manifest = {
        "mixture_columns": p_cols,
        "loss_columns": loss_cols,
        "train_centered_rank": int(
            audit.loc[audit["dataset"] == "train_1m", "centered_p_rank"].iloc[0]
        ),
        "simplex_dimension": len(p_cols) - 1,
        "train_zero_fraction_by_domain": zero_fractions,
        "important_overlap_facts": {
            "test_1m_vs_test_60m": int(
                len(norm_sets["test_1m"] & norm_sets["test_60m"])
            ),
            "train_1m_vs_est_10b": int(
                len(norm_sets["train_1m"] & norm_sets["est_10b"])
            ),
            "train_1m_vs_est_70b": int(
                len(norm_sets["train_1m"] & norm_sets["est_70b"])
            ),
            "est_10b_vs_est_70b": int(
                len(norm_sets["est_10b"] & norm_sets["est_70b"])
            ),
        },
        "modeling_implications": [
            "The 17 proportions live on a simplex with 16 degrees of freedom.",
            "A naive 17-proportion plus intercept ordinary linear regression is rank-deficient.",
            "Zero proportions are frequent, so raw CLR/ILR transforms are undefined without a zero-replacement rule.",
            "Any log-ratio model must therefore include explicit zero replacement and sensitivity analysis; it should not be the only model family.",
            "test_1m and test_60m use the same 256 mixture recipes and should be treated as paired composition designs at different scales.",
            "est_10b and est_70b use the same 63 recipes, all overlapping train_1m; they are extrapolation-at-new-scale sets, not independent composition holdouts.",
            "test_1m/test_60m/test_1B remain excluded from model fitting.",
            "est_10b/est_70b remain excluded from model fitting and independent-test scoring.",
        ],
    }

    (output_dir / "regmix_modeling_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    hard_checks = {
        "all_have_17_mixture_columns": bool((audit["p_cols"] == 17).all()),
        "all_have_13_loss_columns": bool((audit["loss_cols"] == 13).all()),
        "all_no_missing": bool((audit["missing_total"] == 0).all()),
        "all_simplex_error_lt_tolerance": bool(
            (audit["simplex_max_error"] < tol).all()
        ),
        "train_centered_rank_is_16": bool(
            audit.loc[audit["dataset"] == "train_1m", "centered_p_rank"].iloc[0] == 16
        ),
        "all_p_nonnegative": bool((audit["min_p"] >= -tol).all()),
        "all_p_le_1": bool((audit["max_p"] <= 1 + tol).all()),
    }

    (output_dir / "hard_checks.json").write_text(
        json.dumps(hard_checks, indent=2),
        encoding="utf-8",
    )

    if not all(hard_checks.values()):
        raise RuntimeError(f"RegMix hard checks failed: {hard_checks}")

    print("RegMix modeling audit completed.")
    print(f"Output directory: {output_dir}")
    print(f"Mixture columns: {len(p_cols)}; simplex dimension: {len(p_cols)-1}")
    print(f"Loss targets: {len(loss_cols)}")
    print(
        "Recipe overlaps:",
        f"test_1m/test_60m={manifest['important_overlap_facts']['test_1m_vs_test_60m']},",
        f"train_1m/est_10b={manifest['important_overlap_facts']['train_1m_vs_est_10b']},",
        f"train_1m/est_70b={manifest['important_overlap_facts']['train_1m_vs_est_70b']}",
    )
    print("Hard checks: all passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Q1 RegMix data before mixture-loss modeling."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("artifacts/q1/preprocessed"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1/regmix_audit.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_audit"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_regmix_audit(
        preprocessed_dir=args.preprocessed_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )

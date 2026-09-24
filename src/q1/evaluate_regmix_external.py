from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TEST_FILES = {
    "test_1m": "regmix_test_1m_clean.csv",
    "test_60m": "regmix_test_60m_clean.csv",
    "test_1B": "regmix_test_1B_clean.csv",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_corr(
    a: np.ndarray,
    b: np.ndarray,
    method: str,
) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(pd.Series(a).corr(pd.Series(b), method=method))


def _affine_diagnostic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """
    Post-hoc diagnostic only:
        y_true ~= intercept + slope * y_pred.
    This is never used to alter a frozen model.
    """
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)

    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta

    return (
        float(beta[0]),
        float(beta[1]),
        float(r2_score(y, fitted)),
    )


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_sd: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    test_sd = float(y_true.std(ddof=1))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # Remove only the mean residual. This is a diagnostic of whether the
    # composition-response shape transfers across model scales.
    residual = y_pred - y_true
    mean_bias = float(residual.mean())
    shifted_pred = y_pred - mean_bias
    centered_rmse = float(
        np.sqrt(mean_squared_error(y_true, shifted_pred))
    )

    intercept, slope, affine_r2 = _affine_diagnostic(y_true, y_pred)

    return {
        "test_target_mean": float(y_true.mean()),
        "test_target_sd": test_sd,
        "pred_mean": float(y_pred.mean()),
        "mean_bias_pred_minus_true": mean_bias,
        "rmse_direct": rmse,
        "nrmse_direct_by_train_sd": (
            float(rmse / train_sd) if train_sd > 0 else np.nan
        ),
        "nrmse_direct_by_test_sd": (
            float(rmse / test_sd) if test_sd > 0 else np.nan
        ),
        "mae_direct": mae,
        "r2_direct": float(r2_score(y_true, y_pred)),
        "pearson": _safe_corr(y_true, y_pred, "pearson"),
        "spearman": _safe_corr(y_true, y_pred, "spearman"),
        "centered_rmse": centered_rmse,
        "centered_nrmse_by_test_sd": (
            float(centered_rmse / test_sd) if test_sd > 0 else np.nan
        ),
        "affine_diag_intercept": intercept,
        "affine_diag_slope": slope,
        "affine_diag_r2": affine_r2,
    }


def _recipe_signature(
    df: pd.DataFrame,
    p_cols: list[str],
    decimals: int = 12,
) -> pd.Series:
    arr = np.round(df[p_cols].to_numpy(dtype=float), decimals)
    return pd.Series(
        ["|".join(f"{v:.{decimals}f}" for v in row) for row in arr],
        index=df.index,
    )


def run_external_regmix_test(
    preprocessed_dir: Path,
    frozen_dir: Path,
    output_dir: Path,
) -> None:
    completion_marker = output_dir / "EXTERNAL_TEST_EVALUATED.json"
    if completion_marker.exists():
        raise RuntimeError(
            "External test has already been evaluated. "
            f"Completion marker exists: {completion_marker}. "
            "Do not rerun for model selection."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_manifest_path = frozen_dir / "freeze_manifest.json"
    with freeze_manifest_path.open("r", encoding="utf-8") as f:
        freeze_manifest: dict[str, Any] = json.load(f)

    if freeze_manifest.get("status") != "FROZEN_BEFORE_EXTERNAL_TEST":
        raise ValueError(
            "Frozen manifest status is not FROZEN_BEFORE_EXTERNAL_TEST."
        )

    frozen_table = pd.read_csv(
        frozen_dir / "frozen_target_models.csv"
    )

    p_cols = list(freeze_manifest["mixture_columns"])
    loss_targets = list(freeze_manifest["loss_targets"])
    family_map = dict(freeze_manifest["target_family"])
    expected_hashes = dict(freeze_manifest["model_sha256"])

    if set(frozen_table["target"]) != set(loss_targets):
        raise ValueError("Frozen model table does not match manifest targets.")

    models: dict[str, Any] = {}
    train_sd: dict[str, float] = {}

    for _, row in frozen_table.iterrows():
        target = str(row["target"])
        model_path = frozen_dir / str(row["model_file"])

        actual_hash = _sha256(model_path)
        expected_hash = expected_hashes[target]

        if actual_hash != expected_hash:
            raise ValueError(
                f"Model hash mismatch for {target}: "
                f"expected={expected_hash}, actual={actual_hash}"
            )

        models[target] = joblib.load(model_path)
        train_sd[target] = float(row["train_target_sd"])

    tests: dict[str, pd.DataFrame] = {
        name: pd.read_csv(preprocessed_dir / filename)
        for name, filename in TEST_FILES.items()
    }

    target_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for dataset, df in tests.items():
        missing_cols = sorted(
            set(p_cols + loss_targets) - set(df.columns)
        )
        if missing_cols:
            raise ValueError(
                f"{dataset} missing required columns: {missing_cols}"
            )

        X = df[p_cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{dataset} has non-finite mixture values.")
        if np.max(np.abs(X.sum(axis=1) - 1.0)) >= 1e-12:
            raise ValueError(f"{dataset} violates simplex constraint.")

        for target in loss_targets:
            y = pd.to_numeric(
                df[target],
                errors="raise",
            ).to_numpy(dtype=float)

            pred = np.asarray(
                models[target].predict(X),
                dtype=float,
            )

            m = _metrics(
                y_true=y,
                y_pred=pred,
                train_sd=train_sd[target],
            )

            target_rows.append(
                {
                    "dataset": dataset,
                    "scale": str(df["scale"].iloc[0]),
                    "n": int(len(df)),
                    "target": target,
                    "family": family_map[target],
                    **m,
                }
            )

            for i, (yt, yp) in enumerate(zip(y, pred)):
                prediction_rows.append(
                    {
                        "dataset": dataset,
                        "scale": str(df["scale"].iloc[0]),
                        "row_index": int(i),
                        "recipe_index": int(df.iloc[i]["index"]),
                        "target": target,
                        "family": family_map[target],
                        "y_true": float(yt),
                        "y_pred_frozen_1m_model": float(yp),
                        "residual_pred_minus_true": float(yp - yt),
                    }
                )

    target_metrics = pd.DataFrame(target_rows)
    target_metrics.to_csv(
        output_dir / "external_test_target_metrics.csv",
        index=False,
    )

    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "external_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    summary_rows: list[dict[str, Any]] = []
    for dataset, g in target_metrics.groupby("dataset", sort=False):
        summary_rows.append(
            {
                "dataset": dataset,
                "scale": g["scale"].iloc[0],
                "targets": int(len(g)),
                "rows_per_target": int(g["n"].iloc[0]),
                "macro_rmse_direct": float(g["rmse_direct"].mean()),
                "macro_nrmse_direct_by_train_sd": float(
                    g["nrmse_direct_by_train_sd"].mean()
                ),
                "macro_nrmse_direct_by_test_sd": float(
                    g["nrmse_direct_by_test_sd"].mean()
                ),
                "macro_r2_direct": float(g["r2_direct"].mean()),
                "macro_pearson": float(g["pearson"].mean()),
                "macro_spearman": float(g["spearman"].mean()),
                "median_spearman": float(g["spearman"].median()),
                "macro_centered_nrmse_by_test_sd": float(
                    g["centered_nrmse_by_test_sd"].mean()
                ),
                "macro_affine_diag_r2": float(g["affine_diag_r2"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "external_test_summary.csv",
        index=False,
    )

    family_summary = (
        target_metrics.groupby(
            ["dataset", "scale", "family"],
            as_index=False,
        )
        .agg(
            targets=("target", "count"),
            macro_nrmse_direct_by_train_sd=(
                "nrmse_direct_by_train_sd",
                "mean",
            ),
            macro_spearman=("spearman", "mean"),
            macro_centered_nrmse_by_test_sd=(
                "centered_nrmse_by_test_sd",
                "mean",
            ),
            macro_affine_diag_r2=("affine_diag_r2", "mean"),
        )
    )
    family_summary.to_csv(
        output_dir / "external_test_family_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Paired 1m vs 60m actual-loss consistency.
    # These two sets use the same 256 mixture recipes.
    # ----------------------------------------------------------
    d1 = tests["test_1m"].copy()
    d60 = tests["test_60m"].copy()

    d1["recipe_signature"] = _recipe_signature(d1, p_cols)
    d60["recipe_signature"] = _recipe_signature(d60, p_cols)

    if d1["recipe_signature"].duplicated().any():
        raise ValueError("Duplicate recipe signature in test_1m.")
    if d60["recipe_signature"].duplicated().any():
        raise ValueError("Duplicate recipe signature in test_60m.")

    paired = d1[
        ["recipe_signature", *loss_targets]
    ].merge(
        d60[
            ["recipe_signature", *loss_targets]
        ],
        on="recipe_signature",
        suffixes=("_1m", "_60m"),
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != 256:
        raise ValueError(
            f"Expected 256 paired 1m/60m recipes, got {len(paired)}"
        )

    paired_rows = []
    for target in loss_targets:
        y1 = paired[f"{target}_1m"].to_numpy(dtype=float)
        y60 = paired[f"{target}_60m"].to_numpy(dtype=float)

        intercept, slope, affine_r2 = _affine_diagnostic(
            y_true=y60,
            y_pred=y1,
        )

        paired_rows.append(
            {
                "target": target,
                "n_paired_recipes": int(len(paired)),
                "mean_loss_1m": float(y1.mean()),
                "mean_loss_60m": float(y60.mean()),
                "mean_shift_60m_minus_1m": float(
                    (y60 - y1).mean()
                ),
                "pearson_1m_vs_60m": _safe_corr(
                    y1,
                    y60,
                    "pearson",
                ),
                "spearman_1m_vs_60m": _safe_corr(
                    y1,
                    y60,
                    "spearman",
                ),
                "affine_60m_from_1m_intercept": intercept,
                "affine_60m_from_1m_slope": slope,
                "affine_60m_from_1m_r2": affine_r2,
            }
        )

    paired_scale = pd.DataFrame(paired_rows)
    paired_scale.to_csv(
        output_dir / "paired_test_1m_vs_60m_scale_transfer.csv",
        index=False,
    )

    # Write marker last, only after all outputs are complete.
    marker = {
        "status": "EXTERNAL_TEST_EVALUATED_ONCE",
        "frozen_manifest_sha256": _sha256(freeze_manifest_path),
        "frozen_status": freeze_manifest["status"],
        "datasets": {
            name: {
                "file": TEST_FILES[name],
                "rows": int(len(df)),
                "sha256": _sha256(
                    preprocessed_dir / TEST_FILES[name]
                ),
            }
            for name, df in tests.items()
        },
        "model_policy": (
            "All model families and hyperparameter grids were frozen before "
            "this evaluation. External results must not be used to alter them."
        ),
        "metric_interpretation": {
            "test_1m": (
                "Direct RMSE/NRMSE/R2 are primary because frozen models were "
                "trained on 1m loss."
            ),
            "test_60m_test_1B": (
                "Raw direct metrics are reported honestly, but model scale "
                "changes shift absolute loss. Spearman/Pearson and centered "
                "NRMSE are the primary composition-effect transfer diagnostics. "
                "Affine diagnostic quantities are post-hoc descriptive only."
            ),
        },
        "a12_a15_used": False,
    }

    completion_marker.write_text(
        json.dumps(marker, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Frozen RegMix external test completed.")
    print(f"Output directory: {output_dir}")
    print("Frozen models changed after seeing test data: NO")
    print("A12-A15 used: NO")
    print("Status: EXTERNAL_TEST_EVALUATED_ONCE")
    print("\nExternal-test summary:")
    print(summary.round(6).to_string(index=False))
    print("\nPaired 1m-vs-60m actual-loss transfer summary:")
    print(
        paired_scale[
            [
                "target",
                "spearman_1m_vs_60m",
                "affine_60m_from_1m_slope",
                "affine_60m_from_1m_r2",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Q1 RegMix models exactly once on A6-A11."
        )
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("artifacts/q1/preprocessed"),
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_final"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_external_test"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_external_regmix_test(
        preprocessed_dir=args.preprocessed_dir,
        frozen_dir=args.frozen_dir,
        output_dir=args.output_dir,
    )

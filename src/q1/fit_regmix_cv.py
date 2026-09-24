from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid RegMix CV config: {path}")
    return cfg


def helmert_basis(d: int) -> np.ndarray:
    """Return a d x (d-1) orthonormal basis of the sum-zero subspace."""
    if d < 2:
        raise ValueError("d must be >= 2")

    h = np.zeros((d, d - 1), dtype=float)
    for j in range(d - 1):
        denom = np.sqrt((j + 1) * (j + 2))
        h[: j + 1, j] = 1.0 / denom
        h[j + 1, j] = -(j + 1) / denom

    if not np.allclose(h.T @ h, np.eye(d - 1), atol=1e-12):
        raise RuntimeError("Helmert basis is not orthonormal.")
    if not np.allclose(h.sum(axis=0), 0.0, atol=1e-12):
        raise RuntimeError("Helmert basis does not span the simplex tangent space.")
    return h


class SimplexOrthogonalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        self.basis_ = helmert_basis(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.basis_


class ZeroReplacedILRTransformer(BaseEstimator, TransformerMixin):
    """Additive zero replacement followed by orthonormal log-ratio coordinates."""

    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = epsilon

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        self.basis_ = helmert_basis(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        eps = float(self.epsilon)
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError(f"epsilon must be positive and finite, got {eps}")

        xr = X + eps
        xr = xr / xr.sum(axis=1, keepdims=True)

        if np.any(xr <= 0):
            raise ValueError("Zero replacement failed to produce positive compositions.")

        # Because each Helmert column sums to zero, log(x) @ H is an ILR basis.
        return np.log(xr) @ self.basis_


@dataclass
class FamilySpec:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]] | None


def _family_specs(cfg: dict[str, Any]) -> list[FamilySpec]:
    ridge_alphas = [float(x) for x in cfg["ridge_alphas"]]
    elastic_alphas = [float(x) for x in cfg["elasticnet_alphas"]]
    l1_ratios = [float(x) for x in cfg["elasticnet_l1_ratios"]]
    ilr_eps = [float(x) for x in cfg["ilr_epsilons"]]

    specs = [
        FamilySpec(
            name="linear_simplex",
            estimator=Pipeline(
                [
                    ("simplex", SimplexOrthogonalTransformer()),
                    ("scale", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            param_grid=None,
        ),
        FamilySpec(
            name="ridge_simplex",
            estimator=Pipeline(
                [
                    ("simplex", SimplexOrthogonalTransformer()),
                    ("scale", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            param_grid={"model__alpha": ridge_alphas},
        ),
        FamilySpec(
            name="quadratic_ridge_simplex",
            estimator=Pipeline(
                [
                    ("simplex", SimplexOrthogonalTransformer()),
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=2,
                            include_bias=False,
                        ),
                    ),
                    ("scale", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            param_grid={"model__alpha": ridge_alphas},
        ),
        FamilySpec(
            name="elasticnet_simplex",
            estimator=Pipeline(
                [
                    ("simplex", SimplexOrthogonalTransformer()),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            max_iter=int(cfg.get("elasticnet_max_iter", 50000)),
                            tol=float(cfg.get("elasticnet_tol", 1e-7)),
                            selection="cyclic",
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__alpha": elastic_alphas,
                "model__l1_ratio": l1_ratios,
            },
        ),
        FamilySpec(
            name="ilr_ridge",
            estimator=Pipeline(
                [
                    ("ilr", ZeroReplacedILRTransformer()),
                    ("scale", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            param_grid={
                "ilr__epsilon": ilr_eps,
                "model__alpha": ridge_alphas,
            },
        ),
    ]
    return specs


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = pd.Series(np.asarray(y_true, dtype=float))
    b = pd.Series(np.asarray(y_pred, dtype=float))
    return float(a.corr(b, method="spearman"))


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_sd: float,
) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "nrmse_by_train_sd": float(rmse / target_sd) if target_sd > 0 else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": _safe_spearman(y_true, y_pred),
    }


def _fit_with_inner_cv(
    spec: FamilySpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: KFold,
    n_jobs: int,
):
    if not spec.param_grid:
        fitted = spec.estimator.fit(X_train, y_train)
        return fitted, {}

    search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def run_regmix_nested_cv(
    preprocessed_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(config_path)

    data_path = preprocessed_dir / "regmix_train_1m_clean.csv"
    df = pd.read_csv(data_path)

    p_cols = sorted(
        c for c in df.columns
        if c.startswith("p_") and not c.startswith("p_raw_")
    )
    loss_cols = sorted(c for c in df.columns if c.startswith("loss_"))

    if len(p_cols) != 17 or len(loss_cols) != 13:
        raise ValueError(
            f"Expected 17 mixture columns and 13 losses, got "
            f"{len(p_cols)} and {len(loss_cols)}"
        )

    X = df[p_cols].to_numpy(dtype=float)
    Y = df[loss_cols].to_numpy(dtype=float)

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("Non-finite values found in train_1m.")

    if np.max(np.abs(X.sum(axis=1) - 1.0)) >= 1e-12:
        raise ValueError("Mixture rows do not satisfy the simplex constraint.")

    outer_folds = int(cfg.get("outer_folds", 5))
    inner_folds = int(cfg.get("inner_folds", 4))
    seed = int(cfg.get("random_seed", 20260924))
    n_jobs = int(cfg.get("n_jobs", -1))

    outer_cv = KFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=seed,
    )

    specs = _family_specs(cfg)
    family_names = ["mean_baseline", *[s.name for s in specs]]

    oof = {
        (target, family): np.full(len(df), np.nan, dtype=float)
        for target in loss_cols
        for family in family_names
    }

    selected_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for target_idx, target in enumerate(loss_cols):
        y = Y[:, target_idx]
        target_sd = float(y.std(ddof=1))

        for fold_idx, (train_idx, valid_idx) in enumerate(outer_cv.split(X), start=1):
            Xtr, Xva = X[train_idx], X[valid_idx]
            ytr, yva = y[train_idx], y[valid_idx]

            baseline_pred = np.full(len(valid_idx), ytr.mean(), dtype=float)
            oof[(target, "mean_baseline")][valid_idx] = baseline_pred

            bm = _metrics(yva, baseline_pred, target_sd)
            fold_rows.append(
                {
                    "target": target,
                    "family": "mean_baseline",
                    "outer_fold": fold_idx,
                    **bm,
                }
            )

            inner_cv = KFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=seed + 1000 * target_idx + fold_idx,
            )

            for spec in specs:
                model, best_params = _fit_with_inner_cv(
                    spec,
                    Xtr,
                    ytr,
                    inner_cv=inner_cv,
                    n_jobs=n_jobs,
                )
                pred = model.predict(Xva)
                oof[(target, spec.name)][valid_idx] = pred

                fm = _metrics(yva, pred, target_sd)
                fold_rows.append(
                    {
                        "target": target,
                        "family": spec.name,
                        "outer_fold": fold_idx,
                        **fm,
                    }
                )

                row = {
                    "target": target,
                    "family": spec.name,
                    "outer_fold": fold_idx,
                }
                for key, value in best_params.items():
                    row[key] = value
                selected_rows.append(row)

        print(f"Completed target {target_idx + 1}/{len(loss_cols)}: {target}")

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for target_idx, target in enumerate(loss_cols):
        y = Y[:, target_idx]
        target_sd = float(y.std(ddof=1))

        for family in family_names:
            pred = oof[(target, family)]

            if not np.all(np.isfinite(pred)):
                raise RuntimeError(f"OOF prediction missing for {target}/{family}")

            mm = _metrics(y, pred, target_sd)
            metric_rows.append(
                {
                    "target": target,
                    "family": family,
                    "target_mean": float(y.mean()),
                    "target_sd": target_sd,
                    **mm,
                }
            )

            for row_idx, (yt, yp) in enumerate(zip(y, pred)):
                pred_rows.append(
                    {
                        "row_index": int(row_idx),
                        "recipe_index": int(df.iloc[row_idx]["index"]),
                        "target": target,
                        "family": family,
                        "y_true": float(yt),
                        "y_pred_oof": float(yp),
                        "residual": float(yp - yt),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    selected = pd.DataFrame(selected_rows)

    metrics.to_csv(output_dir / "nested_cv_target_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / "nested_cv_fold_metrics.csv", index=False)
    selected.to_csv(output_dir / "nested_cv_selected_params.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(
        output_dir / "nested_cv_oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    summary_rows = []
    for family, g in metrics.groupby("family", sort=False):
        summary_rows.append(
            {
                "family": family,
                "targets": int(len(g)),
                "macro_rmse": float(g["rmse"].mean()),
                "macro_nrmse": float(g["nrmse_by_train_sd"].mean()),
                "median_nrmse": float(g["nrmse_by_train_sd"].median()),
                "macro_mae": float(g["mae"].mean()),
                "macro_r2": float(g["r2"].mean()),
                "macro_spearman": float(g["spearman"].mean()),
                "targets_best_nrmse": 0,
            }
        )

    summary = pd.DataFrame(summary_rows)

    best_by_target = (
        metrics[metrics["family"] != "mean_baseline"]
        .sort_values(["target", "nrmse_by_train_sd", "family"])
        .groupby("target", as_index=False)
        .first()[["target", "family", "nrmse_by_train_sd"]]
    )
    best_counts = best_by_target["family"].value_counts()

    summary["targets_best_nrmse"] = (
        summary["family"].map(best_counts).fillna(0).astype(int)
    )
    summary = summary.sort_values(
        ["macro_nrmse", "macro_rmse", "family"],
        kind="stable",
    ).reset_index(drop=True)

    summary.to_csv(output_dir / "model_family_summary.csv", index=False)
    best_by_target.to_csv(output_dir / "best_family_by_target.csv", index=False)

    # ILR sensitivity: aggregate the selected epsilon frequency across outer folds.
    ilr_rows = selected[selected["family"] == "ilr_ridge"].copy()
    if "ilr__epsilon" in ilr_rows.columns:
        ilr_sensitivity = (
            ilr_rows.groupby(["target", "ilr__epsilon"], dropna=False)
            .size()
            .reset_index(name="selected_outer_folds")
        )
        ilr_sensitivity.to_csv(
            output_dir / "ilr_epsilon_selection.csv",
            index=False,
        )

    manifest = {
        "data_used_for_fitting": "A4/A5 train_1m only",
        "rows": int(len(df)),
        "mixture_dimensions": 17,
        "simplex_degrees_of_freedom": 16,
        "targets": loss_cols,
        "outer_cv": {
            "type": "KFold",
            "folds": outer_folds,
            "shuffle": True,
            "random_seed": seed,
        },
        "inner_cv": {
            "type": "KFold",
            "folds": inner_folds,
            "shuffle": True,
            "purpose": "hyperparameter selection only inside each outer training fold",
        },
        "families": family_names,
        "external_sets_untouched": [
            "test_1m",
            "test_60m",
            "test_1B",
            "est_10b",
            "est_70b",
        ],
        "selection_rule": (
            "Compare model families using out-of-fold target-level metrics; "
            "macro normalized RMSE is the primary family summary. "
            "No external test set participates in model selection."
        ),
        "ilr_zero_replacement": {
            "method": "add epsilon to every component then renormalize",
            "epsilon_grid": [float(x) for x in cfg["ilr_epsilons"]],
            "note": (
                "ILR is a sensitivity baseline because exact zeros are frequent; "
                "the final model is not chosen from ILR without checking epsilon stability."
            ),
        },
    }

    (output_dir / "nested_cv_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("RegMix nested CV completed.")
    print(f"Output directory: {output_dir}")
    print(f"Rows used: {len(df)} (train_1m only)")
    print(f"Targets: {len(loss_cols)}")
    print(f"Outer folds: {outer_folds}; inner folds: {inner_folds}")
    print("External test/extrapolation sets used for fitting: NONE")
    print("\nModel-family summary:")
    print(summary.round(6).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nested-CV comparison of Q1 RegMix mixture-loss model families "
            "using A4/A5 train_1m only."
        )
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("artifacts/q1/preprocessed"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1/regmix_nested_cv.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_cv"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_regmix_nested_cv(
        preprocessed_dir=args.preprocessed_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )

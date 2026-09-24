from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.q1.fit_regmix_cv import (
    SimplexOrthogonalTransformer,
    ZeroReplacedILRTransformer,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid final RegMix config: {path}")
    return cfg


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _build_estimator(
    family: str,
    cfg: dict[str, Any],
) -> tuple[Pipeline, dict[str, list[Any]]]:
    ridge_alphas = [float(v) for v in cfg["ridge_alphas"]]

    if family == "ilr_ridge":
        epsilons = [float(v) for v in cfg["ilr_epsilons"]]
        estimator = Pipeline(
            [
                ("ilr", ZeroReplacedILRTransformer()),
                ("scale", StandardScaler()),
                ("model", Ridge()),
            ]
        )
        grid = {
            "ilr__epsilon": epsilons,
            "model__alpha": ridge_alphas,
        }
        return estimator, grid

    if family == "quadratic_ridge_simplex":
        estimator = Pipeline(
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
        )
        grid = {"model__alpha": ridge_alphas}
        return estimator, grid

    raise ValueError(f"Unsupported frozen family: {family}")


def run_freeze_regmix_models(
    preprocessed_dir: Path,
    regmix_cv_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(config_path)

    train_path = preprocessed_dir / "regmix_train_1m_clean.csv"
    train = pd.read_csv(train_path)

    p_cols = sorted(
        c for c in train.columns
        if c.startswith("p_") and not c.startswith("p_raw_")
    )
    loss_cols = sorted(c for c in train.columns if c.startswith("loss_"))

    if len(p_cols) != 17 or len(loss_cols) != 13:
        raise ValueError(
            f"Expected 17 p columns and 13 loss targets, got "
            f"{len(p_cols)} and {len(loss_cols)}"
        )

    X = train[p_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite mixture values in train_1m.")
    if np.max(np.abs(X.sum(axis=1) - 1.0)) >= 1e-12:
        raise ValueError("train_1m mixture rows violate simplex constraint.")

    family_map = dict(cfg["target_family"])
    if set(family_map) != set(loss_cols):
        missing = sorted(set(loss_cols) - set(family_map))
        extra = sorted(set(family_map) - set(loss_cols))
        raise ValueError(
            f"target_family must specify exactly all 13 targets; "
            f"missing={missing}, extra={extra}"
        )

    # Verify that the family freeze exactly matches Step-10 nested-CV winners.
    best_path = regmix_cv_dir / "best_family_by_target.csv"
    best = pd.read_csv(best_path)
    observed = dict(zip(best["target"], best["family"]))

    mismatches = {
        target: {
            "configured": family_map[target],
            "step10_nested_cv": observed.get(target),
        }
        for target in loss_cols
        if observed.get(target) != family_map[target]
    }
    if mismatches:
        raise ValueError(
            "Frozen target-family map does not match Step-10 nested-CV "
            f"winners: {mismatches}"
        )

    cv_folds = int(cfg.get("final_tuning_folds", 5))
    seed = int(cfg.get("random_seed", 20260924))
    n_jobs = int(cfg.get("n_jobs", -1))

    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )

    summary_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    model_hashes: dict[str, str] = {}

    for target in loss_cols:
        y = pd.to_numeric(train[target], errors="raise").to_numpy(dtype=float)
        family = family_map[target]

        estimator, param_grid = _build_estimator(family, cfg)

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X, y)

        best_index = int(search.best_index_)
        best_rmse = float(np.sqrt(-search.best_score_))

        model_path = model_dir / f"{target}.joblib"
        joblib.dump(search.best_estimator_, model_path)
        model_hashes[target] = _sha256(model_path)

        row = {
            "target": target,
            "family": family,
            "n": int(len(y)),
            "train_target_mean": float(y.mean()),
            "train_target_sd": float(y.std(ddof=1)),
            "cv_rmse_selected": best_rmse,
            "cv_nrmse_selected": (
                float(best_rmse / y.std(ddof=1))
                if y.std(ddof=1) > 0
                else np.nan
            ),
            "model_file": str(model_path.relative_to(output_dir)),
            "model_sha256": model_hashes[target],
        }
        for key, value in search.best_params_.items():
            row[key] = value
        summary_rows.append(row)

        results = pd.DataFrame(search.cv_results_)
        for _, r in results.iterrows():
            cv_row = {
                "target": target,
                "family": family,
                "mean_test_mse": float(-r["mean_test_score"]),
                "std_test_mse": float(r["std_test_score"]),
                "mean_test_rmse_approx": float(
                    np.sqrt(max(0.0, -r["mean_test_score"]))
                ),
                "rank_test_score": int(r["rank_test_score"]),
            }
            params = r["params"]
            for key, value in params.items():
                cv_row[key] = value
            cv_rows.append(cv_row)

        print(
            f"Frozen {target}: family={family}, "
            f"best_params={search.best_params_}, "
            f"CV_RMSE={best_rmse:.6f}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "frozen_target_models.csv",
        index=False,
    )
    pd.DataFrame(cv_rows).to_csv(
        output_dir / "final_tuning_cv_results.csv",
        index=False,
    )

    family_counts = summary["family"].value_counts().to_dict()

    manifest = {
        "status": "FROZEN_BEFORE_EXTERNAL_TEST",
        "training_data": str(train_path),
        "training_data_sha256": _sha256(train_path),
        "rows": int(len(train)),
        "mixture_columns": p_cols,
        "loss_targets": loss_cols,
        "family_selection_source": (
            "Step-10 nested CV on A4/A5 only; Step-10b epsilon sensitivity "
            "was diagnostic and did not change the target-family assignments"
        ),
        "target_family": family_map,
        "family_counts": family_counts,
        "final_hyperparameter_tuning": {
            "data": "A4/A5 train_1m only",
            "cv": f"{cv_folds}-fold shuffled KFold",
            "random_seed": seed,
            "scoring": "negative mean squared error",
            "ridge_alphas": [float(v) for v in cfg["ridge_alphas"]],
            "ilr_epsilons": [float(v) for v in cfg["ilr_epsilons"]],
            "note": (
                "Per-target hyperparameters are selected only on A4/A5. "
                "The expanded epsilon grid is frozen before opening A6-A11."
            ),
        },
        "external_sets_seen_by_this_step": [],
        "external_test_policy": (
            "After this manifest is created, A6-A11 may be evaluated once. "
            "Do not alter target families or tuning grids in response to "
            "external-test results."
        ),
        "model_sha256": model_hashes,
    }

    manifest_path = output_dir / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFinal RegMix models frozen.")
    print(f"Output directory: {output_dir}")
    print(f"Family counts: {family_counts}")
    print("External sets used: NONE")
    print("Status: FROZEN_BEFORE_EXTERNAL_TEST")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the final Q1 target-wise RegMix model family and "
            "hyperparameters using A4/A5 only, before opening A6-A11."
        )
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("artifacts/q1/preprocessed"),
    )
    parser.add_argument(
        "--regmix-cv-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_cv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1/regmix_final.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_final"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_freeze_regmix_models(
        preprocessed_dir=args.preprocessed_dir,
        regmix_cv_dir=args.regmix_cv_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )

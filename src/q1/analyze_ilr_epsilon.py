from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.q1.fit_regmix_cv import ZeroReplacedILRTransformer


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid ILR sensitivity config: {path}")
    return cfg


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


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
        "spearman": _spearman(y_true, y_pred),
    }


def run_ilr_epsilon_sensitivity(
    preprocessed_dir: Path,
    regmix_cv_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(config_path)
    df = pd.read_csv(preprocessed_dir / "regmix_train_1m_clean.csv")

    p_cols = sorted(
        c for c in df.columns
        if c.startswith("p_") and not c.startswith("p_raw_")
    )
    loss_cols = sorted(c for c in df.columns if c.startswith("loss_"))

    if len(p_cols) != 17 or len(loss_cols) != 13:
        raise ValueError(
            f"Expected 17 mixture columns and 13 loss columns, got "
            f"{len(p_cols)} and {len(loss_cols)}"
        )

    X = df[p_cols].to_numpy(dtype=float)
    Y = df[loss_cols].to_numpy(dtype=float)

    epsilons = [float(v) for v in cfg["epsilons"]]
    alphas = [float(v) for v in cfg["ridge_alphas"]]
    outer_folds = int(cfg.get("outer_folds", 5))
    inner_folds = int(cfg.get("inner_folds", 4))
    seed = int(cfg.get("random_seed", 20260924))
    n_jobs = int(cfg.get("n_jobs", -1))
    primary_max = float(cfg.get("primary_zero_replacement_max", 1e-3))

    outer_cv = KFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=seed,
    )

    target_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []

    for target_idx, target in enumerate(loss_cols):
        y = Y[:, target_idx]
        target_sd = float(y.std(ddof=1))

        for eps in epsilons:
            oof = np.full(len(df), np.nan, dtype=float)

            for fold_idx, (train_idx, valid_idx) in enumerate(
                outer_cv.split(X),
                start=1,
            ):
                Xtr, Xva = X[train_idx], X[valid_idx]
                ytr, yva = y[train_idx], y[valid_idx]

                inner_cv = KFold(
                    n_splits=inner_folds,
                    shuffle=True,
                    random_state=seed + 1000 * target_idx + fold_idx,
                )

                pipe = Pipeline(
                    [
                        (
                            "ilr",
                            ZeroReplacedILRTransformer(epsilon=eps),
                        ),
                        ("scale", StandardScaler()),
                        ("model", Ridge()),
                    ]
                )

                search = GridSearchCV(
                    estimator=pipe,
                    param_grid={"model__alpha": alphas},
                    scoring="neg_mean_squared_error",
                    cv=inner_cv,
                    n_jobs=n_jobs,
                    refit=True,
                    error_score="raise",
                )
                search.fit(Xtr, ytr)

                pred = search.predict(Xva)
                oof[valid_idx] = pred

                fm = _metrics(yva, pred, target_sd)
                fold_rows.append(
                    {
                        "target": target,
                        "epsilon": eps,
                        "outer_fold": fold_idx,
                        "best_alpha": float(search.best_params_["model__alpha"]),
                        **fm,
                    }
                )
                param_rows.append(
                    {
                        "target": target,
                        "epsilon": eps,
                        "outer_fold": fold_idx,
                        "best_alpha": float(search.best_params_["model__alpha"]),
                    }
                )

            if not np.all(np.isfinite(oof)):
                raise RuntimeError(f"Missing OOF predictions for {target}, epsilon={eps}")

            tm = _metrics(y, oof, target_sd)
            target_rows.append(
                {
                    "target": target,
                    "epsilon": eps,
                    "within_primary_zero_replacement_range": bool(eps <= primary_max),
                    "target_mean": float(y.mean()),
                    "target_sd": target_sd,
                    **tm,
                }
            )

        print(f"Completed epsilon sensitivity {target_idx + 1}/{len(loss_cols)}: {target}")

    target_metrics = pd.DataFrame(target_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    selected_params = pd.DataFrame(param_rows)

    target_metrics.to_csv(
        output_dir / "epsilon_target_metrics.csv",
        index=False,
    )
    fold_metrics.to_csv(
        output_dir / "epsilon_fold_metrics.csv",
        index=False,
    )
    selected_params.to_csv(
        output_dir / "epsilon_selected_alpha.csv",
        index=False,
    )

    macro = (
        target_metrics.groupby(
            ["epsilon", "within_primary_zero_replacement_range"],
            as_index=False,
        )
        .agg(
            macro_rmse=("rmse", "mean"),
            macro_nrmse=("nrmse_by_train_sd", "mean"),
            median_nrmse=("nrmse_by_train_sd", "median"),
            macro_mae=("mae", "mean"),
            macro_r2=("r2", "mean"),
            macro_spearman=("spearman", "mean"),
        )
        .sort_values(["macro_nrmse", "epsilon"], kind="stable")
        .reset_index(drop=True)
    )
    macro.to_csv(output_dir / "epsilon_macro_summary.csv", index=False)

    best_by_target = (
        target_metrics
        .sort_values(
            ["target", "nrmse_by_train_sd", "epsilon"],
            kind="stable",
        )
        .groupby("target", as_index=False)
        .first()
        [
            [
                "target",
                "epsilon",
                "within_primary_zero_replacement_range",
                "nrmse_by_train_sd",
                "rmse",
                "r2",
                "spearman",
            ]
        ]
    )
    best_by_target.to_csv(
        output_dir / "epsilon_best_by_target.csv",
        index=False,
    )

    # Compare the best epsilon-only ILR curve to the already-computed
    # quadratic-ridge nested-CV result without touching external data.
    cv_metrics_path = regmix_cv_dir / "nested_cv_target_metrics.csv"
    if cv_metrics_path.exists():
        prev = pd.read_csv(cv_metrics_path)
        quad = prev[
            prev["family"] == "quadratic_ridge_simplex"
        ][["target", "nrmse_by_train_sd"]].rename(
            columns={"nrmse_by_train_sd": "quadratic_nrmse"}
        )

        comparison = best_by_target.merge(
            quad,
            on="target",
            how="left",
            validate="one_to_one",
        )
        comparison["delta_best_ILR_minus_quadratic"] = (
            comparison["nrmse_by_train_sd"]
            - comparison["quadratic_nrmse"]
        )
        comparison["best_ILR_beats_quadratic"] = (
            comparison["delta_best_ILR_minus_quadratic"] < 0
        )
        comparison.to_csv(
            output_dir / "best_ilr_vs_quadratic.csv",
            index=False,
        )

    boundary = {
        "min_epsilon": min(epsilons),
        "max_epsilon": max(epsilons),
        "best_global_epsilon": float(
            macro.sort_values("macro_nrmse").iloc[0]["epsilon"]
        ),
        "best_global_at_grid_boundary": bool(
            macro.sort_values("macro_nrmse").iloc[0]["epsilon"]
            in {min(epsilons), max(epsilons)}
        ),
        "targets_best_at_max_grid_epsilon": int(
            (best_by_target["epsilon"] == max(epsilons)).sum()
        ),
        "targets_best_within_primary_range": int(
            best_by_target["within_primary_zero_replacement_range"].sum()
        ),
    }

    manifest = {
        "data_used": "A4/A5 train_1m only",
        "external_sets_used": [],
        "purpose": (
            "Diagnose whether the Step-10 ILR advantage depends on the upper "
            "edge of the original epsilon grid."
        ),
        "epsilons": epsilons,
        "primary_zero_replacement_max": primary_max,
        "interpretation": (
            "epsilon <= primary_zero_replacement_max is the primary "
            "zero-replacement range; larger epsilon values are stress-test "
            "offsets used to detect boundary-driven behavior."
        ),
        "ridge_alphas": alphas,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "random_seed": seed,
        "boundary_diagnostics": boundary,
    }

    (output_dir / "ilr_epsilon_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("ILR epsilon sensitivity completed.")
    print(f"Output directory: {output_dir}")
    print("External test/extrapolation sets used: NONE")
    print("\nMacro epsilon summary:")
    print(macro.round(6).to_string(index=False))
    print("\nBoundary diagnostics:")
    for key, value in boundary.items():
        print(f"{key} = {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nested-CV sensitivity of the zero-replacement epsilon used by "
            "the Q1 ILR-Ridge RegMix model."
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
        default=Path("configs/q1/ilr_epsilon_sensitivity.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/ilr_epsilon_sensitivity"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_ilr_epsilon_sensitivity(
        preprocessed_dir=args.preprocessed_dir,
        regmix_cv_dir=args.regmix_cv_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )

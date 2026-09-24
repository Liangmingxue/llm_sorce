from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


EXTRAPOLATION_FILES = {
    "est_10b": "regmix_est_10b_clean.csv",
    "est_70b": "regmix_est_70b_clean.csv",
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
    y_source: np.ndarray,
    y_target: np.ndarray,
) -> tuple[float, float, float]:
    """
    Descriptive relation only:
        y_target ~= intercept + slope * y_source.
    """
    x = np.asarray(y_source, dtype=float)
    y = np.asarray(y_target, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    return (
        float(beta[0]),
        float(beta[1]),
        float(r2_score(y, fitted)),
    )


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


def _top_fraction_overlap(
    signatures: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    fraction: float = 0.20,
) -> float:
    n = len(a)
    k = max(1, int(np.ceil(fraction * n)))
    idx_a = np.argsort(a)[:k]
    idx_b = np.argsort(b)[:k]
    set_a = set(signatures[idx_a])
    set_b = set(signatures[idx_b])
    return float(len(set_a & set_b) / k)


def _pair_metrics(
    signatures: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    source_name: str,
    target_name: str,
) -> dict[str, Any]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    intercept, slope, affine_r2 = _affine_diagnostic(
        y_source=source,
        y_target=target,
    )

    mean_shift = float((target - source).mean())
    shifted_source = source + mean_shift
    centered_rmse = float(
        np.sqrt(mean_squared_error(target, shifted_source))
    )
    target_sd = float(target.std(ddof=1))

    return {
        "source_scale": source_name,
        "target_scale": target_name,
        "n": int(len(source)),
        "source_mean": float(source.mean()),
        "target_mean": float(target.mean()),
        "mean_shift_target_minus_source": mean_shift,
        "pearson": _safe_corr(source, target, "pearson"),
        "spearman": _safe_corr(source, target, "spearman"),
        "kendall": _safe_corr(source, target, "kendall"),
        "affine_intercept": intercept,
        "affine_slope": slope,
        "affine_r2": affine_r2,
        "centered_rmse": centered_rmse,
        "centered_nrmse_by_target_sd": (
            float(centered_rmse / target_sd)
            if target_sd > 0
            else np.nan
        ),
        "best_20pct_overlap": _top_fraction_overlap(
            signatures,
            source,
            target,
            fraction=0.20,
        ),
    }


def run_regmix_extrapolation(
    preprocessed_dir: Path,
    frozen_dir: Path,
    external_test_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_manifest_path = frozen_dir / "freeze_manifest.json"
    with freeze_manifest_path.open("r", encoding="utf-8") as f:
        freeze_manifest: dict[str, Any] = json.load(f)

    if freeze_manifest.get("status") != "FROZEN_BEFORE_EXTERNAL_TEST":
        raise ValueError(
            "Frozen manifest status is not FROZEN_BEFORE_EXTERNAL_TEST."
        )

    external_marker = external_test_dir / "EXTERNAL_TEST_EVALUATED.json"
    if not external_marker.exists():
        raise RuntimeError(
            "A6-A11 external validation must be completed before A12-A15 "
            "extrapolation analysis."
        )

    p_cols = list(freeze_manifest["mixture_columns"])
    loss_targets = list(freeze_manifest["loss_targets"])
    family_map = dict(freeze_manifest["target_family"])
    expected_hashes = dict(freeze_manifest["model_sha256"])

    train_path = preprocessed_dir / "regmix_train_1m_clean.csv"
    train = pd.read_csv(train_path)
    train["recipe_signature"] = _recipe_signature(train, p_cols)

    if train["recipe_signature"].duplicated().any():
        raise ValueError("Duplicate recipe signatures in train_1m.")

    est = {
        name: pd.read_csv(preprocessed_dir / filename)
        for name, filename in EXTRAPOLATION_FILES.items()
    }

    for name, df in est.items():
        df["recipe_signature"] = _recipe_signature(df, p_cols)
        if df["recipe_signature"].duplicated().any():
            raise ValueError(f"Duplicate recipe signatures in {name}.")
        if len(df) != 63:
            raise ValueError(
                f"Expected 63 rows in {name}, got {len(df)}."
            )

    # Verify the same 63 recipes are used at 10b and 70b.
    sig10 = set(est["est_10b"]["recipe_signature"])
    sig70 = set(est["est_70b"]["recipe_signature"])
    if sig10 != sig70:
        raise ValueError(
            "est_10b and est_70b do not contain identical recipe sets."
        )

    # Verify every extrapolation recipe is already present in 1m train.
    train_sig = set(train["recipe_signature"])
    if not sig10.issubset(train_sig):
        missing = len(sig10 - train_sig)
        raise ValueError(
            f"{missing} extrapolation recipes are absent from train_1m."
        )

    # Match the observed 1m losses to the exact 63 extrapolation recipes.
    base = (
        est["est_10b"][["recipe_signature", *p_cols]]
        .merge(
            train[["recipe_signature", *loss_targets]],
            on="recipe_signature",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={target: f"{target}__1m_observed"
                     for target in loss_targets}
        )
    )

    ten = est["est_10b"][
        ["recipe_signature", *loss_targets]
    ].rename(
        columns={target: f"{target}__10b_est"
                 for target in loss_targets}
    )
    seventy = est["est_70b"][
        ["recipe_signature", *loss_targets]
    ].rename(
        columns={target: f"{target}__70b_est"
                 for target in loss_targets}
    )

    paired = (
        base.merge(
            ten,
            on="recipe_signature",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            seventy,
            on="recipe_signature",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(paired) != 63:
        raise ValueError(
            f"Expected 63 fully paired recipes, got {len(paired)}."
        )

    signatures = paired["recipe_signature"].to_numpy()

    pair_rows: list[dict[str, Any]] = []
    monotone_rows: list[dict[str, Any]] = []

    for target in loss_targets:
        y1 = paired[f"{target}__1m_observed"].to_numpy(dtype=float)
        y10 = paired[f"{target}__10b_est"].to_numpy(dtype=float)
        y70 = paired[f"{target}__70b_est"].to_numpy(dtype=float)

        for source, dest, source_name, dest_name in [
            (y1, y10, "1m_observed", "10b_estimated"),
            (y1, y70, "1m_observed", "70b_estimated"),
            (y10, y70, "10b_estimated", "70b_estimated"),
        ]:
            pair_rows.append(
                {
                    "target": target,
                    "family": family_map[target],
                    **_pair_metrics(
                        signatures=signatures,
                        source=source,
                        target=dest,
                        source_name=source_name,
                        target_name=dest_name,
                    ),
                }
            )

        monotone_rows.append(
            {
                "target": target,
                "family": family_map[target],
                "n": int(len(y1)),
                "fraction_10b_loss_below_1m": float(
                    np.mean(y10 < y1)
                ),
                "fraction_70b_loss_below_10b": float(
                    np.mean(y70 < y10)
                ),
                "fraction_strict_1m_gt_10b_gt_70b": float(
                    np.mean((y1 > y10) & (y10 > y70))
                ),
                "median_drop_1m_to_10b": float(
                    np.median(y1 - y10)
                ),
                "median_drop_10b_to_70b": float(
                    np.median(y10 - y70)
                ),
            }
        )

    pair_metrics = pd.DataFrame(pair_rows)
    pair_metrics.to_csv(
        output_dir / "extrapolation_pair_metrics.csv",
        index=False,
    )

    monotonicity = pd.DataFrame(monotone_rows)
    monotonicity.to_csv(
        output_dir / "extrapolation_monotonicity.csv",
        index=False,
    )

    summary = (
        pair_metrics.groupby(
            ["source_scale", "target_scale"],
            as_index=False,
        )
        .agg(
            targets=("target", "count"),
            macro_pearson=("pearson", "mean"),
            macro_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            macro_kendall=("kendall", "mean"),
            macro_affine_r2=("affine_r2", "mean"),
            median_affine_slope=("affine_slope", "median"),
            macro_centered_nrmse=(
                "centered_nrmse_by_target_sd",
                "mean",
            ),
            macro_best_20pct_overlap=(
                "best_20pct_overlap",
                "mean",
            ),
        )
    )
    summary.to_csv(
        output_dir / "extrapolation_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Frozen model vs A12-A15 estimated losses.
    # This is descriptive only because the 63 recipes are a subset
    # of the A4/A5 training recipes.
    # ----------------------------------------------------------
    frozen_table = pd.read_csv(
        frozen_dir / "frozen_target_models.csv"
    )
    model_paths = dict(
        zip(
            frozen_table["target"],
            frozen_table["model_file"],
        )
    )

    model_rows: list[dict[str, Any]] = []

    X63 = paired[p_cols].to_numpy(dtype=float)

    for target in loss_targets:
        model_path = frozen_dir / model_paths[target]
        actual_hash = _sha256(model_path)
        if actual_hash != expected_hashes[target]:
            raise ValueError(
                f"Frozen model hash mismatch for {target}."
            )

        model = joblib.load(model_path)
        pred = np.asarray(model.predict(X63), dtype=float)

        for scale_name, col in [
            ("10b_estimated", f"{target}__10b_est"),
            ("70b_estimated", f"{target}__70b_est"),
        ]:
            y = paired[col].to_numpy(dtype=float)
            intercept, slope, affine_r2 = _affine_diagnostic(
                y_source=pred,
                y_target=y,
            )
            model_rows.append(
                {
                    "target": target,
                    "family": family_map[target],
                    "target_scale": scale_name,
                    "n": int(len(y)),
                    "spearman_frozen_prediction_vs_estimate":
                        _safe_corr(pred, y, "spearman"),
                    "pearson_frozen_prediction_vs_estimate":
                        _safe_corr(pred, y, "pearson"),
                    "affine_slope": slope,
                    "affine_intercept": intercept,
                    "affine_r2": affine_r2,
                    "best_20pct_overlap": _top_fraction_overlap(
                        signatures,
                        pred,
                        y,
                        fraction=0.20,
                    ),
                    "independent_test_claim": False,
                }
            )

    model_transfer = pd.DataFrame(model_rows)
    model_transfer.to_csv(
        output_dir / "frozen_model_extrapolation_transfer.csv",
        index=False,
    )

    model_summary = (
        model_transfer.groupby(
            "target_scale",
            as_index=False,
        )
        .agg(
            targets=("target", "count"),
            macro_spearman=(
                "spearman_frozen_prediction_vs_estimate",
                "mean",
            ),
            macro_pearson=(
                "pearson_frozen_prediction_vs_estimate",
                "mean",
            ),
            macro_affine_r2=("affine_r2", "mean"),
            macro_best_20pct_overlap=(
                "best_20pct_overlap",
                "mean",
            ),
        )
    )
    model_summary.to_csv(
        output_dir / "frozen_model_extrapolation_summary.csv",
        index=False,
    )

    monotone_summary = {
        "macro_fraction_10b_loss_below_1m": float(
            monotonicity["fraction_10b_loss_below_1m"].mean()
        ),
        "macro_fraction_70b_loss_below_10b": float(
            monotonicity["fraction_70b_loss_below_10b"].mean()
        ),
        "macro_fraction_strict_1m_gt_10b_gt_70b": float(
            monotonicity[
                "fraction_strict_1m_gt_10b_gt_70b"
            ].mean()
        ),
    }

    marker = {
        "status": "EXTRAPOLATION_ANALYZED",
        "a12_a15_role": (
            "Estimated/extrapolation tables used for robustness discussion; "
            "not independent experimental observations."
        ),
        "recipe_overlap": {
            "rows_10b": int(len(est["est_10b"])),
            "rows_70b": int(len(est["est_70b"])),
            "same_10b_70b_recipe_set": True,
            "all_extrapolation_recipes_in_train_1m": True,
            "independent_external_test": False,
        },
        "frozen_models_modified": False,
        "external_test_model_selection_reopened": False,
        "freeze_manifest_sha256": _sha256(freeze_manifest_path),
        "input_sha256": {
            "train_1m": _sha256(train_path),
            **{
                name: _sha256(preprocessed_dir / filename)
                for name, filename in EXTRAPOLATION_FILES.items()
            },
        },
        "monotonicity_summary": monotone_summary,
        "interpretation_policy": (
            "Use rank preservation, affine shape transfer, top-quintile "
            "overlap and monotonic loss reduction as robustness evidence. "
            "Do not report A12-A15 as independent test accuracy and do not "
            "retune the frozen model from these results."
        ),
    }

    (output_dir / "EXTRAPOLATION_ANALYZED.json").write_text(
        json.dumps(marker, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("RegMix A12-A15 extrapolation analysis completed.")
    print(f"Output directory: {output_dir}")
    print("Frozen models modified: NO")
    print("A12-A15 treated as independent experimental tests: NO")
    print("Status: EXTRAPOLATION_ANALYZED")
    print("\nExtrapolation pair summary:")
    print(summary.round(6).to_string(index=False))
    print("\nFrozen-model transfer to extrapolation tables:")
    print(model_summary.round(6).to_string(index=False))
    print("\nMonotonicity summary:")
    for key, value in monotone_summary.items():
        print(f"{key} = {value:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Q1 A12-A15 extrapolation stability without reopening "
            "model selection."
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
        "--external-test-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_external_test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_extrapolation"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_regmix_extrapolation(
        preprocessed_dir=args.preprocessed_dir,
        frozen_dir=args.frozen_dir,
        external_test_dir=args.external_test_dir,
        output_dir=args.output_dir,
    )

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_frozen(
    frozen_dir: Path,
) -> tuple[
    list[str],
    list[str],
    dict[str, Any],
    dict[str, float],
    dict[str, float],
    dict[str, str],
]:
    manifest_path = frozen_dir / "freeze_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("status") != "FROZEN_BEFORE_EXTERNAL_TEST":
        raise ValueError(
            "freeze_manifest.json is not in FROZEN_BEFORE_EXTERNAL_TEST state."
        )

    table = pd.read_csv(frozen_dir / "frozen_target_models.csv")
    p_cols = list(manifest["mixture_columns"])
    targets = list(manifest["loss_targets"])
    expected_hashes = dict(manifest["model_sha256"])

    models: dict[str, Any] = {}
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    families: dict[str, str] = {}

    for _, row in table.iterrows():
        target = str(row["target"])
        model_path = frozen_dir / str(row["model_file"])
        if _sha256(model_path) != expected_hashes[target]:
            raise ValueError(f"Frozen model hash mismatch: {target}")
        models[target] = joblib.load(model_path)
        means[target] = float(row["train_target_mean"])
        sds[target] = float(row["train_target_sd"])
        families[target] = str(row["family"])

    if set(models) != set(targets):
        raise ValueError("Frozen model table does not contain all targets.")

    return p_cols, targets, models, means, sds, families


def _predict_matrix(
    X: np.ndarray,
    targets: list[str],
    models: dict[str, Any],
) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(models[t].predict(X), dtype=float)
            for t in targets
        ]
    )


def _standardized_objective(
    pred: np.ndarray,
    targets: list[str],
    means: dict[str, float],
    sds: dict[str, float],
) -> np.ndarray:
    mu = np.asarray([means[t] for t in targets], dtype=float)
    sd = np.asarray([sds[t] for t in targets], dtype=float)
    z = (pred - mu[None, :]) / sd[None, :]
    return z.mean(axis=1)


def _bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])

    # Chunked bootstrap to avoid a large temporary array.
    means = np.empty(draws, dtype=float)
    chunk = 250
    pos = 0
    while pos < draws:
        m = min(chunk, draws - pos)
        idx = rng.integers(0, n, size=(m, n))
        means[pos : pos + m] = values[idx].mean(axis=1)
        pos += m

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _enrich_one(
    X: np.ndarray,
    j: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add delta mass to component j and reduce every other component
    proportionally. Returns (perturbed_X, feasible_mask).
    """
    X = np.asarray(X, dtype=float)
    donor_mass = 1.0 - X[:, j]
    feasible = donor_mass >= delta - 1e-12

    out = X[feasible].copy()
    donor = donor_mass[feasible]

    scale = np.divide(
        donor - delta,
        donor,
        out=np.zeros_like(donor),
        where=donor > 0,
    )

    mask = np.ones(X.shape[1], dtype=bool)
    mask[j] = False
    out[:, mask] *= scale[:, None]
    out[:, j] += delta

    return out, feasible


def _substitute(
    X: np.ndarray,
    receiver: int,
    donor: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transfer exactly delta mass from donor to receiver.
    """
    X = np.asarray(X, dtype=float)
    feasible = (
        (X[:, donor] >= delta - 1e-12)
        & (X[:, receiver] <= 1.0 - delta + 1e-12)
    )

    out = X[feasible].copy()
    out[:, receiver] += delta
    out[:, donor] -= delta
    return out, feasible


def _pair_enrichment_variants(
    X: np.ndarray,
    j: int,
    k: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a common-donor interaction contrast.

    The donor pool excludes j and k. From the same base recipe:
      single_j: +delta to j, k unchanged
      single_k: +delta to k, j unchanged
      joint:    +delta to both j and k

    All mass is removed proportionally from the remaining 15 domains.
    """
    X = np.asarray(X, dtype=float)
    donor_idx = np.ones(X.shape[1], dtype=bool)
    donor_idx[[j, k]] = False

    donor_mass = X[:, donor_idx].sum(axis=1)
    feasible = donor_mass >= 2.0 * delta - 1e-12

    base = X[feasible].copy()
    donor = donor_mass[feasible]

    def build(add_j: float, add_k: float) -> np.ndarray:
        out = base.copy()
        total_add = add_j + add_k
        scale = (donor - total_add) / donor
        out[:, donor_idx] *= scale[:, None]
        out[:, j] += add_j
        out[:, k] += add_k
        return out

    single_j = build(delta, 0.0)
    single_k = build(0.0, delta)
    joint = build(delta, delta)

    return base, single_j, single_k, joint, feasible


def run_regmix_effect_analysis(
    preprocessed_dir: Path,
    frozen_dir: Path,
    external_test_dir: Path,
    extrapolation_dir: Path,
    output_dir: Path,
    deltas: tuple[float, ...] = (0.01, 0.02),
    bootstrap_draws: int = 1000,
    random_seed: int = 20260924,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enforce the intended workflow: effect interpretation happens only
    # after frozen external validation and extrapolation diagnostics.
    ext_marker = external_test_dir / "EXTERNAL_TEST_EVALUATED.json"
    extra_marker = extrapolation_dir / "EXTRAPOLATION_ANALYZED.json"
    if not ext_marker.exists():
        raise RuntimeError("Step 11 external validation marker is missing.")
    if not extra_marker.exists():
        raise RuntimeError("Step 12 extrapolation marker is missing.")

    (
        p_cols,
        targets,
        models,
        means,
        sds,
        families,
    ) = _load_frozen(frozen_dir)

    train_path = preprocessed_dir / "regmix_train_1m_clean.csv"
    train = pd.read_csv(train_path)
    X = train[p_cols].to_numpy(dtype=float)

    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite values in train mixture matrix.")
    if np.max(np.abs(X.sum(axis=1) - 1.0)) >= 1e-12:
        raise ValueError("Training mixture rows violate simplex constraint.")

    base_pred = _predict_matrix(X, targets, models)
    base_obj = _standardized_objective(
        base_pred,
        targets,
        means,
        sds,
    )

    rng = np.random.default_rng(random_seed)

    domain_rows: list[dict[str, Any]] = []
    domain_target_rows: list[dict[str, Any]] = []

    for delta in deltas:
        for j, p_col in enumerate(p_cols):
            Xp, feasible = _enrich_one(X, j, delta)
            base_idx = np.flatnonzero(feasible)

            pred_p = _predict_matrix(Xp, targets, models)
            obj_p = _standardized_objective(
                pred_p,
                targets,
                means,
                sds,
            )
            diff = obj_p - base_obj[base_idx]

            lo, hi = _bootstrap_ci(
                diff,
                rng=rng,
                draws=bootstrap_draws,
            )

            domain_rows.append(
                {
                    "delta": delta,
                    "domain": p_col.removeprefix("p_"),
                    "feasible_n": int(len(diff)),
                    "mean_delta_objective": float(diff.mean()),
                    "median_delta_objective": float(np.median(diff)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "effect_per_1pct_mass": float(
                        diff.mean() * (0.01 / delta)
                    ),
                    "beneficial_if_negative": bool(diff.mean() < 0),
                }
            )

            normalized_target_change = (
                pred_p - base_pred[base_idx]
            ) / np.asarray([sds[t] for t in targets])[None, :]

            for t_idx, target in enumerate(targets):
                vals = normalized_target_change[:, t_idx]
                domain_target_rows.append(
                    {
                        "delta": delta,
                        "domain": p_col.removeprefix("p_"),
                        "target": target,
                        "family": families[target],
                        "feasible_n": int(len(vals)),
                        "mean_normalized_loss_change": float(
                            vals.mean()
                        ),
                        "median_normalized_loss_change": float(
                            np.median(vals)
                        ),
                    }
                )

    domain_effects = pd.DataFrame(domain_rows)
    domain_effects.to_csv(
        output_dir / "domain_enrichment_effects.csv",
        index=False,
    )

    pd.DataFrame(domain_target_rows).to_csv(
        output_dir / "domain_enrichment_by_target.csv",
        index=False,
    )

    # Rank stability across the two perturbation magnitudes.
    rank_stability = np.nan
    if len(deltas) >= 2:
        d0 = (
            domain_effects[domain_effects["delta"] == deltas[0]]
            .set_index("domain")["effect_per_1pct_mass"]
        )
        d1 = (
            domain_effects[domain_effects["delta"] == deltas[1]]
            .set_index("domain")["effect_per_1pct_mass"]
        )
        common = d0.index.intersection(d1.index)
        rank_stability = float(
            spearmanr(d0.loc[common], d1.loc[common]).statistic
        )

    # ----------------------------------------------------------
    # Ordered pairwise substitution: donor -> receiver.
    # A negative change means moving one percentage point of mixture
    # from donor to receiver improves the equal-target standardized loss.
    # ----------------------------------------------------------
    delta_pair = float(deltas[0])
    substitution_rows: list[dict[str, Any]] = []

    for receiver, rec_col in enumerate(p_cols):
        for donor, don_col in enumerate(p_cols):
            if receiver == donor:
                continue

            Xp, feasible = _substitute(
                X,
                receiver=receiver,
                donor=donor,
                delta=delta_pair,
            )
            base_idx = np.flatnonzero(feasible)
            if len(base_idx) == 0:
                continue

            pred_p = _predict_matrix(Xp, targets, models)
            obj_p = _standardized_objective(
                pred_p,
                targets,
                means,
                sds,
            )
            diff = obj_p - base_obj[base_idx]
            lo, hi = _bootstrap_ci(
                diff,
                rng=rng,
                draws=bootstrap_draws,
            )

            substitution_rows.append(
                {
                    "delta": delta_pair,
                    "receiver_domain": rec_col.removeprefix("p_"),
                    "donor_domain": don_col.removeprefix("p_"),
                    "feasible_n": int(len(diff)),
                    "mean_delta_objective": float(diff.mean()),
                    "median_delta_objective": float(np.median(diff)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "beneficial_if_negative": bool(diff.mean() < 0),
                }
            )

    substitutions = pd.DataFrame(substitution_rows)
    substitutions.to_csv(
        output_dir / "pairwise_substitution_effects.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Unordered pair interaction using a common donor pool.
    #
    # interaction = J(j+k) - J(j) - J(k) + J(base)
    #
    # negative -> synergy: joint enrichment is better than the sum of
    # separate enrichment effects.
    # positive -> antagonism/redundancy.
    # ----------------------------------------------------------
    interaction_rows: list[dict[str, Any]] = []

    for j in range(len(p_cols)):
        for k in range(j + 1, len(p_cols)):
            (
                base_pair,
                Xj,
                Xk,
                Xjk,
                feasible,
            ) = _pair_enrichment_variants(
                X,
                j=j,
                k=k,
                delta=delta_pair,
            )

            base_idx = np.flatnonzero(feasible)
            if len(base_idx) == 0:
                continue

            obj_base = base_obj[base_idx]
            obj_j = _standardized_objective(
                _predict_matrix(Xj, targets, models),
                targets,
                means,
                sds,
            )
            obj_k = _standardized_objective(
                _predict_matrix(Xk, targets, models),
                targets,
                means,
                sds,
            )
            obj_jk = _standardized_objective(
                _predict_matrix(Xjk, targets, models),
                targets,
                means,
                sds,
            )

            interaction = obj_jk - obj_j - obj_k + obj_base
            lo, hi = _bootstrap_ci(
                interaction,
                rng=rng,
                draws=bootstrap_draws,
            )

            interaction_rows.append(
                {
                    "delta_each": delta_pair,
                    "domain_a": p_cols[j].removeprefix("p_"),
                    "domain_b": p_cols[k].removeprefix("p_"),
                    "feasible_n": int(len(interaction)),
                    "mean_interaction": float(interaction.mean()),
                    "median_interaction": float(
                        np.median(interaction)
                    ),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "synergy_if_negative": bool(
                        interaction.mean() < 0
                    ),
                }
            )

    interactions = pd.DataFrame(interaction_rows)
    interactions.to_csv(
        output_dir / "pairwise_interaction_effects.csv",
        index=False,
    )

    # Empirical mixture summary for interpretability.
    mixture_summary = pd.DataFrame(
        {
            "domain": [c.removeprefix("p_") for c in p_cols],
            "mean_p": X.mean(axis=0),
            "median_p": np.median(X, axis=0),
            "p95": np.quantile(X, 0.95, axis=0),
            "max_p": X.max(axis=0),
            "zero_fraction": (X == 0).mean(axis=0),
        }
    )
    mixture_summary.to_csv(
        output_dir / "training_mixture_support.csv",
        index=False,
    )

    primary = domain_effects[
        domain_effects["delta"] == delta_pair
    ].sort_values(
        "mean_delta_objective",
        kind="stable",
    )

    top_sub = substitutions.sort_values(
        "mean_delta_objective",
        kind="stable",
    )
    top_syn = interactions.sort_values(
        "mean_interaction",
        kind="stable",
    )
    top_ant = interactions.sort_values(
        "mean_interaction",
        ascending=False,
        kind="stable",
    )

    manifest = {
        "status": "REGMIX_EFFECTS_ANALYZED",
        "data_used": "A4/A5 train_1m compositions only for perturbation references",
        "models": "Step-10c frozen target-wise models",
        "external_test_data_used_for_effect_estimation": False,
        "extrapolation_data_used_for_effect_estimation": False,
        "objective": (
            "Equal-weight mean across 13 predicted losses after each loss is "
            "standardized by its A4/A5 training mean and standard deviation."
        ),
        "single_domain_operator": (
            "Add delta mass to one domain; remove the same total mass "
            "proportionally from all other domains."
        ),
        "pairwise_substitution_operator": (
            "Transfer exactly 0.01 mixture mass from donor to receiver."
        ),
        "interaction_definition": (
            "Common-donor finite-difference contrast: "
            "J(j+k)-J(j)-J(k)+J(base). Negative denotes synergy."
        ),
        "deltas": list(deltas),
        "bootstrap_draws": bootstrap_draws,
        "random_seed": random_seed,
        "domain_effect_rank_spearman_across_deltas": rank_stability,
        "interpretation": (
            "These are model-based local/global-average composition effects, "
            "not causal effects. They respect the simplex by explicit mass "
            "redistribution."
        ),
    }

    (output_dir / "REGMIX_EFFECTS_ANALYZED.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("RegMix composition-effect analysis completed.")
    print(f"Output directory: {output_dir}")
    print("External A6-A11 labels used for effect estimation: NO")
    print("A12-A15 used for effect estimation: NO")
    print("Status: REGMIX_EFFECTS_ANALYZED")
    print(
        f"Domain-effect rank Spearman ({deltas[0]:.3f} vs "
        f"{deltas[1]:.3f}) = {rank_stability:.6f}"
    )

    show_cols = [
        "domain",
        "feasible_n",
        "mean_delta_objective",
        "ci95_low",
        "ci95_high",
        "effect_per_1pct_mass",
    ]

    print("\nMost beneficial 1%-mass enrichments:")
    print(primary.head(8)[show_cols].round(6).to_string(index=False))

    print("\nMost harmful 1%-mass enrichments:")
    print(
        primary.tail(8)
        .sort_values("mean_delta_objective", ascending=False)
        [show_cols]
        .round(6)
        .to_string(index=False)
    )

    print("\nStrongest beneficial substitutions (donor -> receiver):")
    print(
        top_sub.head(10)[
            [
                "donor_domain",
                "receiver_domain",
                "feasible_n",
                "mean_delta_objective",
                "ci95_low",
                "ci95_high",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\nStrongest synergies:")
    print(
        top_syn.head(10)[
            [
                "domain_a",
                "domain_b",
                "feasible_n",
                "mean_interaction",
                "ci95_low",
                "ci95_high",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\nStrongest antagonisms/redundancies:")
    print(
        top_ant.head(10)[
            [
                "domain_a",
                "domain_b",
                "feasible_n",
                "mean_interaction",
                "ci95_low",
                "ci95_high",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze single-domain, substitution and pairwise interaction "
            "effects for the frozen Q1 RegMix model."
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
        "--extrapolation-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_extrapolation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_effects"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_regmix_effect_analysis(
        preprocessed_dir=args.preprocessed_dir,
        frozen_dir=args.frozen_dir,
        external_test_dir=args.external_test_dir,
        extrapolation_dir=args.extrapolation_dir,
        output_dir=args.output_dir,
    )

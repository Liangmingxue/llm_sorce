from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


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
]:
    manifest_path = frozen_dir / "freeze_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("status") != "FROZEN_BEFORE_EXTERNAL_TEST":
        raise ValueError("Unexpected freeze manifest status.")

    table = pd.read_csv(frozen_dir / "frozen_target_models.csv")
    p_cols = list(manifest["mixture_columns"])
    targets = list(manifest["loss_targets"])
    expected_hashes = dict(manifest["model_sha256"])

    models: dict[str, Any] = {}
    means: dict[str, float] = {}
    sds: dict[str, float] = {}

    for _, row in table.iterrows():
        target = str(row["target"])
        path = frozen_dir / str(row["model_file"])
        if _sha256(path) != expected_hashes[target]:
            raise ValueError(f"Frozen model hash mismatch: {target}")
        models[target] = joblib.load(path)
        means[target] = float(row["train_target_mean"])
        sds[target] = float(row["train_target_sd"])

    return p_cols, targets, models, means, sds


def _predict(
    X: np.ndarray,
    targets: list[str],
    models: dict[str, Any],
) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.column_stack(
        [
            np.asarray(models[t].predict(X), dtype=float)
            for t in targets
        ]
    )


def _z_losses(
    pred: np.ndarray,
    targets: list[str],
    means: dict[str, float],
    sds: dict[str, float],
) -> np.ndarray:
    mu = np.asarray([means[t] for t in targets], dtype=float)
    sd = np.asarray([sds[t] for t in targets], dtype=float)
    return (pred - mu[None, :]) / sd[None, :]


def _objective_from_z(
    z: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "equal_mean":
        return z.mean(axis=1)
    if mode == "mean_plus_025sd":
        return z.mean(axis=1) + 0.25 * z.std(axis=1, ddof=0)
    if mode == "mean_plus_050sd":
        return z.mean(axis=1) + 0.50 * z.std(axis=1, ddof=0)
    raise ValueError(f"Unknown objective mode: {mode}")


def _hellinger_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # For probability vectors, Euclidean distance in sqrt-space divided
    # by sqrt(2) is the Hellinger distance.
    return cdist(np.sqrt(A), np.sqrt(B), metric="euclidean") / np.sqrt(2.0)


def _nearest_support_distance(
    X: np.ndarray,
    train_X: np.ndarray,
    batch_size: int = 5000,
) -> np.ndarray:
    out = np.empty(len(X), dtype=float)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        d = _hellinger_matrix(X[start:stop], train_X)
        out[start:stop] = d.min(axis=1)
    return out


def _make_local_candidate_bank(
    train_X: np.ndarray,
    neighbor_index: np.ndarray,
    samples_per_recipe: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate candidates inside local convex hulls of empirical recipes.

    Returns:
      candidate_p: [M, d]
      basis_seed:  [M] row index whose neighborhood defines the hull
      init_weight:[M, k] convex weights used to generate the candidate
    """
    rng = np.random.default_rng(seed)
    n, d = train_X.shape
    k = neighbor_index.shape[1]

    cand = []
    basis_seed = []
    init_weight = []

    # Always include the observed recipes themselves.
    for i in range(n):
        w = np.zeros(k, dtype=float)
        loc = np.flatnonzero(neighbor_index[i] == i)
        if len(loc) == 0:
            raise RuntimeError("Each neighborhood must include its seed.")
        w[int(loc[0])] = 1.0
        cand.append(train_X[i])
        basis_seed.append(i)
        init_weight.append(w)

    half = samples_per_recipe // 2
    remain = samples_per_recipe - half

    for i in range(n):
        basis = train_X[neighbor_index[i]]

        # alpha < 1 explores hull edges; alpha = 1 explores interiors.
        w_edge = rng.dirichlet(
            np.full(k, 0.35),
            size=half,
        )
        w_inner = rng.dirichlet(
            np.ones(k),
            size=remain,
        )
        W = np.vstack([w_edge, w_inner])
        P = W @ basis

        cand.append(P)
        basis_seed.extend([i] * len(P))
        init_weight.append(W)

    # Flatten mixed list of singleton rows / matrices.
    rows = []
    weights = []
    offset = 0
    for item, w in zip(cand, init_weight):
        arr = np.asarray(item, dtype=float)
        ww = np.asarray(w, dtype=float)
        if arr.ndim == 1:
            rows.append(arr[None, :])
            weights.append(ww[None, :])
        else:
            rows.append(arr)
            weights.append(ww)

    candidate_p = np.vstack(rows)
    init_w = np.vstack(weights)

    # Rebuild basis_seed in exactly the same order as rows above.
    seed_ids = np.concatenate(
        [
            np.arange(n, dtype=int),
            np.repeat(np.arange(n, dtype=int), samples_per_recipe),
        ]
    )

    if len(candidate_p) != len(seed_ids):
        raise RuntimeError("Candidate bank bookkeeping mismatch.")

    return candidate_p, seed_ids, init_w


def _refine_in_local_hull(
    basis: np.ndarray,
    w0: np.ndarray,
    targets: list[str],
    models: dict[str, Any],
    means: dict[str, float],
    sds: dict[str, float],
    objective_mode: str,
) -> tuple[np.ndarray, float, bool]:
    k = len(w0)

    def fun(w: np.ndarray) -> float:
        p = np.asarray(w, dtype=float) @ basis
        pred = _predict(p[None, :], targets, models)
        z = _z_losses(pred, targets, means, sds)
        return float(_objective_from_z(z, objective_mode)[0])

    cons = {
        "type": "eq",
        "fun": lambda w: float(np.sum(w) - 1.0),
    }
    res = minimize(
        fun,
        x0=np.asarray(w0, dtype=float),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[cons],
        options={
            "maxiter": 500,
            "ftol": 1e-12,
            "disp": False,
        },
    )

    w = np.asarray(res.x, dtype=float)
    w = np.clip(w, 0.0, 1.0)
    if w.sum() <= 0:
        w = np.asarray(w0, dtype=float)
    w /= w.sum()

    p = w @ basis
    value = fun(w)
    return p, value, bool(res.success)


def run_regmix_optimization(
    preprocessed_dir: Path,
    frozen_dir: Path,
    effects_dir: Path,
    output_dir: Path,
    neighbors: int = 12,
    samples_per_recipe: int = 80,
    random_seeds: tuple[int, ...] = (20260924, 20260925, 20260926),
    support_quantile: float = 0.95,
    top_refine: int = 40,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    effect_marker = effects_dir / "REGMIX_EFFECTS_ANALYZED.json"
    if not effect_marker.exists():
        raise RuntimeError("Step 13 effect-analysis marker is missing.")

    p_cols, targets, models, means, sds = _load_frozen(frozen_dir)

    train_path = preprocessed_dir / "regmix_train_1m_clean.csv"
    train = pd.read_csv(train_path)
    X = train[p_cols].to_numpy(dtype=float)

    if np.max(np.abs(X.sum(axis=1) - 1.0)) >= 1e-12:
        raise ValueError("Training mixtures violate simplex constraint.")

    # ----------------------------------------------------------
    # Empirical support geometry in Hellinger distance.
    # ----------------------------------------------------------
    D = _hellinger_matrix(X, X)
    np.fill_diagonal(D, np.inf)

    loo_nn = D.min(axis=1)
    support_threshold = float(
        np.quantile(loo_nn, support_quantile)
    )

    # Include self + nearest neighbors.
    D2 = _hellinger_matrix(X, X)
    neighbor_index = np.argsort(D2, axis=1)[:, :neighbors]

    support_summary = {
        "hellinger_loo_nn_median": float(np.median(loo_nn)),
        "hellinger_loo_nn_p90": float(np.quantile(loo_nn, 0.90)),
        "hellinger_loo_nn_p95": float(np.quantile(loo_nn, 0.95)),
        "hellinger_loo_nn_max": float(np.max(loo_nn)),
        "support_quantile": float(support_quantile),
        "support_threshold": support_threshold,
        "neighbors_per_local_hull": int(neighbors),
    }

    objective_modes = [
        "equal_mean",
        "mean_plus_025sd",
        "mean_plus_050sd",
    ]

    seed_rows: list[dict[str, Any]] = []
    all_candidates = []
    all_seed_ids = []
    all_init_w = []

    # Candidate generation is repeated under independent RNG seeds.
    for seed in random_seeds:
        cand, seed_ids, init_w = _make_local_candidate_bank(
            train_X=X,
            neighbor_index=neighbor_index,
            samples_per_recipe=samples_per_recipe,
            seed=seed,
        )

        d_support = _nearest_support_distance(cand, X)
        keep = d_support <= support_threshold + 1e-12

        cand = cand[keep]
        seed_ids = seed_ids[keep]
        init_w = init_w[keep]
        d_support = d_support[keep]

        pred = _predict(cand, targets, models)
        z = _z_losses(pred, targets, means, sds)
        score = _objective_from_z(z, "equal_mean")
        j = int(np.argmin(score))

        seed_rows.append(
            {
                "random_seed": seed,
                "candidate_count_after_support_filter": int(len(cand)),
                "best_equal_mean_objective_before_refine": float(score[j]),
                "best_support_distance": float(d_support[j]),
                **{
                    col.removeprefix("p_"): float(cand[j, q])
                    for q, col in enumerate(p_cols)
                },
            }
        )

        all_candidates.append(cand)
        all_seed_ids.append(seed_ids)
        all_init_w.append(init_w)

    candidate_p = np.vstack(all_candidates)
    candidate_seed_id = np.concatenate(all_seed_ids)
    candidate_init_w = np.vstack(all_init_w)

    # Deduplicate approximately identical mixture vectors.
    rounded = np.round(candidate_p, 12)
    _, unique_idx = np.unique(
        rounded,
        axis=0,
        return_index=True,
    )
    unique_idx = np.sort(unique_idx)

    candidate_p = candidate_p[unique_idx]
    candidate_seed_id = candidate_seed_id[unique_idx]
    candidate_init_w = candidate_init_w[unique_idx]

    candidate_pred = _predict(candidate_p, targets, models)
    candidate_z = _z_losses(
        candidate_pred,
        targets,
        means,
        sds,
    )

    solution_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for mode in objective_modes:
        candidate_score = _objective_from_z(candidate_z, mode)
        order = np.argsort(candidate_score)

        refined: list[tuple[np.ndarray, float, bool, int]] = []

        for idx in order[:top_refine]:
            seed_id = int(candidate_seed_id[idx])
            basis = X[neighbor_index[seed_id]]
            w0 = candidate_init_w[idx]

            p_ref, val_ref, ok = _refine_in_local_hull(
                basis=basis,
                w0=w0,
                targets=targets,
                models=models,
                means=means,
                sds=sds,
                objective_mode=mode,
            )

            support_dist = float(
                _nearest_support_distance(
                    p_ref[None, :],
                    X,
                )[0]
            )

            if support_dist <= support_threshold + 1e-12:
                refined.append(
                    (p_ref, val_ref, ok, seed_id)
                )

        if not refined:
            # Fall back to best support-filtered sampled candidate.
            idx = int(order[0])
            p_best = candidate_p[idx]
            value_best = float(candidate_score[idx])
            success = False
            seed_id = int(candidate_seed_id[idx])
        else:
            best = min(refined, key=lambda x: x[1])
            p_best, value_best, success, seed_id = best

        pred_best = _predict(
            p_best[None, :],
            targets,
            models,
        )[0]
        z_best = _z_losses(
            pred_best[None, :],
            targets,
            means,
            sds,
        )[0]

        support_dist = float(
            _nearest_support_distance(
                p_best[None, :],
                X,
            )[0]
        )

        solution_rows.append(
            {
                "objective_mode": mode,
                "objective_value": float(value_best),
                "optimizer_success": bool(success),
                "support_distance": support_dist,
                "support_threshold": support_threshold,
                "within_support": bool(
                    support_dist <= support_threshold + 1e-12
                ),
                "local_hull_seed_row": seed_id,
                "mixture_entropy": float(
                    -np.sum(
                        np.where(
                            p_best > 0,
                            p_best * np.log(p_best),
                            0.0,
                        )
                    )
                ),
                "effective_number_domains": float(
                    np.exp(
                        -np.sum(
                            np.where(
                                p_best > 0,
                                p_best * np.log(p_best),
                                0.0,
                            )
                        )
                    )
                ),
                **{
                    col.removeprefix("p_"): float(p_best[q])
                    for q, col in enumerate(p_cols)
                },
            }
        )

        for t_idx, target in enumerate(targets):
            target_rows.append(
                {
                    "objective_mode": mode,
                    "target": target,
                    "predicted_loss": float(pred_best[t_idx]),
                    "standardized_predicted_loss": float(z_best[t_idx]),
                }
            )

    solutions = pd.DataFrame(solution_rows)
    solutions.to_csv(
        output_dir / "support_aware_optimal_mixtures.csv",
        index=False,
    )

    pd.DataFrame(target_rows).to_csv(
        output_dir / "optimal_mixture_predicted_losses.csv",
        index=False,
    )

    pd.DataFrame(seed_rows).to_csv(
        output_dir / "search_seed_stability.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "recipe_row": np.arange(len(X)),
            "loo_nearest_hellinger": loo_nn,
        }
    ).to_csv(
        output_dir / "training_support_distances.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Baselines for the primary equal_mean objective.
    # ----------------------------------------------------------
    uniform = np.full(X.shape[1], 1.0 / X.shape[1])
    empirical_mean = X.mean(axis=0)

    train_pred = _predict(X, targets, models)
    train_z = _z_losses(
        train_pred,
        targets,
        means,
        sds,
    )
    train_score = _objective_from_z(train_z, "equal_mean")
    best_observed_idx = int(np.argmin(train_score))

    primary = solutions[
        solutions["objective_mode"] == "equal_mean"
    ].iloc[0]
    primary_p = np.asarray(
        [
            primary[c.removeprefix("p_")]
            for c in p_cols
        ],
        dtype=float,
    )

    baseline_points = {
        "uniform_17_domains": uniform,
        "empirical_mean_recipe": empirical_mean,
        "best_observed_training_recipe": X[best_observed_idx],
        "support_aware_optimized": primary_p,
    }

    baseline_rows = []
    for name, p in baseline_points.items():
        pred = _predict(p[None, :], targets, models)
        z = _z_losses(pred, targets, means, sds)
        value = float(_objective_from_z(z, "equal_mean")[0])
        baseline_rows.append(
            {
                "reference": name,
                "equal_mean_objective": value,
                "support_distance": float(
                    _nearest_support_distance(
                        p[None, :],
                        X,
                    )[0]
                ),
                **{
                    col.removeprefix("p_"): float(p[q])
                    for q, col in enumerate(p_cols)
                },
            }
        )

    baselines = pd.DataFrame(baseline_rows)
    best_obs_obj = float(
        baselines.loc[
            baselines["reference"]
            == "best_observed_training_recipe",
            "equal_mean_objective",
        ].iloc[0]
    )
    opt_obj = float(
        baselines.loc[
            baselines["reference"]
            == "support_aware_optimized",
            "equal_mean_objective",
        ].iloc[0]
    )

    baselines["improvement_vs_best_observed"] = (
        best_obs_obj - baselines["equal_mean_objective"]
    )
    baselines.to_csv(
        output_dir / "optimization_baselines.csv",
        index=False,
    )

    # Objective-sensitivity distances.
    mix_cols_out = [c.removeprefix("p_") for c in p_cols]
    sens_rows = []
    primary_vec = solutions.loc[
        solutions["objective_mode"] == "equal_mean",
        mix_cols_out,
    ].to_numpy(dtype=float)[0]

    for _, row in solutions.iterrows():
        vec = row[mix_cols_out].to_numpy(dtype=float)
        sens_rows.append(
            {
                "objective_mode": row["objective_mode"],
                "l1_distance_from_equal_mean": float(
                    np.abs(vec - primary_vec).sum()
                ),
                "max_abs_domain_shift": float(
                    np.max(np.abs(vec - primary_vec))
                ),
            }
        )

    sensitivity = pd.DataFrame(sens_rows)
    sensitivity.to_csv(
        output_dir / "objective_sensitivity.csv",
        index=False,
    )

    manifest = {
        "status": "SUPPORT_AWARE_MIXTURE_OPTIMIZED",
        "optimization_data": "A4/A5 train_1m composition support only",
        "frozen_models": "Step-10c target-wise models",
        "external_A6_A11_used_for_optimization": False,
        "A12_A15_used_for_optimization": False,
        "primary_objective": (
            "Equal-weight mean of 13 predicted losses standardized by "
            "A4/A5 target mean and standard deviation."
        ),
        "support_geometry": (
            "Hellinger distance on the 17-domain simplex. Candidate points "
            "are local convex combinations of empirical recipes and must have "
            "nearest empirical-recipe distance <= the 95th percentile of "
            "leave-one-out training nearest-neighbor distances."
        ),
        "neighbors": neighbors,
        "samples_per_recipe": samples_per_recipe,
        "random_seeds": list(random_seeds),
        "support_summary": support_summary,
        "objective_modes": objective_modes,
        "primary_improvement_vs_best_observed": float(
            best_obs_obj - opt_obj
        ),
        "interpretation": (
            "The reported optimum is a support-constrained model optimum, "
            "not a claim about the unrestricted 17-simplex. Robust objective "
            "variants are reported as sensitivity checks."
        ),
    }

    (output_dir / "SUPPORT_AWARE_MIXTURE_OPTIMIZED.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Support-aware RegMix optimization completed.")
    print(f"Output directory: {output_dir}")
    print("External A6-A11 labels used for optimization: NO")
    print("A12-A15 used for optimization: NO")
    print("Status: SUPPORT_AWARE_MIXTURE_OPTIMIZED")
    print(
        "Support threshold (Hellinger p95 LOO-NN) = "
        f"{support_threshold:.6f}"
    )
    print(
        "Primary improvement vs best observed training recipe = "
        f"{best_obs_obj - opt_obj:.6f}"
    )

    print("\nOptimal mixtures:")
    print(
        solutions[
            [
                "objective_mode",
                "objective_value",
                "support_distance",
                "within_support",
                "effective_number_domains",
                *mix_cols_out,
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\nSearch-seed stability:")
    print(
        pd.DataFrame(seed_rows)[
            [
                "random_seed",
                "candidate_count_after_support_filter",
                "best_equal_mean_objective_before_refine",
                "best_support_distance",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\nObjective sensitivity:")
    print(
        sensitivity.round(6).to_string(index=False)
    )

    print("\nBaselines:")
    print(
        baselines[
            [
                "reference",
                "equal_mean_objective",
                "support_distance",
                "improvement_vs_best_observed",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Support-aware optimization of the frozen Q1 17-domain RegMix "
            "model on the empirical simplex support."
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
        "--effects-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_effects"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/regmix_optimization"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_regmix_optimization(
        preprocessed_dir=args.preprocessed_dir,
        frozen_dir=args.frozen_dir,
        effects_dir=args.effects_dir,
        output_dir=args.output_dir,
    )

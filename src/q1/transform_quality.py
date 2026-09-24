from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.q1.preprocess import QUALITY_FIELDS


QURATER_COMPONENTS = [
    "qurater_writing_style",
    "qurater_required_expertise",
    "qurater_facts_trivia",
    "qurater_educational_value",
]


def _finite_array(series: pd.Series) -> np.ndarray:
    # pandas 2.x may expose a read-only NumPy view here (notably with
    # copy-on-write / extension-backed columns).  We intentionally copy
    # because the next line replaces non-finite values in-place.
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan
    return x


def _mid_ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    if ref.size == 0:
        raise ValueError("Reference distribution has no finite values.")

    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    mask = np.isfinite(vals)
    v = vals[mask]

    left = np.searchsorted(ref, v, side="left")
    right = np.searchsorted(ref, v, side="right")
    out[mask] = (left + right) / (2.0 * ref.size)
    return out


def _typicality_from_u(u: np.ndarray) -> np.ndarray:
    out = 1.0 - 2.0 * np.abs(u - 0.5)
    return np.clip(out, 0.0, 1.0)


def _upper_tail_penalty_from_u(u: np.ndarray, q0: float) -> np.ndarray:
    if not 0.0 < q0 < 1.0:
        raise ValueError(f"upper-tail start quantile must be in (0,1), got {q0}")
    out = np.full(u.shape, np.nan, dtype=float)
    mask = np.isfinite(u)
    um = u[mask]
    score = np.ones_like(um)
    hi = um > q0
    score[hi] = (1.0 - um[hi]) / (1.0 - q0)
    out[mask] = np.clip(score, 0.0, 1.0)
    return out


def _apply_direction(u: np.ndarray, direction: str, q0: float = 0.90) -> np.ndarray:
    if direction == "positive":
        return u
    if direction == "negative":
        return 1.0 - u
    if direction == "typicality":
        return _typicality_from_u(u)
    if direction == "upper_tail_penalty":
        return _upper_tail_penalty_from_u(u, q0)
    raise ValueError(f"Unsupported direction: {direction}")


def _domain_reference(a1: pd.DataFrame, domain: str, field: str) -> np.ndarray:
    ref = a1.loc[a1["source_domain"] == domain, field]
    arr = _finite_array(ref)
    if np.isfinite(arr).sum() == 0:
        raise ValueError(f"No finite A1 reference values for domain={domain}, field={field}")
    return arr


def _global_ecdf_score(a1: pd.DataFrame, target: pd.DataFrame, field: str, positive: bool) -> np.ndarray:
    ref = _finite_array(a1[field])
    val = _finite_array(target[field])
    u = _mid_ecdf(ref, val)
    return u if positive else 1.0 - u


def _domain_ecdf_directional(
    a1: pd.DataFrame,
    target: pd.DataFrame,
    field: str,
    directions: dict[str, str],
    q0: float = 0.90,
) -> np.ndarray:
    out = np.full(len(target), np.nan, dtype=float)

    for domain, idx in target.groupby("source_domain", sort=False).groups.items():
        domain = str(domain)
        if domain not in directions:
            raise ValueError(f"No configured direction for field={field}, domain={domain}")

        ref = _domain_reference(a1, domain, field)
        vals = _finite_array(target.loc[idx, field])
        u = _mid_ecdf(ref, vals)
        out[target.index.get_indexer(idx)] = _apply_direction(
            u,
            directions[domain],
            q0=q0,
        )

    return out


def _domain_typicality(a1: pd.DataFrame, target: pd.DataFrame, field: str) -> np.ndarray:
    out = np.full(len(target), np.nan, dtype=float)

    for domain, idx in target.groupby("source_domain", sort=False).groups.items():
        domain = str(domain)
        ref = _domain_reference(a1, domain, field)
        vals = _finite_array(target.loc[idx, field])
        u = _mid_ecdf(ref, vals)
        out[target.index.get_indexer(idx)] = _typicality_from_u(u)

    return out


def _domain_p95_saturation(
    a1: pd.DataFrame,
    target: pd.DataFrame,
    field: str,
    saturation_quantile: float,
) -> np.ndarray:
    if not 0.0 < saturation_quantile <= 1.0:
        raise ValueError("saturation_quantile must be in (0,1].")

    out = np.full(len(target), np.nan, dtype=float)

    for domain, idx in target.groupby("source_domain", sort=False).groups.items():
        domain = str(domain)
        ref = _domain_reference(a1, domain, field)
        vals = _finite_array(target.loc[idx, field])
        u = _mid_ecdf(ref, vals)
        score = np.minimum(u / saturation_quantile, 1.0)
        out[target.index.get_indexer(idx)] = np.clip(score, 0.0, 1.0)

    return out



def _domain_length_conditioned_ecdf_positive(
    a1: pd.DataFrame,
    target: pd.DataFrame,
    field: str,
    length_field: str,
    max_bins: int = 30,
    min_bins: int = 4,
    target_rows_per_bin: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Positive ECDF conditional on source domain and document length.

    Bin boundaries are fitted only on A1 within each source domain. A2/A3
    reuse those A1 boundaries and A1 conditional empirical distributions,
    so expansion data never refit the quality scale.
    """
    if max_bins < 2 or min_bins < 2 or min_bins > max_bins:
        raise ValueError("Require 2 <= min_bins <= max_bins.")
    if target_rows_per_bin < 2:
        raise ValueError("target_rows_per_bin must be >= 2.")

    out = np.full(len(target), np.nan, dtype=float)
    params: dict[str, Any] = {}

    for domain, idx in target.groupby("source_domain", sort=False).groups.items():
        domain = str(domain)
        ref_g = a1[a1["source_domain"] == domain]

        ref_length = _finite_array(ref_g[length_field])
        ref_y = _finite_array(ref_g[field])
        ref_length_ok = np.isfinite(ref_length) & (ref_length >= 0)

        n_ref_length = int(ref_length_ok.sum())
        if n_ref_length < min_bins:
            raise ValueError(
                f"Too few A1 reference rows for conditional ECDF: "
                f"domain={domain}, field={field}, n={n_ref_length}"
            )

        q_requested = min(
            max_bins,
            max(min_bins, n_ref_length // target_rows_per_bin),
            n_ref_length,
        )

        z_ref = np.log1p(ref_length[ref_length_ok])
        _, raw_edges = pd.qcut(
            z_ref,
            q=q_requested,
            retbins=True,
            duplicates="drop",
        )
        raw_edges = np.asarray(raw_edges, dtype=float)

        if raw_edges.size < 2:
            raise ValueError(
                f"Could not form length bins: domain={domain}, field={field}"
            )

        assign_edges = raw_edges.copy()
        assign_edges[0] = -np.inf
        assign_edges[-1] = np.inf
        n_bins = int(assign_edges.size - 1)

        ref_bins = np.full(len(ref_g), -1, dtype=int)
        ref_bins[ref_length_ok] = np.searchsorted(
            assign_edges[1:-1],
            np.log1p(ref_length[ref_length_ok]),
            side="right",
        )

        target_pos = target.index.get_indexer(idx)
        target_length = _finite_array(target.loc[idx, length_field])
        target_y = _finite_array(target.loc[idx, field])
        target_length_ok = np.isfinite(target_length) & (target_length >= 0)

        target_bins = np.full(len(idx), -1, dtype=int)
        target_bins[target_length_ok] = np.searchsorted(
            assign_edges[1:-1],
            np.log1p(target_length[target_length_ok]),
            side="right",
        )

        ref_bin_sizes: list[int] = []
        for bin_id in range(n_bins):
            ref_mask = (ref_bins == bin_id) & np.isfinite(ref_y)
            ref_vals = ref_y[ref_mask]
            ref_bin_sizes.append(int(ref_vals.size))

            if ref_vals.size == 0:
                raise ValueError(
                    f"Empty A1 conditional reference bin: "
                    f"domain={domain}, field={field}, bin={bin_id}"
                )

            local_pos = np.where(target_bins == bin_id)[0]
            if local_pos.size == 0:
                continue

            out[target_pos[local_pos]] = _mid_ecdf(
                ref_vals,
                target_y[local_pos],
            )

        params[domain] = {
            "length_field": length_field,
            "length_transform": "log1p",
            "requested_bins": int(q_requested),
            "actual_bins": n_bins,
            "target_rows_per_bin": int(target_rows_per_bin),
            "fitted_log1p_length_edges": [float(v) for v in raw_edges],
            "reference_bin_sizes": ref_bin_sizes,
        }

    return np.clip(out, 0.0, 1.0), params


def _fit_quadratic_residual_model(
    g: pd.DataFrame,
    length_field: str,
    target_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    length = _finite_array(g[length_field])
    y = _finite_array(g[target_field])
    mask = np.isfinite(length) & np.isfinite(y) & (length >= 0)

    if mask.sum() < 10:
        raise ValueError(
            f"Too few valid rows for residual model: field={target_field}, n={mask.sum()}"
        )

    z = np.log1p(length[mask])
    X = np.column_stack([np.ones(mask.sum()), z, z * z])
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)

    pred = X @ beta
    abs_resid = np.abs(y[mask] - pred)
    return beta, abs_resid


def _predict_abs_residual(
    df: pd.DataFrame,
    idx: pd.Index,
    length_field: str,
    target_field: str,
    beta: np.ndarray,
) -> np.ndarray:
    length = _finite_array(df.loc[idx, length_field])
    y = _finite_array(df.loc[idx, target_field])
    out = np.full(len(idx), np.nan, dtype=float)

    mask = np.isfinite(length) & np.isfinite(y) & (length >= 0)
    z = np.log1p(length[mask])
    X = np.column_stack([np.ones(mask.sum()), z, z * z])
    pred = X @ beta
    out[mask] = np.abs(y[mask] - pred)
    return out


def _domain_length_residual_typicality(
    a1: pd.DataFrame,
    target: pd.DataFrame,
    field: str,
    length_field: str,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    out = np.full(len(target), np.nan, dtype=float)
    params: dict[str, list[float]] = {}

    for domain, idx in target.groupby("source_domain", sort=False).groups.items():
        domain = str(domain)
        ref_g = a1[a1["source_domain"] == domain]
        beta, ref_abs_resid = _fit_quadratic_residual_model(
            ref_g,
            length_field=length_field,
            target_field=field,
        )
        params[domain] = [float(v) for v in beta]

        target_abs_resid = _predict_abs_residual(
            target,
            idx,
            length_field=length_field,
            target_field=field,
            beta=beta,
        )
        u = _mid_ecdf(ref_abs_resid, target_abs_resid)
        score = 1.0 - u
        out[target.index.get_indexer(idx)] = np.clip(score, 0.0, 1.0)

    return out, params


def _qurater_component_mean(a1: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    component_scores = []

    for field in QURATER_COMPONENTS:
        ref = _finite_array(a1[field])
        vals = _finite_array(target[field])
        component_scores.append(_mid_ecdf(ref, vals))

    arr = np.column_stack(component_scores)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=1)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "transforms" not in cfg:
        raise ValueError(f"Invalid transform config: {path}")
    return cfg


def transform_one_dataset(
    a1: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = pd.DataFrame(
        {
            "id": target["id"].to_numpy(),
            "sub_path": target["sub_path"].to_numpy(),
            "source_dataset": target["source_dataset"].to_numpy(),
            "source_domain": target["source_domain"].to_numpy(),
        }
    )

    fitted_params: dict[str, Any] = {}

    transforms = cfg["transforms"]

    for field in QUALITY_FIELDS:
        spec = transforms[field]
        method = spec["method"]
        score_col = f"u_{field}"

        if method == "global_ecdf_positive":
            score = _global_ecdf_score(a1, target, field, positive=True)

        elif method == "global_ecdf_negative":
            score = _global_ecdf_score(a1, target, field, positive=False)

        elif method == "component_ecdf_mean":
            score = _qurater_component_mean(a1, target)

        elif method == "domain_ecdf_directional":
            score = _domain_ecdf_directional(
                a1,
                target,
                field,
                directions=spec["directions"],
                q0=float(spec.get("upper_tail_start_quantile", 0.90)),
            )

        elif method == "domain_ecdf_p95_saturation":
            score = _domain_p95_saturation(
                a1,
                target,
                field,
                saturation_quantile=float(spec.get("saturation_quantile", 0.95)),
            )

        elif method == "domain_length_conditioned_ecdf_positive":
            score, params = _domain_length_conditioned_ecdf_positive(
                a1,
                target,
                field,
                length_field=spec["length_field"],
                max_bins=int(spec.get("max_bins", 30)),
                min_bins=int(spec.get("min_bins", 4)),
                target_rows_per_bin=int(spec.get("target_rows_per_bin", 20)),
            )
            fitted_params[field] = params

        elif method == "domain_length_residual_typicality":
            score, params = _domain_length_residual_typicality(
                a1,
                target,
                field,
                length_field=spec["length_field"],
            )
            fitted_params[field] = params

        elif method == "domain_typicality":
            score = _domain_typicality(a1, target, field)

        else:
            raise ValueError(f"Unsupported transform method for {field}: {method}")

        finite = score[np.isfinite(score)]
        if finite.size and (finite.min() < -1e-12 or finite.max() > 1.0 + 1e-12):
            raise ValueError(
                f"Transformed score outside [0,1]: field={field}, "
                f"min={finite.min()}, max={finite.max()}"
            )

        scores[score_col] = np.clip(score, 0.0, 1.0)

    expected_score_cols = [f"u_{f}" for f in QUALITY_FIELDS]
    actual_score_cols = [c for c in scores.columns if c.startswith("u_")]
    if actual_score_cols != expected_score_cols:
        raise ValueError(
            "22-score output columns do not match the original QUALITY_FIELDS order."
        )

    return scores, fitted_params


def _write_audit(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    rows = []

    for dataset, df in frames.items():
        for field in QUALITY_FIELDS:
            col = f"u_{field}"
            s = pd.to_numeric(df[col], errors="coerce")
            finite = s[np.isfinite(s)]
            rows.append(
                {
                    "dataset": dataset,
                    "field": field,
                    "rows": int(len(df)),
                    "missing": int(s.isna().sum()),
                    "min": float(finite.min()) if len(finite) else np.nan,
                    "mean": float(finite.mean()) if len(finite) else np.nan,
                    "median": float(finite.median()) if len(finite) else np.nan,
                    "max": float(finite.max()) if len(finite) else np.nan,
                }
            )

    pd.DataFrame(rows).to_csv(
        output_dir / "quality_22_transform_audit.csv",
        index=False,
    )


def run_quality_transform(
    preprocessed_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = {
        "A1": pd.read_csv(preprocessed_dir / "a1_quality_clean.csv.gz"),
        "A2": pd.read_csv(preprocessed_dir / "a2_quality_clean.csv.gz"),
        "A3": pd.read_csv(preprocessed_dir / "a3_quality_clean.csv.gz"),
    }

    cfg = _load_config(config_path)

    configured = set(cfg["transforms"])
    required = set(QUALITY_FIELDS)
    if configured != required:
        raise ValueError(
            f"Transform config mismatch. Missing={sorted(required-configured)}, "
            f"extra={sorted(configured-required)}"
        )

    transformed: dict[str, pd.DataFrame] = {}
    parameter_manifest: dict[str, Any] = {}

    for dataset, df in raw.items():
        scores, params = transform_one_dataset(raw["A1"], df, cfg)
        transformed[dataset] = scores
        parameter_manifest[dataset] = params

        scores.to_csv(
            output_dir / f"{dataset.lower()}_quality_22_scores.csv.gz",
            index=False,
            compression="gzip",
        )

    union = pd.concat(
        [transformed["A1"], transformed["A2"], transformed["A3"]],
        ignore_index=True,
    )
    union["source_priority"] = (
        union["source_dataset"].map({"A1": 0, "A2": 1, "A3": 1}).fillna(9)
    )
    union = (
        union.sort_values(["source_priority", "id"], kind="stable")
        .drop_duplicates(subset=["id"], keep="first")
        .drop(columns=["source_priority"])
        .reset_index(drop=True)
    )
    union.to_csv(
        output_dir / "quality_union_22_scores_dedup.csv.gz",
        index=False,
        compression="gzip",
    )

    _write_audit(transformed, output_dir)

    manifest = {
        "output_dimension": len(QUALITY_FIELDS),
        "quality_fields": QUALITY_FIELDS,
        "score_columns": [f"u_{f}" for f in QUALITY_FIELDS],
        "dataset_rows": {k: int(len(v)) for k, v in transformed.items()},
        "dedup_union_rows": int(len(union)),
        "all_scores_higher_is_better": True,
        "missing_policy": (
            "NaN is preserved at indicator level. Final Q must renormalize weights "
            "over available indicators instead of dropping the whole record."
        ),
        "reference_policy": cfg.get("reference_policy", {}),
        "fitted_special_parameters": parameter_manifest,
    }

    (output_dir / "quality_22_transform_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("22-indicator quality transformation completed.")
    print(f"Output directory: {output_dir}")
    print(
        "Rows:",
        ", ".join(f"{k}={len(v)}" for k, v in transformed.items()),
        f", dedup_union={len(union)}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform all 22 Q1 quality indicators to [0,1], higher-is-better scores."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("artifacts/q1/preprocessed"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1/quality_transform.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/transformed"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_quality_transform(
        preprocessed_dir=args.preprocessed_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


IDENTITY_COLUMNS = [
    "id",
    "sub_path",
    "source_dataset",
    "source_domain",
]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid aggregation config: {path}")
    return cfg


def _load_weights(path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    wt = pd.read_csv(path)

    required = {"field", "final_weight"}
    missing = required - set(wt.columns)
    if missing:
        raise ValueError(f"Weight file missing columns: {sorted(missing)}")

    if len(wt) != 22 or wt["field"].nunique() != 22:
        raise ValueError(
            f"Expected 22 unique indicators, got rows={len(wt)}, "
            f"unique={wt['field'].nunique()}"
        )

    w = pd.to_numeric(wt["final_weight"], errors="coerce").to_numpy(dtype=float)

    if not np.all(np.isfinite(w)):
        raise ValueError("Final weights contain non-finite values.")
    if np.any(w <= 0):
        raise ValueError("All final weights must be positive.")
    if not np.isclose(w.sum(), 1.0, atol=1e-10):
        raise ValueError(f"Final weights must sum to 1, got {w.sum():.12f}")

    return wt, wt["field"].tolist(), w


def _score_matrix(df: pd.DataFrame, fields: list[str]) -> np.ndarray:
    cols = [f"u_{field}" for field in fields]
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"Transformed table missing score columns: {missing}")

    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    x[~np.isfinite(x)] = np.nan
    return x


def _weighted_median_1d(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[mask]
    w = w[mask]

    if x.size == 0:
        return np.nan

    order = np.argsort(x, kind="stable")
    x = x[order]
    w = w[order]

    total = float(w.sum())
    if total <= 0:
        return np.nan

    cw = np.cumsum(w / total)
    idx = int(np.searchsorted(cw, 0.5, side="left"))
    return float(x[min(idx, len(x) - 1)])


def _weighted_median_rows(x: np.ndarray, base_w: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x)
    safe_x = np.where(valid, x, np.inf)
    row_w = np.where(valid, base_w[None, :], 0.0)

    order = np.argsort(safe_x, axis=1, kind="stable")
    sx = np.take_along_axis(safe_x, order, axis=1)
    sw = np.take_along_axis(row_w, order, axis=1)

    total = sw.sum(axis=1)
    cw = np.cumsum(sw, axis=1)

    threshold = 0.5 * total
    ge = cw >= threshold[:, None]
    idx = np.argmax(ge, axis=1)

    med = sx[np.arange(len(x)), idx].astype(float, copy=True)
    med[total <= 0] = np.nan
    return med


def calibrate_huber_scale(
    a1: pd.DataFrame,
    fields: list[str],
    base_w: np.ndarray,
    mad_consistency: float,
) -> tuple[pd.DataFrame, float]:
    x = _score_matrix(a1, fields)
    centers = _weighted_median_rows(x, base_w)
    abs_residual = np.abs(x - centers[:, None])

    rows = []

    for domain, idx in a1.groupby("source_domain", sort=False).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        a = abs_residual[pos]
        valid = np.isfinite(a)

        row_w = np.where(valid, base_w[None, :], 0.0)
        row_sum = row_w.sum(axis=1)

        good_rows = row_sum > 0
        if not np.any(good_rows):
            raise ValueError(f"No valid quality scores for domain={domain}")

        a = a[good_rows]
        valid = valid[good_rows]
        row_w = row_w[good_rows]
        row_sum = row_sum[good_rows]

        row_w = row_w / row_sum[:, None]

        vals = a[valid]
        weights = row_w[valid]

        mad = _weighted_median_1d(vals, weights)
        robust_scale = mad_consistency * mad

        rows.append(
            {
                "domain": str(domain),
                "n": int(len(pos)),
                "weighted_MAD": float(mad),
                "robust_scale": float(robust_scale),
            }
        )

    diag = pd.DataFrame(rows)
    global_scale = float(diag["robust_scale"].median())
    return diag, global_scale


def _weighted_mean_rows(x: np.ndarray, base_w: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x)
    w = np.where(valid, base_w[None, :], 0.0)
    den = w.sum(axis=1)
    num = np.nansum(x * base_w[None, :], axis=1)

    return np.divide(
        num,
        den,
        out=np.full(len(x), np.nan, dtype=float),
        where=den > 0,
    )


def huber_bisection_rows(
    x: np.ndarray,
    base_w: np.ndarray,
    delta: float,
    iterations: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if delta <= 0 or not np.isfinite(delta):
        raise ValueError(f"delta must be positive and finite, got {delta}")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    n = len(x)
    q = np.full(n, np.nan, dtype=float)
    root_residual = np.full(n, np.nan, dtype=float)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        xc = x[start:stop]

        valid = np.isfinite(xc)
        w = np.where(valid, base_w[None, :], 0.0)
        row_sum = w.sum(axis=1)
        good = row_sum > 0

        lo = np.min(np.where(valid, xc, np.inf), axis=1)
        hi = np.max(np.where(valid, xc, -np.inf), axis=1)

        lo[~good] = np.nan
        hi[~good] = np.nan

        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            r = np.where(valid, xc - mid[:, None], 0.0)
            psi = np.clip(r, -delta, delta)
            f = np.sum(w * psi, axis=1)

            move_lo = good & (f > 0)
            move_hi = good & ~move_lo
            lo[move_lo] = mid[move_lo]
            hi[move_hi] = mid[move_hi]

        qc = 0.5 * (lo + hi)
        qc[~good] = np.nan

        r = np.where(valid, xc - qc[:, None], 0.0)
        psi = np.clip(r, -delta, delta)
        f = np.sum(w * psi, axis=1)

        rr = np.divide(
            np.abs(f),
            row_sum,
            out=np.full(len(xc), np.nan, dtype=float),
            where=row_sum > 0,
        )

        q[start:stop] = qc
        root_residual[start:stop] = rr

    return q, root_residual


def score_dataset(
    df: pd.DataFrame,
    fields: list[str],
    base_w: np.ndarray,
    delta: float,
    bisection_iterations: int,
    chunk_size: int,
) -> pd.DataFrame:
    x = _score_matrix(df, fields)
    valid = np.isfinite(x)

    raw_w = np.where(valid, base_w[None, :], 0.0)
    available_weight = raw_w.sum(axis=1)

    norm_w = np.divide(
        raw_w,
        available_weight[:, None],
        out=np.zeros_like(raw_w),
        where=available_weight[:, None] > 0,
    )

    q, root_residual = huber_bisection_rows(
        x,
        base_w=base_w,
        delta=delta,
        iterations=bisection_iterations,
        chunk_size=chunk_size,
    )

    q_mean = _weighted_mean_rows(x, base_w)

    residual = x - q[:, None]
    abs_residual = np.abs(residual)

    beyond = valid & (abs_residual > delta)

    factor = np.ones_like(abs_residual, dtype=float)
    factor[beyond] = delta / abs_residual[beyond]
    factor[~valid] = 0.0

    base_weight_beyond_delta = np.sum(
        norm_w * beyond.astype(float),
        axis=1,
    )
    huber_weight_discount = np.sum(
        norm_w * (1.0 - factor) * valid,
        axis=1,
    )
    weighted_abs_deviation = np.nansum(
        norm_w * abs_residual,
        axis=1,
    )

    out = df[IDENTITY_COLUMNS].copy()

    out["Q"] = q
    out["Q_weighted_mean"] = q_mean
    out["huber_adjustment"] = q - q_mean
    out["valid_indicator_count"] = valid.sum(axis=1).astype(int)
    out["available_base_weight"] = available_weight
    out["base_weight_beyond_delta"] = base_weight_beyond_delta
    out["huber_weight_discount"] = huber_weight_discount
    out["weighted_abs_deviation"] = weighted_abs_deviation
    out["huber_root_residual"] = root_residual

    return out


def _audit_scores(scored: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for dataset, df in scored.items():
        q = pd.to_numeric(df["Q"], errors="coerce")
        finite_q = q[np.isfinite(q)]

        rows.append(
            {
                "dataset": dataset,
                "rows": int(len(df)),
                "missing_Q": int(q.isna().sum()),
                "min_Q": float(finite_q.min()) if len(finite_q) else np.nan,
                "mean_Q": float(finite_q.mean()) if len(finite_q) else np.nan,
                "median_Q": float(finite_q.median()) if len(finite_q) else np.nan,
                "max_Q": float(finite_q.max()) if len(finite_q) else np.nan,
                "mean_valid_indicator_count": float(
                    df["valid_indicator_count"].mean()
                ),
                "min_valid_indicator_count": int(
                    df["valid_indicator_count"].min()
                ),
                "mean_huber_weight_discount": float(
                    df["huber_weight_discount"].mean()
                ),
                "p95_huber_weight_discount": float(
                    df["huber_weight_discount"].quantile(0.95)
                ),
                "max_root_residual": float(
                    df["huber_root_residual"].max()
                ),
            }
        )

    return pd.DataFrame(rows)


def run_quality_aggregation(
    transformed_dir: Path,
    weights_path: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(config_path)
    wt, fields, base_w = _load_weights(weights_path)

    huber_cfg = cfg["huber"]
    mad_consistency = float(huber_cfg.get("mad_consistency", 1.4826))
    delta_multiplier = float(huber_cfg.get("delta_multiplier", 1.345))
    bisection_iterations = int(huber_cfg.get("bisection_iterations", 70))
    chunk_size = int(huber_cfg.get("chunk_size", 20000))

    transformed = {
        "A1": pd.read_csv(transformed_dir / "a1_quality_22_scores.csv.gz"),
        "A2": pd.read_csv(transformed_dir / "a2_quality_22_scores.csv.gz"),
        "A3": pd.read_csv(transformed_dir / "a3_quality_22_scores.csv.gz"),
        "UNION": pd.read_csv(
            transformed_dir / "quality_union_22_scores_dedup.csv.gz"
        ),
    }

    calibration, robust_scale = calibrate_huber_scale(
        transformed["A1"],
        fields=fields,
        base_w=base_w,
        mad_consistency=mad_consistency,
    )
    delta = delta_multiplier * robust_scale

    calibration.to_csv(
        output_dir / "huber_calibration_domain.csv",
        index=False,
    )

    scored: dict[str, pd.DataFrame] = {}
    filenames = {
        "A1": "a1_quality_Q.csv.gz",
        "A2": "a2_quality_Q.csv.gz",
        "A3": "a3_quality_Q.csv.gz",
        "UNION": "quality_union_Q_dedup.csv.gz",
    }

    for dataset, df in transformed.items():
        score = score_dataset(
            df,
            fields=fields,
            base_w=base_w,
            delta=delta,
            bisection_iterations=bisection_iterations,
            chunk_size=chunk_size,
        )
        scored[dataset] = score
        score.to_csv(
            output_dir / filenames[dataset],
            index=False,
            compression="gzip",
        )

    audit = _audit_scores(scored)
    audit.to_csv(
        output_dir / "quality_Q_audit.csv",
        index=False,
    )

    manifest = {
        "method": "hierarchical_CRITIC_plus_Huber_location",
        "weight_file": str(weights_path),
        "indicator_count": len(fields),
        "fields": fields,
        "base_weight_sum": float(base_w.sum()),
        "scale_reference": "A1 only",
        "domain_balance": (
            "compute weighted MAD scale separately in each A1 source domain; "
            "take the median of the seven domain robust scales"
        ),
        "mad_consistency": mad_consistency,
        "robust_scale": robust_scale,
        "delta_multiplier": delta_multiplier,
        "delta": delta,
        "solver": "deterministic_bisection",
        "bisection_iterations": bisection_iterations,
        "chunk_size": chunk_size,
        "missing_policy": (
            "preserve indicator NaN and renormalize the available base weights "
            "within each record; do not drop the record"
        ),
        "dataset_rows": {
            name: int(len(df))
            for name, df in scored.items()
        },
        "validation_note": (
            "A1 sensitivity analysis showed Spearman > 0.9949 versus the "
            "1.345*s reference for delta multipliers 1.0, 1.75, and 2.0. "
            "IRLS and bisection agreed to numerical precision; bisection is "
            "used to remove iterative-convergence ambiguity."
        ),
    }

    (output_dir / "quality_Q_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Huber conflict-resolved quality aggregation completed.")
    print(f"Output directory: {output_dir}")
    print(f"Indicators: {len(fields)}")
    print(f"Base weight sum: {base_w.sum():.12f}")
    print(f"A1 robust scale: {robust_scale:.9f}")
    print(f"Huber delta: {delta:.9f} (= {delta_multiplier:.3f} * scale)")
    print(
        "Rows:",
        ", ".join(f"{name}={len(df)}" for name, df in scored.items()),
    )
    print(
        "Max bisection root residual:",
        f"{audit['max_root_residual'].max():.3e}",
    )
    print(
        "Missing Q:",
        int(audit["missing_Q"].sum()),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Q1 record-level quality Q using hierarchical CRITIC "
            "weights and deterministic Huber location aggregation."
        )
    )
    parser.add_argument(
        "--transformed-dir",
        type=Path,
        default=Path("artifacts/q1/transformed"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "artifacts/q1/model/hierarchical_group_critic_weights.csv"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1/quality_aggregation.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/quality"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_quality_aggregation(
        transformed_dir=args.transformed_dir,
        weights_path=args.weights,
        config_path=args.config,
        output_dir=args.output_dir,
    )

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid domain-quality config: {path}")
    return cfg


def _finite_q(df: pd.DataFrame) -> np.ndarray:
    q = pd.to_numeric(df["Q"], errors="coerce").to_numpy(dtype=float, copy=True)
    q = q[np.isfinite(q)]
    if q.size == 0:
        raise ValueError("No finite Q values found.")
    return q


def _bootstrap_mean_replicates(
    values: np.ndarray,
    reps: int,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")
    if reps < 1:
        raise ValueError("bootstrap reps must be >= 1")
    if batch_size < 1:
        raise ValueError("bootstrap batch_size must be >= 1")

    n = values.size
    out = np.empty(reps, dtype=float)

    for start in range(0, reps, batch_size):
        stop = min(start + batch_size, reps)
        b = stop - start
        idx = rng.integers(0, n, size=(b, n), endpoint=False)
        out[start:stop] = values[idx].mean(axis=1)

    return out


def _ci(reps: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    q = np.percentile(reps, [lo, hi])
    return float(q[0]), float(q[1])


def _describe(
    label: str,
    values: np.ndarray,
    boot: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ci_low, ci_high = _ci(boot, alpha)

    return {
        "set": label,
        "n": int(values.size),
        "mean_Q": float(values.mean()),
        "median_Q": float(np.median(values)),
        "std_Q": float(values.std(ddof=1)) if values.size > 1 else np.nan,
        "P05_Q": float(np.quantile(values, 0.05)),
        "P95_Q": float(np.quantile(values, 0.95)),
        "mean_ci95_low": ci_low,
        "mean_ci95_high": ci_high,
    }


def _standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled_var = (
        ((len(a) - 1) * va + (len(b) - 1) * vb)
        / (len(a) + len(b) - 2)
    )
    if pooled_var <= 0:
        return np.nan

    return float((b.mean() - a.mean()) / np.sqrt(pooled_var))


def _check_overlap(
    sample: pd.DataFrame,
    expanded: pd.DataFrame,
    expected_overlap: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = sample[["id", "Q"]].merge(
        expanded[["id", "Q"]],
        on="id",
        how="inner",
        suffixes=("_sample", "_expanded"),
        validate="one_to_one",
    )

    if len(merged) != expected_overlap:
        raise ValueError(
            f"{label}: expected overlap={expected_overlap}, got {len(merged)}"
        )

    delta = (
        pd.to_numeric(merged["Q_sample"], errors="coerce")
        - pd.to_numeric(merged["Q_expanded"], errors="coerce")
    ).abs()

    max_delta = float(delta.max()) if len(delta) else np.nan
    if not np.isfinite(max_delta) or max_delta > 1e-12:
        raise ValueError(
            f"{label}: overlapping records do not have identical Q; "
            f"max |delta|={max_delta}"
        )

    added = expanded[~expanded["id"].isin(sample["id"])].copy()
    return merged, added


def run_domain_quality_analysis(
    quality_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(config_path)
    boot_cfg = cfg["bootstrap"]
    reps = int(boot_cfg.get("reps", 2000))
    alpha = float(boot_cfg.get("alpha", 0.05))
    seed = int(boot_cfg.get("seed", 20260924))
    batch_size = int(boot_cfg.get("batch_size", 16))

    if not 0 < alpha < 1:
        raise ValueError("bootstrap alpha must be in (0,1)")

    a1 = pd.read_csv(quality_dir / "a1_quality_Q.csv.gz")
    a2 = pd.read_csv(quality_dir / "a2_quality_Q.csv.gz")
    a3 = pd.read_csv(quality_dir / "a3_quality_Q.csv.gz")

    for name, df in {"A1": a1, "A2": a2, "A3": a3}.items():
        if df["id"].nunique() != len(df):
            raise ValueError(f"{name} contains duplicate IDs.")
        if pd.to_numeric(df["Q"], errors="coerce").isna().any():
            raise ValueError(f"{name} contains missing Q.")

    a1_arxiv = a1[a1["source_domain"].astype(str) == "arxiv"].copy()
    a1_github = a1[a1["source_domain"].astype(str) == "github"].copy()

    _, a2_added = _check_overlap(
        a1_arxiv,
        a2,
        expected_overlap=1419,
        label="arxiv A1/A2",
    )
    _, a3_added = _check_overlap(
        a1_github,
        a3,
        expected_overlap=10000,
        label="github A1/A3",
    )

    seed_seq = np.random.SeedSequence(seed)
    children = iter(seed_seq.spawn(32))

    # ----------------------------------------------------------
    # A1 baseline quality by domain
    # ----------------------------------------------------------
    a1_rows = []
    a1_boot: dict[str, np.ndarray] = {}

    domain_order = [
        "arxiv",
        "book",
        "c4",
        "commoncrawl",
        "github",
        "stackexchange",
        "wikipedia",
    ]

    for domain in domain_order:
        g = a1[a1["source_domain"].astype(str) == domain]
        values = _finite_q(g)
        rng = np.random.default_rng(next(children))
        boot = _bootstrap_mean_replicates(
            values,
            reps=reps,
            rng=rng,
            batch_size=batch_size,
        )
        a1_boot[domain] = boot

        row = _describe(
            f"A1_{domain}",
            values,
            boot,
            alpha,
        )
        row["domain"] = domain
        row["source_basis"] = "A1_sample"
        a1_rows.append(row)

    a1_summary = pd.DataFrame(a1_rows)
    a1_summary = a1_summary[
        [
            "domain",
            "source_basis",
            "n",
            "mean_Q",
            "median_Q",
            "std_Q",
            "P05_Q",
            "P95_Q",
            "mean_ci95_low",
            "mean_ci95_high",
        ]
    ]
    a1_summary.to_csv(
        output_dir / "a1_domain_quality.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Expansion comparison.
    # Bootstrap sample and added-only independently, then derive
    # full-expanded replicates as their fixed-size weighted mixture.
    # This preserves the known subset relation instead of pretending
    # sample and full-expanded are independent.
    # ----------------------------------------------------------
    expansion_rows = []
    final_domain_rows = []

    comparisons = [
        (
            "arxiv",
            a1_arxiv,
            a2_added,
            a2,
            "A2_expanded_full",
        ),
        (
            "github",
            a1_github,
            a3_added,
            a3,
            "A3_expanded_full",
        ),
    ]

    expanded_boot: dict[str, np.ndarray] = {}

    for domain, sample_df, added_df, full_df, full_basis in comparisons:
        sample = _finite_q(sample_df)
        added = _finite_q(added_df)
        full = _finite_q(full_df)

        rng_sample = np.random.default_rng(next(children))
        rng_added = np.random.default_rng(next(children))

        bs = _bootstrap_mean_replicates(
            sample,
            reps=reps,
            rng=rng_sample,
            batch_size=batch_size,
        )
        ba = _bootstrap_mean_replicates(
            added,
            reps=reps,
            rng=rng_added,
            batch_size=batch_size,
        )

        ns = len(sample)
        na = len(added)
        nf = len(full)

        if ns + na != nf:
            raise ValueError(
                f"{domain}: sample + added != expanded full "
                f"({ns}+{na}!={nf})"
            )

        bf = (ns * bs + na * ba) / nf
        expanded_boot[domain] = bf

        delta_added = ba - bs
        delta_full = bf - bs

        sample_ci = _ci(bs, alpha)
        added_ci = _ci(ba, alpha)
        full_ci = _ci(bf, alpha)
        delta_added_ci = _ci(delta_added, alpha)
        delta_full_ci = _ci(delta_full, alpha)

        expansion_rows.append(
            {
                "domain": domain,
                "sample_set": f"A1_{domain}",
                "expanded_set": full_basis,
                "sample_n": ns,
                "added_n": na,
                "expanded_n": nf,
                "sample_mean_Q": float(sample.mean()),
                "sample_ci95_low": sample_ci[0],
                "sample_ci95_high": sample_ci[1],
                "added_mean_Q": float(added.mean()),
                "added_ci95_low": added_ci[0],
                "added_ci95_high": added_ci[1],
                "expanded_mean_Q": float(full.mean()),
                "expanded_ci95_low": full_ci[0],
                "expanded_ci95_high": full_ci[1],
                "delta_added_minus_sample": float(added.mean() - sample.mean()),
                "delta_added_ci95_low": delta_added_ci[0],
                "delta_added_ci95_high": delta_added_ci[1],
                "delta_full_minus_sample": float(full.mean() - sample.mean()),
                "delta_full_ci95_low": delta_full_ci[0],
                "delta_full_ci95_high": delta_full_ci[1],
                "standardized_added_minus_sample": (
                    _standardized_mean_difference(sample, added)
                ),
            }
        )

    expansion = pd.DataFrame(expansion_rows)
    expansion.to_csv(
        output_dir / "expansion_quality_comparison.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Final seven-domain quality table:
    # arxiv/github use expanded data; other domains use A1.
    # ----------------------------------------------------------
    for domain in domain_order:
        if domain == "arxiv":
            df_domain = a2
            basis = "A2_expanded_full"
            boot = expanded_boot[domain]
        elif domain == "github":
            df_domain = a3
            basis = "A3_expanded_full"
            boot = expanded_boot[domain]
        else:
            df_domain = a1[a1["source_domain"].astype(str) == domain]
            basis = "A1_sample"
            boot = a1_boot[domain]

        values = _finite_q(df_domain)
        row = _describe(
            domain,
            values,
            boot,
            alpha,
        )
        row["domain"] = domain
        row["source_basis"] = basis
        final_domain_rows.append(row)

    final_domains = pd.DataFrame(final_domain_rows)
    final_domains = final_domains[
        [
            "domain",
            "source_basis",
            "n",
            "mean_Q",
            "median_Q",
            "std_Q",
            "P05_Q",
            "P95_Q",
            "mean_ci95_low",
            "mean_ci95_high",
        ]
    ]
    final_domains.to_csv(
        output_dir / "final_domain_quality.csv",
        index=False,
    )

    manifest = {
        "record_quality_source": str(quality_dir),
        "domain_quality_definition": "arithmetic mean of record-level robust Q",
        "uncertainty": "nonparametric percentile bootstrap of the mean",
        "bootstrap_reps": reps,
        "bootstrap_alpha": alpha,
        "bootstrap_seed": seed,
        "bootstrap_batch_size": batch_size,
        "expansion_bootstrap": (
            "stratified bootstrap: resample A1 sample and added-only records "
            "independently; derive expanded-full mean using fixed observed "
            "stratum sizes. This preserves the known subset relation."
        ),
        "final_domain_basis": {
            "arxiv": "A2 expanded full",
            "github": "A3 expanded full",
            "book": "A1",
            "c4": "A1",
            "commoncrawl": "A1",
            "stackexchange": "A1",
            "wikipedia": "A1",
        },
        "overlap_checks": {
            "A1_arxiv_in_A2": 1419,
            "A1_github_in_A3": 10000,
            "overlap_Q_tolerance": 1e-12,
        },
    }

    (output_dir / "domain_quality_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Domain-quality analysis completed.")
    print(f"Output directory: {output_dir}")
    print(f"Bootstrap reps: {reps}")
    print("Final domain basis: A2 for arxiv, A3 for github, A1 for other domains")
    print(
        "Expansion rows:",
        f"arxiv sample={len(a1_arxiv)}, added={len(a2_added)}, full={len(a2)};",
        f"github sample={len(a1_github)}, added={len(a3_added)}, full={len(a3)}",
    )
    print("Overlap Q consistency: passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Q1 domain-level quality with bootstrap confidence "
            "intervals and sample-versus-expansion comparisons."
        )
    )
    parser.add_argument(
        "--quality-dir",
        type=Path,
        default=Path("artifacts/q1/quality"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1/domain_quality.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/domain_quality"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_domain_quality_analysis(
        quality_dir=args.quality_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )

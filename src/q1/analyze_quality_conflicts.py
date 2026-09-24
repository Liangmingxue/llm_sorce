from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GROUPS = {
    "semantic_quality": [
        "fineweb_edu",
        "fluency_en",
        "modernbert_cleanliness",
        "modernbert_readability",
        "modernbert_reasoning",
        "modernbert_professionalism",
        "qurater",
        "ad_en",
    ],
    "dsir": [
        "dsir_books",
        "dsir_wiki",
        "dsir_math",
    ],
    "length_structure": [
        "rps_doc_word_count",
        "rps_doc_num_sentences",
    ],
    "rps_other": [
        "rps_doc_unigram_entropy",
        "rps_doc_frac_unique_words",
        "rps_doc_frac_no_alph_words",
        "rps_doc_frac_chars_top_2gram",
        "rps_doc_frac_chars_top_3gram",
        "rps_lines_uppercase_letter_fraction",
        "rps_lines_ending_with_terminal_punctution_mark",
        "rps_lines_numerical_chars_fraction",
        "rps_doc_mean_word_length",
    ],
}


def _load_weights(path: Path) -> pd.DataFrame:
    w = pd.read_csv(path)
    required = {"field", "final_weight"}
    if not required.issubset(w.columns):
        raise ValueError(f"Weight file missing columns: {required - set(w.columns)}")
    if len(w) != 22 or w["field"].nunique() != 22:
        raise ValueError("Expected 22 unique final quality weights.")
    w["final_weight"] = pd.to_numeric(w["final_weight"], errors="raise")
    if not np.isclose(w["final_weight"].sum(), 1.0, atol=1e-10):
        raise ValueError("Final quality weights do not sum to 1.")
    return w


def _join_scores(
    transformed: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    qcols = [
        "id",
        "Q",
        "Q_weighted_mean",
        "huber_adjustment",
        "base_weight_beyond_delta",
        "huber_weight_discount",
        "weighted_abs_deviation",
    ]
    out = transformed.merge(
        scored[qcols],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(out) != len(transformed) or len(out) != len(scored):
        raise ValueError("Transformed/Q join lost rows.")
    return out


def _group_score(
    df: pd.DataFrame,
    fields: list[str],
    weight_map: dict[str, float],
) -> np.ndarray:
    cols = [f"u_{f}" for f in fields]
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    w = np.asarray([weight_map[f] for f in fields], dtype=float)

    valid = np.isfinite(x)
    ww = np.where(valid, w[None, :], 0.0)
    den = ww.sum(axis=1)
    num = np.nansum(x * w[None, :], axis=1)
    return np.divide(
        num,
        den,
        out=np.full(len(df), np.nan, dtype=float),
        where=den > 0,
    )


def _with_conflict_features(
    df: pd.DataFrame,
    weight_map: dict[str, float],
) -> pd.DataFrame:
    out = df.copy()

    group_cols = []
    for group, fields in GROUPS.items():
        col = f"group_{group}"
        out[col] = _group_score(out, fields, weight_map)
        group_cols.append(col)

    G = out[group_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(G)):
        raise ValueError("Non-finite group scores found.")

    out["conflict_group_range"] = G.max(axis=1) - G.min(axis=1)

    hi = np.argmax(G, axis=1)
    lo = np.argmin(G, axis=1)
    names = np.asarray(list(GROUPS), dtype=object)
    out["highest_group"] = names[hi]
    out["lowest_group"] = names[lo]
    out["group_conflict_pair"] = (
        out["highest_group"].astype(str)
        + "__vs__"
        + out["lowest_group"].astype(str)
    )

    return out


def _bootstrap_rate(
    flag: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flag = np.asarray(flag, dtype=float)
    n = len(flag)
    out = np.empty(reps, dtype=float)
    chunk = 250
    pos = 0
    while pos < reps:
        m = min(chunk, reps - pos)
        idx = rng.integers(0, n, size=(m, n))
        out[pos:pos + m] = flag[idx].mean(axis=1)
        pos += m
    return out


def _rate_ci(boot: np.ndarray) -> tuple[float, float]:
    q = np.quantile(boot, [0.025, 0.975])
    return float(q[0]), float(q[1])


def _summary_row(
    name: str,
    df: pd.DataFrame,
    threshold: float,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    flag = (df["weighted_abs_deviation"].to_numpy(dtype=float) >= threshold)
    b = _bootstrap_rate(flag, reps, rng)
    lo, hi = _rate_ci(b)

    return {
        "set": name,
        "n": int(len(df)),
        "conflict_rate": float(flag.mean()),
        "conflict_rate_ci95_low": lo,
        "conflict_rate_ci95_high": hi,
        "mean_weighted_abs_deviation": float(df["weighted_abs_deviation"].mean()),
        "median_weighted_abs_deviation": float(df["weighted_abs_deviation"].median()),
        "mean_group_range": float(df["conflict_group_range"].mean()),
        "mean_huber_weight_discount": float(df["huber_weight_discount"].mean()),
    }


def run_conflict_analysis(
    transformed_dir: Path,
    quality_dir: Path,
    weights_path: Path,
    output_dir: Path,
    bootstrap_reps: int = 2000,
    random_seed: int = 20260924,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    wt = _load_weights(weights_path)
    weight_map = dict(zip(wt["field"], wt["final_weight"]))

    expected_fields = {f for fields in GROUPS.values() for f in fields}
    if expected_fields != set(weight_map):
        raise ValueError(
            "GROUPS must partition the exact 22 quality indicators. "
            f"missing={sorted(set(weight_map)-expected_fields)}, "
            f"extra={sorted(expected_fields-set(weight_map))}"
        )

    datasets = {}
    for name in ["A1", "A2", "A3"]:
        t = pd.read_csv(
            transformed_dir / f"{name.lower()}_quality_22_scores.csv.gz"
        )
        q = pd.read_csv(
            quality_dir / f"{name.lower()}_quality_Q.csv.gz"
        )
        datasets[name] = _with_conflict_features(
            _join_scores(t, q),
            weight_map,
        )

    a1 = datasets["A1"]
    a2 = datasets["A2"]
    a3 = datasets["A3"]

    # Primary definition: the most-discordant 10% of A1 records under the
    # same weighted absolute-deviation measure used to diagnose Huber Q.
    # Threshold is frozen on A1 and reused unchanged on A2/A3.
    a1_disagreement = a1["weighted_abs_deviation"].to_numpy(dtype=float)
    thresholds = {
        "q85": float(np.quantile(a1_disagreement, 0.85)),
        "q90_primary": float(np.quantile(a1_disagreement, 0.90)),
        "q95": float(np.quantile(a1_disagreement, 0.95)),
    }
    threshold = thresholds["q90_primary"]

    for df in datasets.values():
        df["strong_conflict"] = (
            df["weighted_abs_deviation"] >= threshold
        )

    # Save record-level diagnostics for reproducibility.
    keep_cols = [
        "id",
        "source_dataset",
        "source_domain",
        "Q",
        "Q_weighted_mean",
        "huber_adjustment",
        "weighted_abs_deviation",
        "base_weight_beyond_delta",
        "huber_weight_discount",
        "conflict_group_range",
        "highest_group",
        "lowest_group",
        "group_conflict_pair",
        "strong_conflict",
        *[f"group_{g}" for g in GROUPS],
    ]
    for name, df in datasets.items():
        df[keep_cols].to_csv(
            output_dir / f"{name.lower()}_conflict_records.csv.gz",
            index=False,
            compression="gzip",
        )

    seed_seq = np.random.SeedSequence(random_seed)
    rngs = iter(np.random.default_rng(s) for s in seed_seq.spawn(64))

    # Overall and A1 domain summaries.
    summary_rows = [
        _summary_row("A1_all", a1, threshold, bootstrap_reps, next(rngs)),
        _summary_row("A2_all", a2, threshold, bootstrap_reps, next(rngs)),
        _summary_row("A3_all", a3, threshold, bootstrap_reps, next(rngs)),
    ]

    for domain, g in a1.groupby("source_domain", sort=True):
        summary_rows.append(
            _summary_row(
                f"A1_{domain}",
                g,
                threshold,
                bootstrap_reps,
                next(rngs),
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "conflict_rate_summary.csv", index=False)

    # Expansion validation, preserving subset structure.
    expansion_rows = []
    specs = [
        ("arxiv", a1[a1["source_domain"] == "arxiv"], a2),
        ("github", a1[a1["source_domain"] == "github"], a3),
    ]

    for domain, sample, expanded in specs:
        overlap = sample[["id", "strong_conflict", "weighted_abs_deviation"]].merge(
            expanded[["id", "strong_conflict", "weighted_abs_deviation"]],
            on="id",
            how="inner",
            suffixes=("_sample", "_expanded"),
            validate="one_to_one",
        )
        if len(overlap) != len(sample):
            raise ValueError(f"{domain}: expanded set does not contain all sample IDs.")
        if not np.array_equal(
            overlap["strong_conflict_sample"].to_numpy(),
            overlap["strong_conflict_expanded"].to_numpy(),
        ):
            raise ValueError(f"{domain}: conflict flags differ on overlapping records.")

        added = expanded[~expanded["id"].isin(sample["id"])].copy()

        fs = sample["strong_conflict"].to_numpy(dtype=float)
        fa = added["strong_conflict"].to_numpy(dtype=float)
        ff = expanded["strong_conflict"].to_numpy(dtype=float)

        bs = _bootstrap_rate(fs, bootstrap_reps, next(rngs))
        ba = _bootstrap_rate(fa, bootstrap_reps, next(rngs))

        ns, na = len(fs), len(fa)
        bf = (ns * bs + na * ba) / (ns + na)

        delta_added = ba - bs
        delta_full = bf - bs

        s_ci = _rate_ci(bs)
        a_ci = _rate_ci(ba)
        f_ci = _rate_ci(bf)
        da_ci = _rate_ci(delta_added)
        df_ci = _rate_ci(delta_full)

        expansion_rows.append({
            "domain": domain,
            "sample_n": ns,
            "added_n": na,
            "expanded_n": len(ff),
            "sample_conflict_rate": float(fs.mean()),
            "sample_ci95_low": s_ci[0],
            "sample_ci95_high": s_ci[1],
            "added_conflict_rate": float(fa.mean()),
            "added_ci95_low": a_ci[0],
            "added_ci95_high": a_ci[1],
            "expanded_conflict_rate": float(ff.mean()),
            "expanded_ci95_low": f_ci[0],
            "expanded_ci95_high": f_ci[1],
            "delta_added_minus_sample": float(fa.mean() - fs.mean()),
            "delta_added_ci95_low": da_ci[0],
            "delta_added_ci95_high": da_ci[1],
            "delta_full_minus_sample": float(ff.mean() - fs.mean()),
            "delta_full_ci95_low": df_ci[0],
            "delta_full_ci95_high": df_ci[1],
            "overlap_flags_identical": True,
        })

    expansion = pd.DataFrame(expansion_rows)
    expansion.to_csv(
        output_dir / "conflict_expansion_validation.csv",
        index=False,
    )

    # Primary conflict-pair causes on A1 and expansions.
    pair_rows = []
    for name, df in datasets.items():
        g = (
            df[df["strong_conflict"]]
            .groupby(
                ["highest_group", "lowest_group", "group_conflict_pair"],
                as_index=False,
            )
            .size()
            .sort_values("size", ascending=False)
        )
        total = int(g["size"].sum())
        g["dataset"] = name
        g["fraction_of_conflicts"] = (
            g["size"] / total if total > 0 else np.nan
        )
        pair_rows.append(g)

    pair_causes = pd.concat(pair_rows, ignore_index=True)
    pair_causes.to_csv(
        output_dir / "conflict_group_pair_causes.csv",
        index=False,
    )

    # Indicator-level drivers: weighted absolute deviation from robust Q.
    driver_rows = []
    for name, df in datasets.items():
        conflict = df["strong_conflict"].to_numpy(dtype=bool)
        Q = df["Q"].to_numpy(dtype=float)

        for field, w in weight_map.items():
            u = pd.to_numeric(
                df[f"u_{field}"],
                errors="coerce",
            ).to_numpy(dtype=float)
            dev = np.abs(u - Q)
            signed = u - Q

            row = {
                "dataset": name,
                "field": field,
                "final_weight": float(w),
                "mean_abs_dev_conflict": float(np.nanmean(dev[conflict])),
                "mean_abs_dev_nonconflict": float(np.nanmean(dev[~conflict])),
                "excess_abs_dev_conflict_minus_nonconflict": float(
                    np.nanmean(dev[conflict]) - np.nanmean(dev[~conflict])
                ),
                "weighted_driver_score": float(
                    w * (
                        np.nanmean(dev[conflict])
                        - np.nanmean(dev[~conflict])
                    )
                ),
                "mean_signed_dev_conflict": float(np.nanmean(signed[conflict])),
            }
            driver_rows.append(row)

    drivers = pd.DataFrame(driver_rows)
    drivers.to_csv(
        output_dir / "conflict_indicator_drivers.csv",
        index=False,
    )

    # Threshold sensitivity. The ranking/cause story should not rely on one
    # exact percentile threshold.
    sensitivity_rows = []
    for label, tau in thresholds.items():
        for name, df in datasets.items():
            flag = df["weighted_abs_deviation"].to_numpy(dtype=float) >= tau
            sensitivity_rows.append({
                "threshold_label": label,
                "threshold": tau,
                "dataset": name,
                "n": len(df),
                "conflict_rate": float(flag.mean()),
            })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        output_dir / "conflict_threshold_sensitivity.csv",
        index=False,
    )

    manifest = {
        "status": "QUALITY_CONFLICT_ANALYZED",
        "definition": (
            "Record-level disagreement is the final-Q weighted absolute "
            "deviation already produced by the Huber aggregation. A strong "
            "conflict is a record at or above the A1 90th-percentile "
            "disagreement threshold. The threshold is frozen on A1 and reused "
            "unchanged on A2/A3."
        ),
        "primary_threshold": threshold,
        "threshold_sensitivity": thresholds,
        "group_partition": GROUPS,
        "cause_definition": (
            "For each strong-conflict record, the highest and lowest "
            "within-group weighted composite identify the dominant opposing "
            "quality dimensions. Indicator drivers compare |u_j-Q| in "
            "conflict versus non-conflict records."
        ),
        "resolution": (
            "The final Q is the deterministic weighted Huber location; "
            "discordant indicators beyond the Huber threshold have bounded "
            "influence rather than being deleted."
        ),
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": random_seed,
        "expansion_validation": (
            "A1 arxiv is compared with A2 added/full; A1 github with A3 "
            "added/full. Overlapping records must retain identical conflict "
            "flags. Bootstrap preserves the known sample/added subset relation."
        ),
    }
    (output_dir / "QUALITY_CONFLICT_ANALYZED.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Quality-conflict analysis completed.")
    print(f"Output directory: {output_dir}")
    print("Status: QUALITY_CONFLICT_ANALYZED")
    print(
        "Primary A1 q90 disagreement threshold = "
        f"{threshold:.6f}"
    )

    print("\nConflict-rate summary:")
    print(
        summary[
            [
                "set",
                "n",
                "conflict_rate",
                "conflict_rate_ci95_low",
                "conflict_rate_ci95_high",
                "mean_weighted_abs_deviation",
                "mean_group_range",
                "mean_huber_weight_discount",
            ]
        ].round(6).to_string(index=False)
    )

    print("\nExpansion validation:")
    print(expansion.round(6).to_string(index=False))

    print("\nTop A1 conflict group-pair causes:")
    print(
        pair_causes[pair_causes["dataset"] == "A1"]
        .head(10)[
            [
                "highest_group",
                "lowest_group",
                "size",
                "fraction_of_conflicts",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\nTop A1 indicator drivers:")
    print(
        drivers[drivers["dataset"] == "A1"]
        .sort_values("weighted_driver_score", ascending=False)
        .head(10)[
            [
                "field",
                "final_weight",
                "excess_abs_dev_conflict_minus_nonconflict",
                "weighted_driver_score",
                "mean_signed_dev_conflict",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify and validate Q1 quality-signal conflicts."
    )
    parser.add_argument(
        "--transformed-dir",
        type=Path,
        default=Path("artifacts/q1/transformed"),
    )
    parser.add_argument(
        "--quality-dir",
        type=Path,
        default=Path("artifacts/q1/quality"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "artifacts/q1/model/hierarchical_group_critic_weights.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/conflict_analysis"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_conflict_analysis(
        transformed_dir=args.transformed_dir,
        quality_dir=args.quality_dir,
        weights_path=args.weights,
        output_dir=args.output_dir,
    )

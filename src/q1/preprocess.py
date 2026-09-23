from __future__ import annotations

import argparse
import json
import lzma
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


QUALITY_FIELDS = [
    "fineweb_edu",
    "fluency_en",
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
    "ad_en",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_mean_word_length",
]

MODERNBERT_FIELDS = [
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
]

RPS_AND_SCALAR_FIELDS = [
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_mean_word_length",
]


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _require_real_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    if _is_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is still a Git LFS pointer. Run 'git lfs pull' in the repository first."
        )


def _finite_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if math.isfinite(v) else np.nan


def _softmax(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([_finite_float(x) for x in values], dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return np.full(arr.shape, np.nan, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    den = exp.sum()
    return exp / den if den > 0 else np.full(arr.shape, np.nan, dtype=float)


def _binary_positive_probability(values: Any) -> float:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return np.nan
    p = _softmax(values)
    return float(p[1]) if p.size == 2 and np.isfinite(p[1]) else np.nan


def _ordinal_expected_score(values: Any) -> float:
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        return np.nan
    p = _softmax(values)
    if p.size != 6 or not np.all(np.isfinite(p)):
        return np.nan
    return float(np.dot(np.arange(6, dtype=float), p) / 5.0)


def _fineweb_scalar(values: Any) -> float:
    if isinstance(values, (list, tuple)) and len(values) >= 1:
        return _finite_float(values[0])
    return _finite_float(values)


def _qurater_scalar(values: Any) -> tuple[float, float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    parts = [_finite_float(v) for v in values]
    mean = float(np.nanmean(parts)) if np.any(np.isfinite(parts)) else np.nan
    return (parts[0], parts[1], parts[2], parts[3], mean)


def _infer_extended_domain(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("arxiv_"):
        return "arxiv"
    if name.startswith("github_"):
        return "github"
    return "unknown"


def _iter_jsonl_xz(path: Path):
    _require_real_file(path)
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error in {path}:{line_no}: {exc}") from exc


def _quality_record_to_row(record: dict[str, Any], source_dataset: str, source_domain: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": record.get("id"),
        "sub_path": record.get("sub_path"),
        "source_dataset": source_dataset,
        "source_domain": source_domain,
    }

    row["fineweb_edu"] = _fineweb_scalar(record.get("fineweb_edu"))
    row["fluency_en"] = _binary_positive_probability(record.get("fluency_en"))

    for field in MODERNBERT_FIELDS:
        row[field] = _ordinal_expected_score(record.get(field))

    q0, q1, q2, q3, qmean = _qurater_scalar(record.get("qurater"))
    row["qurater_writing_style"] = q0
    row["qurater_required_expertise"] = q1
    row["qurater_facts_trivia"] = q2
    row["qurater_educational_value"] = q3
    row["qurater"] = qmean

    # ad_en stores [has_ad, no_ad] logits. Higher transformed value means cleaner/no-ad.
    row["ad_en"] = _binary_positive_probability(record.get("ad_en"))

    for field in RPS_AND_SCALAR_FIELDS:
        row[field] = _finite_float(record.get(field))

    return row


def preprocess_quality_file(path: Path, source_dataset: str, source_domain: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, record in _iter_jsonl_xz(path):
        domain = source_domain or str(record.get("_source_domain", "unknown"))
        rows.append(_quality_record_to_row(record, source_dataset=source_dataset, source_domain=domain))
    df = pd.DataFrame(rows)

    required = ["id", "source_dataset", "source_domain", *QUALITY_FIELDS]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing expected columns after preprocessing: {missing_cols}")
    return df


def _write_quality_reports(frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    audits = []
    missing_rows = []

    for name, df in frames.items():
        audits.append(
            {
                "dataset": name,
                "rows": len(df),
                "unique_ids": int(df["id"].nunique(dropna=True)),
                "duplicate_ids": int(df["id"].duplicated().sum()),
                "domains": int(df["source_domain"].nunique(dropna=True)),
            }
        )
        for col in [*QUALITY_FIELDS, "qurater_writing_style", "qurater_required_expertise",
                    "qurater_facts_trivia", "qurater_educational_value"]:
            s = pd.to_numeric(df[col], errors="coerce")
            missing_rows.append(
                {
                    "dataset": name,
                    "field": col,
                    "missing_or_nonfinite": int((~np.isfinite(s.to_numpy(dtype=float))).sum()),
                    "min": float(np.nanmin(s)) if s.notna().any() else np.nan,
                    "max": float(np.nanmax(s)) if s.notna().any() else np.nan,
                    "mean": float(np.nanmean(s)) if s.notna().any() else np.nan,
                }
            )

    pd.DataFrame(audits).to_csv(out_dir / "quality_dataset_audit.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(out_dir / "quality_field_audit.csv", index=False)


def preprocess_quality(data_root: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    a_root = data_root / "A_data_value"
    a1 = a_root / "slimpajama_quality_signal_sample.jsonl.xz"
    ext_dir = a_root / "slimpajama_quality_extended"

    frames: dict[str, pd.DataFrame] = {
        "A1": preprocess_quality_file(a1, "A1"),
    }

    for path in sorted(ext_dir.glob("*.jsonl.xz")):
        domain = _infer_extended_domain(path)
        dataset = "A2" if domain == "arxiv" else "A3" if domain == "github" else path.stem
        frames[dataset] = preprocess_quality_file(path, dataset, source_domain=domain)

    expected = {"A1", "A2", "A3"}
    if set(frames) != expected:
        raise ValueError(f"Expected A1/A2/A3 quality datasets, got {sorted(frames)}")

    for name, df in frames.items():
        df.to_csv(out_dir / f"{name.lower()}_quality_clean.csv.gz", index=False, compression="gzip")

    # The extended arXiv/GitHub sets contain the A1 samples. Keep A1 rows in the de-duplicated
    # union so source-domain labels remain explicit and no overlapping id is counted twice.
    union = pd.concat([frames["A1"], frames["A2"], frames["A3"]], ignore_index=True)
    union["source_priority"] = union["source_dataset"].map({"A1": 0, "A2": 1, "A3": 1}).fillna(9)
    union = (
        union.sort_values(["source_priority", "id"], kind="stable")
        .drop_duplicates(subset=["id"], keep="first")
        .drop(columns=["source_priority"])
        .reset_index(drop=True)
    )
    union.to_csv(out_dir / "quality_union_dedup.csv.gz", index=False, compression="gzip")

    overlap = {
        "A1_A2_id_overlap": int(len(set(frames["A1"]["id"]).intersection(set(frames["A2"]["id"])))),
        "A1_A3_id_overlap": int(len(set(frames["A1"]["id"]).intersection(set(frames["A3"]["id"])))),
        "A2_A3_id_overlap": int(len(set(frames["A2"]["id"]).intersection(set(frames["A3"]["id"])))),
        "dedup_union_rows": int(len(union)),
    }
    (out_dir / "quality_overlap.json").write_text(
        json.dumps(overlap, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_quality_reports(frames, out_dir)
    return frames


def _clean_mixture_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prefix = "train_the_pile_"
    mix_cols = [c for c in df.columns if c.startswith(prefix)]
    if len(mix_cols) != 17:
        raise ValueError(f"Expected 17 mixture columns, found {len(mix_cols)}")

    domains = [c[len(prefix):] for c in mix_cols]
    out = df.copy()
    for col, domain in zip(mix_cols, domains):
        out[f"p_raw_{domain}"] = pd.to_numeric(out[col], errors="raise")
    raw_cols = [f"p_raw_{d}" for d in domains]
    out["mixture_sum_raw"] = out[raw_cols].sum(axis=1)
    if (out["mixture_sum_raw"] <= 0).any():
        raise ValueError("Found non-positive mixture row sum.")

    # Preserve original columns and add exact-simplex normalized columns.
    for domain in domains:
        out[f"p_{domain}"] = out[f"p_raw_{domain}"] / out["mixture_sum_raw"]
    return out, domains


def _clean_loss_columns(df: pd.DataFrame) -> pd.DataFrame:
    prefix = "metric/the_pile_"
    suffix = "_val_loss"
    out = df.copy()
    rename = {}
    for col in df.columns:
        if col.startswith(prefix) and col.endswith(suffix):
            domain = col[len(prefix):-len(suffix)]
            rename[col] = f"loss_{domain}"
    if len(rename) != 13:
        raise ValueError(f"Expected 13 loss columns, found {len(rename)}")
    out = out.rename(columns=rename)
    return out


REGMIX_PAIRS = [
    ("train_1m", "train_mixture_1m.csv", "train_pile_loss_1m.csv", "train"),
    ("test_1m", "test_mixture_1m.csv", "test_pile_loss_1m.csv", "test"),
    ("test_60m", "test_mixture_60m.csv", "test_pile_loss_60m.csv", "test"),
    ("test_1B", "test_mixture_1B.csv", "test_pile_loss_1B.csv", "test"),
    ("est_10b", "est_mixture_10b.csv", "est_pile_loss_10b.csv", "estimated"),
    ("est_70b", "est_mixture_70b.csv", "est_pile_loss_70b.csv", "estimated"),
]


def preprocess_regmix(data_root: Path, out_dir: Path) -> None:
    table_dir = data_root / "A_data_value" / "regmix_tables"
    audit_rows = []
    expected_domains: list[str] | None = None

    for label, mixture_name, loss_name, split in REGMIX_PAIRS:
        mix_path = table_dir / mixture_name
        loss_path = table_dir / loss_name
        _require_real_file(mix_path)
        _require_real_file(loss_path)

        mix = pd.read_csv(mix_path)
        loss = pd.read_csv(loss_path)

        if mix["index"].duplicated().any() or loss["index"].duplicated().any():
            raise ValueError(f"Duplicate index detected in pair {label}")

        mix, domains = _clean_mixture_columns(mix)
        loss = _clean_loss_columns(loss)

        if expected_domains is None:
            expected_domains = domains
        elif domains != expected_domains:
            raise ValueError(f"Mixture domain mismatch in {label}")

        merged = mix.merge(loss, on="index", how="outer", validate="one_to_one", indicator=True)
        if not (merged["_merge"] == "both").all():
            raise ValueError(f"Index mismatch between mixture/loss tables for {label}")
        merged = merged.drop(columns=["_merge"])
        merged.insert(1, "split", split)
        merged.insert(2, "scale", label.replace("train_", "").replace("test_", "").replace("est_", ""))

        if merged.isna().any().any():
            raise ValueError(f"Unexpected missing values after merging {label}")

        p_cols = [f"p_{d}" for d in domains]
        normalized_sum_error = float((merged[p_cols].sum(axis=1) - 1.0).abs().max())
        merged.to_csv(out_dir / f"regmix_{label}_clean.csv", index=False)

        audit_rows.append(
            {
                "dataset": label,
                "split": split,
                "rows": len(merged),
                "index_min": int(merged["index"].min()),
                "index_max": int(merged["index"].max()),
                "raw_sum_min": float(merged["mixture_sum_raw"].min()),
                "raw_sum_max": float(merged["mixture_sum_raw"].max()),
                "normalized_sum_max_abs_error": normalized_sum_error,
                "loss_min": float(merged.filter(like="loss_").min().min()),
                "loss_max": float(merged.filter(like="loss_").max().max()),
            }
        )

    pd.DataFrame(audit_rows).to_csv(out_dir / "regmix_dataset_audit.csv", index=False)

    mapping_path = data_root / "A_data_value" / "domain_mapping_guide.csv"
    mapping = pd.read_csv(mapping_path)
    mapping["quality_domain"] = mapping["quality_domain"].replace({"(none)": np.nan})
    if expected_domains is not None and set(mapping["mixture_domain"]) != set(expected_domains):
        raise ValueError("A16 mapping domains do not match the 17 RegMix mixture domains.")
    mapping.to_csv(out_dir / "domain_mapping_clean.csv", index=False)


def run_preprocessing(data_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_quality(data_root, output_dir)
    preprocess_regmix(data_root, output_dir)

    manifest = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "notes": [
            "A1/A2/A3 are kept as separate cleaned tables for required comparison.",
            "quality_union_dedup removes A1-vs-A2/A3 id overlap for pooled analyses only.",
            "No global normalization, direction unification, clipping, or imputation is performed here.",
            "RegMix original rounded proportions are preserved as p_raw_* and exact-simplex values are added as p_*.",
            "A6-A11 remain separate test sets; A12-A15 remain estimated/extrapolation sets.",
        ],
    }
    (output_dir / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Question 1 (A1-A16) data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("real_attachments"),
        help="Path to real_attachments/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/q1/preprocessed"),
        help="Directory for generated clean tables and audits.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_preprocessing(args.data_root, args.output_dir)

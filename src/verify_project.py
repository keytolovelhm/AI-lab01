from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import joblib
import pandas as pd

from .lab1_common import (
    PROJECT_ROOT,
    CleanedTextPreprocessor,
    build_vectorizer,
    load_config,
    sha256_file,
)


def main() -> int:
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    required = [
        "README.md",
        "configs/experiment_config.json",
        "data/raw/train_data.csv",
        "data/raw/test_data_unlabeled.csv",
        "data/splits/train.csv",
        "data/splits/validation.csv",
        "data/splits/internal_test.csv",
        "results/experiment_results.csv",
        "results/model_selection.json",
        "results/final_metrics.json",
        "models/final_pipeline.joblib",
        "submission/predictions.csv",
    ]
    missing = [item for item in required if not (PROJECT_ROOT / item).exists()]
    check("required_files", not missing, f"missing={missing}")

    for source in (PROJECT_ROOT / "src").glob("*.py"):
        py_compile.compile(str(source), doraise=True)
    check("python_compilation", True, "all src/*.py compiled")

    config = load_config()
    cleaned_vectorizer = build_vectorizer(config, "cleaned_tuned_word_tfidf")
    cleaned_preprocessor = cleaned_vectorizer.preprocessor
    normalization_ok = (
        isinstance(cleaned_preprocessor, CleanedTextPreprocessor)
        and cleaned_preprocessor.lowercase is True
        and cleaned_preprocessor.strip_accents == "unicode"
        and cleaned_vectorizer.lowercase is False
        and cleaned_vectorizer.strip_accents is None
    )
    check(
        "cleaned_ablation_normalization",
        normalization_ok,
        "custom cleaner explicitly preserves lowercase=True and strip_accents=unicode semantics",
    )
    train_raw = pd.read_csv(PROJECT_ROOT / "data/raw/train_data.csv")
    external = pd.read_csv(PROJECT_ROOT / "data/raw/test_data_unlabeled.csv")
    split_frames = {
        name: pd.read_csv(PROJECT_ROOT / f"data/splits/{name}.csv")
        for name in ("train", "validation", "internal_test")
    }
    ids = {name: set(frame["source_row_id"]) for name, frame in split_frames.items()}
    overlap = (ids["train"] & ids["validation"]) | (ids["train"] & ids["internal_test"]) | (
        ids["validation"] & ids["internal_test"]
    )
    union = set.union(*ids.values())
    check("split_disjoint", not overlap, f"overlap_count={len(overlap)}")
    check(
        "split_complete",
        union == set(range(len(train_raw))),
        f"union_rows={len(union)} raw_rows={len(train_raw)}",
    )

    results = pd.read_csv(PROJECT_ROOT / "results/experiment_results.csv")
    expected_experiments = sum(
        len(model["values"]) for model in config["models"].values()
    ) * len(config["feature_configs"])
    check(
        "experiment_count",
        len(results) == expected_experiments,
        f"actual={len(results)} expected={expected_experiments}",
    )
    recomputed = results.sort_values(
        ["val_macro_f1", "val_accuracy", "val_weighted_f1", "total_seconds"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]
    selection = json.loads(
        (PROJECT_ROOT / "results/model_selection.json").read_text(encoding="utf-8")
    )
    check(
        "selection_recomputed_from_validation",
        recomputed["experiment_id"] == selection["selected_experiment_id"],
        f"recomputed={recomputed['experiment_id']} stored={selection['selected_experiment_id']}",
    )
    check(
        "test_lock_policy_recorded",
        selection.get("selected_using_validation_only") is True
        and selection.get("internal_test_was_untouched_until_after_lock") is True,
        "validation-only selection flags",
    )

    prediction_path = PROJECT_ROOT / "submission/predictions.csv"
    predictions = pd.read_csv(prediction_path, header=None)
    prediction_values = predictions.iloc[:, 0]
    allowed_labels = set(train_raw["target"].unique())
    check("prediction_one_column", predictions.shape[1] == 1, f"shape={predictions.shape}")
    check(
        "prediction_row_count",
        len(predictions) == len(external),
        f"predictions={len(predictions)} external={len(external)}",
    )
    check(
        "prediction_integer_labels",
        prediction_values.notna().all()
        and set(prediction_values.astype(int).unique()).issubset(allowed_labels)
        and (prediction_values == prediction_values.astype(int)).all(),
        f"unique={sorted(prediction_values.unique().tolist())}",
    )
    with prediction_path.open("r", encoding="utf-8") as stream:
        first_line = stream.readline().strip()
    check("prediction_no_header", first_line in {str(x) for x in allowed_labels}, f"first_line={first_line}")

    pipeline = joblib.load(PROJECT_ROOT / "models/final_pipeline.joblib")
    regenerated = pipeline.predict(external["text"])
    check(
        "saved_model_reproduces_submission",
        regenerated.astype(int).tolist() == prediction_values.astype(int).tolist(),
        "model predictions exactly match submission",
    )

    manifest = json.loads(
        (PROJECT_ROOT / "results/input_manifest.json").read_text(encoding="utf-8")
    )
    check(
        "raw_train_hash",
        sha256_file(PROJECT_ROOT / "data/raw/train_data.csv")
        == manifest["files"]["train_data.csv"]["sha256"],
        "raw train copy unchanged",
    )
    check(
        "raw_external_hash",
        sha256_file(PROJECT_ROOT / "data/raw/test_data_unlabeled.csv")
        == manifest["files"]["test_data_unlabeled.csv"]["sha256"],
        "raw external test copy unchanged",
    )

    passed = all(item["passed"] for item in checks)
    report = {
        "passed": passed,
        "check_count": len(checks),
        "failed_count": sum(not item["passed"] for item in checks),
        "submission_sha256": sha256_file(prediction_path),
        "checks": checks,
    }
    output = PROJECT_ROOT / "results/verification_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_text = "\n".join(
        f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check']}: {item['detail']}"
        for item in checks
    )
    (PROJECT_ROOT / "results/verification.log").write_text(log_text + "\n", encoding="utf-8")
    print(log_text)
    print(f"Overall verification: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

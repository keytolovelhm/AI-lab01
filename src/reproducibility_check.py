from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .lab1_common import PROJECT_ROOT, sha256_file


BASELINE_PATH = PROJECT_ROOT / "results" / "reproducibility_baseline.json"
REPORT_PATH = PROJECT_ROOT / "results" / "reproducibility_report.json"


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_signature() -> dict:
    metrics_dir = PROJECT_ROOT / "results"
    selection = json.loads((metrics_dir / "model_selection.json").read_text(encoding="utf-8"))
    final = json.loads((metrics_dir / "final_metrics.json").read_text(encoding="utf-8"))
    experiments = pd.read_csv(PROJECT_ROOT / "results" / "experiment_results.csv")
    stable_columns = [
        "experiment_id",
        "feature_config",
        "feature_count",
        "model",
        "parameter_name",
        "parameter_value",
        "val_accuracy",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
        "val_weighted_precision",
        "val_weighted_recall",
        "val_weighted_f1",
    ]
    stable_experiments = (
        experiments[stable_columns]
        .sort_values("experiment_id")
        .to_dict(orient="records")
    )
    split_hashes = {
        name: sha256_file(PROJECT_ROOT / "data" / "splits" / f"{name}.csv")
        for name in ("train", "validation", "internal_test")
    }
    return {
        "selected_experiment_id": selection["selected_experiment_id"],
        "validation_metrics": selection["validation_metrics"],
        "internal_test_metrics": final["internal_test_metrics"],
        "predictions_sha256": sha256_file(PROJECT_ROOT / "submission" / "predictions.csv"),
        "split_hashes": split_hashes,
        "validation_table_sha256": canonical_hash(stable_experiments),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()
    current = stable_signature()
    if args.save_baseline:
        BASELINE_PATH.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Saved reproducibility baseline: {BASELINE_PATH}")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    keys = sorted(set(baseline) | set(current))
    comparisons = [
        {
            "item": key,
            "passed": baseline.get(key) == current.get(key),
            "baseline": baseline.get(key),
            "rerun": current.get(key),
        }
        for key in keys
    ]
    passed = all(item["passed"] for item in comparisons)
    report = {
        "passed": passed,
        "purpose": "Compare deterministic splits, all validation metrics, locked test metrics and final predictions across two full runs.",
        "comparisons": comparisons,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    for item in comparisons:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['item']}")
    print(f"Full-rerun reproducibility: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

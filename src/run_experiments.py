from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .lab1_common import (
    PROJECT_ROOT,
    build_model,
    build_pipeline,
    build_vectorizer,
    classification_metrics,
    load_config,
    set_global_seed,
    sha256_file,
    validate_raw_inputs,
)


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("lab1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def create_splits(train_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, ...]:
    ratios = config["split_ratios"]
    seed = int(config["random_seed"])
    validation_seed = int(config["validation_split_seed"])
    indexed = train_df.copy()
    indexed.insert(0, "source_row_id", np.arange(len(indexed), dtype=int))

    development, internal_test = train_test_split(
        indexed,
        test_size=float(ratios["internal_test"]),
        random_state=seed,
        stratify=indexed["target"],
    )
    validation_fraction_of_development = float(ratios["validation"]) / (
        float(ratios["train"]) + float(ratios["validation"])
    )
    train, validation = train_test_split(
        development,
        test_size=validation_fraction_of_development,
        random_state=validation_seed,
        stratify=development["target"],
    )
    return (
        train.sort_values("source_row_id").reset_index(drop=True),
        validation.sort_values("source_row_id").reset_index(drop=True),
        internal_test.sort_values("source_row_id").reset_index(drop=True),
    )


def save_data_audit(
    train_df: pd.DataFrame,
    external_test_df: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    internal_test: pd.DataFrame,
    tables_dir: Path,
    metrics_dir: Path,
) -> None:
    raw_overlap = len(set(train_df["text"]) & set(external_test_df["text"]))
    audit = {
        "raw_train_rows": int(len(train_df)),
        "external_unlabeled_test_rows": int(len(external_test_df)),
        "class_count": int(train_df["target"].nunique()),
        "labels": sorted(int(x) for x in train_df["target"].unique()),
        "raw_train_missing": int(train_df.isna().sum().sum()),
        "external_test_missing": int(external_test_df.isna().sum().sum()),
        "raw_train_duplicate_rows": int(train_df.duplicated().sum()),
        "external_test_duplicate_rows": int(external_test_df.duplicated().sum()),
        "train_external_exact_text_overlap": int(raw_overlap),
        "split_rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "internal_test": int(len(internal_test)),
        },
    }
    write_json(metrics_dir / "data_audit.json", audit)

    distribution_rows = []
    for split_name, frame in (
        ("raw_labeled", train_df),
        ("train", train),
        ("validation", validation),
        ("internal_test", internal_test),
    ):
        counts = frame["target"].value_counts().sort_index()
        for label, count in counts.items():
            distribution_rows.append(
                {"split": split_name, "target": int(label), "count": int(count)}
            )
    pd.DataFrame(distribution_rows).to_csv(
        tables_dir / "class_distribution.csv", index=False
    )


def run_validation_grid(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    feature_rows: list[dict] = []
    seed = int(config["random_seed"])

    for feature_name, feature_cfg in config["feature_configs"].items():
        logger.info("Fitting feature configuration: %s", feature_name)
        vectorizer = build_vectorizer(config, feature_name)
        feature_start = time.perf_counter()
        x_train = vectorizer.fit_transform(train["text"])
        x_validation = vectorizer.transform(validation["text"])
        feature_seconds = time.perf_counter() - feature_start
        density = float(x_train.nnz / (x_train.shape[0] * x_train.shape[1]))
        feature_rows.append(
            {
                "feature_config": feature_name,
                "description": feature_cfg["description"],
                "train_rows": int(x_train.shape[0]),
                "feature_count": int(x_train.shape[1]),
                "nonzero_values": int(x_train.nnz),
                "density": density,
                "feature_fit_transform_seconds": feature_seconds,
            }
        )

        for model_name, model_cfg in config["models"].items():
            parameter_name = model_cfg["parameter_name"]
            for parameter_value in model_cfg["values"]:
                experiment_id = (
                    f"{feature_name}__{model_name}__"
                    f"{parameter_name}_{str(parameter_value).replace('.', 'p')}"
                )
                model = build_model(config, model_name, float(parameter_value), seed)
                warning_messages: list[str] = []
                model_start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(x_train, train["target"].to_numpy())
                    model_fit_seconds = time.perf_counter() - model_start
                    prediction_start = time.perf_counter()
                    predicted = model.predict(x_validation)
                    prediction_seconds = time.perf_counter() - prediction_start
                    warning_messages = [str(item.message) for item in caught]
                metrics = classification_metrics(
                    validation["target"].to_numpy(), np.asarray(predicted)
                )
                row = {
                    "experiment_id": experiment_id,
                    "feature_config": feature_name,
                    "feature_count": int(x_train.shape[1]),
                    "model": model_name,
                    "model_display_name": model_cfg["display_name"],
                    "parameter_name": parameter_name,
                    "parameter_value": float(parameter_value),
                    "feature_fit_transform_seconds": feature_seconds,
                    "model_fit_seconds": model_fit_seconds,
                    "validation_predict_seconds": prediction_seconds,
                    "total_seconds": feature_seconds
                    + model_fit_seconds
                    + prediction_seconds,
                    "warnings": " | ".join(warning_messages),
                }
                row.update({f"val_{key}": value for key, value in metrics.items()})
                rows.append(row)
                logger.info(
                    "%s | val_acc=%.4f val_macro_f1=%.4f model_fit=%.3fs",
                    experiment_id,
                    row["val_accuracy"],
                    row["val_macro_f1"],
                    model_fit_seconds,
                )

    results = pd.DataFrame(rows)
    feature_table = pd.DataFrame(feature_rows)
    return results, feature_table


def select_validation_winner(results: pd.DataFrame) -> pd.Series:
    ranked = results.sort_values(
        ["val_macro_f1", "val_accuracy", "val_weighted_f1", "total_seconds"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def plot_validation_results(results: pd.DataFrame, plots_dir: Path) -> None:
    best_by_group = (
        results.sort_values("val_macro_f1", ascending=False)
        .groupby(["feature_config", "model_display_name"], as_index=False)
        .first()
    )
    pivot = best_by_group.pivot(
        index="model_display_name", columns="feature_config", values="val_macro_f1"
    )
    ax = pivot.plot(kind="bar", figsize=(12, 6), ylim=(0, 1), rot=10)
    ax.set_title("Best validation macro-F1 by model and TF-IDF configuration")
    ax.set_xlabel("Model")
    ax.set_ylabel("Validation macro-F1")
    ax.legend(
        title="TF-IDF configuration",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(rect=[0, 0, 0.83, 1])
    plt.savefig(plots_dir / "validation_model_comparison.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (model_name, group) in zip(axes, results.groupby("model", sort=True)):
        for feature_name, feature_group in group.groupby("feature_config"):
            ordered = feature_group.sort_values("parameter_value")
            ax.plot(
                ordered["parameter_value"],
                ordered["val_macro_f1"],
                marker="o",
                label=feature_name,
            )
        ax.set_xscale("log")
        ax.set_title(group["model_display_name"].iloc[0])
        ax.set_xlabel(group["parameter_name"].iloc[0])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Validation macro-F1")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Hyperparameter validation curves")
    fig.tight_layout()
    fig.savefig(plots_dir / "validation_hyperparameter_curves.png", dpi=180)
    plt.close(fig)


def plot_split_distribution(tables_dir: Path, plots_dir: Path) -> None:
    distribution = pd.read_csv(tables_dir / "class_distribution.csv")
    subset = distribution[distribution["split"].isin(["train", "validation", "internal_test"])]
    pivot = subset.pivot(index="target", columns="split", values="count")
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_title("Class distribution after stratified 70/15/15 split")
    ax.set_xlabel("Target label")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="split", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(plots_dir / "split_class_distribution.png", dpi=180)
    plt.close()


def build_innovation_ablation(results: pd.DataFrame) -> pd.DataFrame:
    """Create paired feature-ablation deltas at identical model parameters."""

    index_cols = ["model", "model_display_name", "parameter_name", "parameter_value"]
    macro = results.pivot(index=index_cols, columns="feature_config", values="val_macro_f1")
    accuracy = results.pivot(index=index_cols, columns="feature_config", values="val_accuracy")
    table = macro.add_suffix("__macro_f1").join(accuracy.add_suffix("__accuracy")).reset_index()
    table["tuned_minus_teacher_macro_f1"] = (
        table["tuned_word_tfidf__macro_f1"] - table["teacher_baseline__macro_f1"]
    )
    table["cleaned_minus_tuned_macro_f1"] = (
        table["cleaned_tuned_word_tfidf__macro_f1"]
        - table["tuned_word_tfidf__macro_f1"]
    )
    return table.sort_values(["model", "parameter_value"]).reset_index(drop=True)


def plot_confusion(cm: np.ndarray, labels: list[int], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Internal holdout confusion matrix (locked model)",
    )
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete Lab 1 experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment_config.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    set_global_seed(int(config["random_seed"]))

    for relative in (
        "data/splits",
        "results",
        "results/plots",
        "logs",
        "models",
        "submission",
    ):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
    tables_dir = PROJECT_ROOT / "results"
    metrics_dir = PROJECT_ROOT / "results"
    plots_dir = PROJECT_ROOT / "results" / "plots"
    logger = configure_logging(PROJECT_ROOT / "logs" / "experiment.log")
    run_start = time.perf_counter()

    raw_train_path = PROJECT_ROOT / "data" / "raw" / "train_data.csv"
    external_test_path = PROJECT_ROOT / "data" / "raw" / "test_data_unlabeled.csv"
    teacher_sample_path = (
        PROJECT_ROOT / "reference" / "teacher_original" / "predictions.csv"
    )
    train_df = pd.read_csv(raw_train_path)
    external_test_df = pd.read_csv(external_test_path)
    validate_raw_inputs(train_df, external_test_df)
    logger.info("Loaded %d labeled and %d external test rows", len(train_df), len(external_test_df))

    train, validation, internal_test = create_splits(train_df, config)
    split_dir = PROJECT_ROOT / "data" / "splits"
    train.to_csv(split_dir / "train.csv", index=False)
    validation.to_csv(split_dir / "validation.csv", index=False)
    internal_test.to_csv(split_dir / "internal_test.csv", index=False)
    split_ids = [set(x["source_row_id"]) for x in (train, validation, internal_test)]
    if split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]:
        raise RuntimeError("Split leakage detected: source_row_id overlap")
    if set.union(*split_ids) != set(range(len(train_df))):
        raise RuntimeError("Split rows do not reconstruct the labeled dataset")

    save_data_audit(
        train_df,
        external_test_df,
        train,
        validation,
        internal_test,
        tables_dir,
        metrics_dir,
    )
    plot_split_distribution(tables_dir, plots_dir)

    input_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "train_data.csv": {"sha256": sha256_file(raw_train_path), "rows": len(train_df)},
            "test_data_unlabeled.csv": {
                "sha256": sha256_file(external_test_path),
                "rows": len(external_test_df),
            },
        },
    }
    # The repository does not redistribute teacher reference files. The output
    # contract remains one integer label per row, without a header or index.
    if teacher_sample_path.exists():
        with teacher_sample_path.open("rb") as sample_stream:
            line_count = sum(1 for _ in sample_stream)
        input_manifest["files"]["teacher_predictions_example.csv"] = {
            "sha256": sha256_file(teacher_sample_path),
            "line_count": line_count,
        }
    write_json(metrics_dir / "input_manifest.json", input_manifest)

    results, feature_table = run_validation_grid(train, validation, config, logger)
    results.to_csv(tables_dir / "experiment_results.csv", index=False)
    feature_table.to_csv(tables_dir / "feature_statistics.csv", index=False)
    build_innovation_ablation(results).to_csv(
        tables_dir / "innovation_ablation.csv", index=False
    )
    ranked = results.sort_values(
        ["val_macro_f1", "val_accuracy", "val_weighted_f1", "total_seconds"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    ranked.head(10).to_csv(tables_dir / "validation_top10.csv", index=False)
    plot_validation_results(results, plots_dir)

    winner = select_validation_winner(results)
    feature_name = str(winner["feature_config"])
    model_name = str(winner["model"])
    parameter_value = float(winner["parameter_value"])
    selection = {
        "selected_experiment_id": str(winner["experiment_id"]),
        "selection_metric": config["selection_metric"],
        "feature_config": feature_name,
        "model": model_name,
        "model_display_name": str(winner["model_display_name"]),
        "parameter_name": str(winner["parameter_name"]),
        "parameter_value": parameter_value,
        "validation_metrics": {
            key.removeprefix("val_"): float(winner[key])
            for key in winner.index
            if key.startswith("val_")
        },
        "selected_using_validation_only": True,
        "internal_test_was_untouched_until_after_lock": True,
        "tie_breakers": ["val_accuracy", "val_weighted_f1", "lower_total_seconds"],
    }
    write_json(metrics_dir / "model_selection.json", selection)
    logger.info("Locked winner before internal test: %s", selection["selected_experiment_id"])

    development = pd.concat([train, validation], ignore_index=True)
    locked_pipeline = build_pipeline(config, feature_name, model_name, parameter_value)
    locked_fit_start = time.perf_counter()
    locked_pipeline.fit(development["text"], development["target"])
    locked_fit_seconds = time.perf_counter() - locked_fit_start
    internal_pred = locked_pipeline.predict(internal_test["text"])
    internal_metrics = classification_metrics(
        internal_test["target"].to_numpy(), np.asarray(internal_pred)
    )
    train_pred = locked_pipeline.predict(development["text"])
    development_metrics = classification_metrics(
        development["target"].to_numpy(), np.asarray(train_pred)
    )

    labels = sorted(int(x) for x in train_df["target"].unique())
    report = classification_report(
        internal_test["target"],
        internal_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose().reset_index(names="label")
    report_frame.to_csv(tables_dir / "classification_report.csv", index=False)
    cm = confusion_matrix(internal_test["target"], internal_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        tables_dir / "confusion_matrix.csv", index_label="true_label"
    )
    plot_confusion(cm, labels, plots_dir / "confusion_matrix.png")

    error_mask = internal_test["target"].to_numpy() != np.asarray(internal_pred)
    error_rows = internal_test.loc[error_mask, ["source_row_id", "target", "text"]].copy()
    error_rows["predicted"] = np.asarray(internal_pred)[error_mask]
    error_rows["text_preview"] = (
        error_rows["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 500)
    )
    error_rows.drop(columns="text").head(100).to_csv(
        tables_dir / "error_analysis_samples.csv", index=False
    )

    final_pipeline = build_pipeline(config, feature_name, model_name, parameter_value)
    final_fit_start = time.perf_counter()
    final_pipeline.fit(train_df["text"], train_df["target"])
    final_fit_seconds = time.perf_counter() - final_fit_start
    external_predictions = final_pipeline.predict(external_test_df["text"])
    submission_path = PROJECT_ROOT / "submission" / "predictions.csv"
    pd.Series(external_predictions, dtype="int64").to_csv(
        submission_path, index=False, header=False, lineterminator="\n"
    )
    joblib.dump(final_pipeline, PROJECT_ROOT / "models" / "final_pipeline.joblib")
    prediction_distribution = (
        pd.Series(external_predictions, name="target")
        .value_counts()
        .sort_index()
        .rename_axis("target")
        .reset_index(name="count")
    )
    prediction_distribution.to_csv(
        tables_dir / "external_prediction_distribution.csv", index=False
    )

    final_payload = {
        "locked_spec": selection,
        "development_fit_rows": int(len(development)),
        "internal_test_rows": int(len(internal_test)),
        "internal_test_metrics": internal_metrics,
        "development_fit_metrics": development_metrics,
        "locked_development_fit_seconds": locked_fit_seconds,
        "final_refit_rows": int(len(train_df)),
        "final_refit_seconds": final_fit_seconds,
        "external_prediction_rows": int(len(external_predictions)),
        "external_predictions_sha256": sha256_file(submission_path),
        "test_policy": (
            "Model/feature/hyperparameters were selected only on validation. "
            "The labeled internal test was evaluated once after locking. "
            "The external unlabeled test was used only for final prediction."
        ),
    }
    write_json(metrics_dir / "final_metrics.json", final_payload)

    model_metadata = {
        "feature_config": feature_name,
        "feature_parameters": config["feature_configs"][feature_name],
        "model": model_name,
        "model_display_name": config["models"][model_name]["display_name"],
        "parameter_name": config["models"][model_name]["parameter_name"],
        "parameter_value": parameter_value,
        "trained_on_all_labeled_rows": int(len(train_df)),
        "random_seed": int(config["random_seed"]),
        "labels": labels,
        "input_train_sha256": sha256_file(raw_train_path),
        "external_test_sha256": sha256_file(external_test_path),
        "predictions_sha256": sha256_file(submission_path),
    }
    write_json(PROJECT_ROOT / "models" / "model_metadata.json", model_metadata)

    run_summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - run_start,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "experiment_count": int(len(results)),
        "selected_experiment_id": selection["selected_experiment_id"],
        "submission": str(submission_path.relative_to(PROJECT_ROOT)),
        "submission_sha256": sha256_file(submission_path),
    }
    write_json(metrics_dir / "run_summary.json", run_summary)

    summary_lines = [
        "# 实验结果速览（报告素材，不是实验报告）",
        "",
        f"- 验证实验总数：{len(results)}",
        f"- 验证集最佳方案：`{selection['selected_experiment_id']}`",
        f"- 验证集 Accuracy：{selection['validation_metrics']['accuracy']:.4f}",
        f"- 验证集 Macro-F1：{selection['validation_metrics']['macro_f1']:.4f}",
        f"- 封存内部测试集 Accuracy：{internal_metrics['accuracy']:.4f}",
        f"- 封存内部测试集 Macro-F1：{internal_metrics['macro_f1']:.4f}",
        f"- 最终外部预测行数：{len(external_predictions)}",
        f"- predictions.csv SHA-256：`{sha256_file(submission_path)}`",
        "",
        "完整逐组结果见 `results/experiment_results.csv`，混淆矩阵与图表见 `results/plots/`。",
    ]
    (PROJECT_ROOT / "RESULTS_SUMMARY.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    logger.info("Completed all experiments in %.2f seconds", run_summary["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

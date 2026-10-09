from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    strip_accents_ascii,
    strip_accents_unicode,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or PROJECT_ROOT / "configs" / "experiment_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def set_global_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_HEADER_SPLIT_RE = re.compile(r"\r?\n\s*\r?\n", flags=re.MULTILINE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_WS_RE = re.compile(r"[ \t]+")


def clean_newsgroup_text(text: str) -> str:
    """Deterministically remove dataset-specific metadata without using labels."""

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    parts = _HEADER_SPLIT_RE.split(normalized, maxsplit=1)
    header = parts[0]
    body = parts[1] if len(parts) == 2 else normalized

    subject_parts: list[str] = []
    for line in header.splitlines():
        if line.lower().startswith("subject:"):
            subject_parts.append(line.split(":", 1)[1].strip())

    kept: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if stripped == "--":
            break
        if stripped.startswith(">"):
            continue
        if lower.startswith("in article ") or lower.endswith(" writes:"):
            continue
        if re.fullmatch(r"[-_=*]{4,}", stripped):
            continue
        line = _EMAIL_RE.sub(" ", line)
        line = _URL_RE.sub(" ", line)
        line = _WS_RE.sub(" ", line).strip()
        if line:
            kept.append(line)

    cleaned = "\n".join(subject_parts + kept).strip()
    return cleaned if cleaned else normalized.strip()


@dataclass(frozen=True)
class CleanedTextPreprocessor:
    """Compose deterministic cleaning with sklearn-equivalent normalization.

    A custom ``TfidfVectorizer.preprocessor`` replaces sklearn's built-in
    lowercase and accent-stripping stage. Keeping those operations here makes
    the cleaned ablation differ from ``tuned_word_tfidf`` only by the cleaning
    rules in ``clean_newsgroup_text``.
    """

    lowercase: bool
    strip_accents: str | None

    def __call__(self, text: str) -> str:
        cleaned = clean_newsgroup_text(text)
        if self.lowercase:
            cleaned = cleaned.lower()
        if self.strip_accents == "unicode":
            cleaned = strip_accents_unicode(cleaned)
        elif self.strip_accents == "ascii":
            cleaned = strip_accents_ascii(cleaned)
        elif self.strip_accents is not None:
            raise ValueError(f"Unsupported strip_accents value: {self.strip_accents}")
        return cleaned


def build_vectorizer(config: dict[str, Any], feature_name: str) -> TfidfVectorizer:
    params = dict(config["feature_configs"][feature_name])
    params.pop("description", None)
    preprocessor_name = params.pop("preprocessor")
    params["ngram_range"] = tuple(params["ngram_range"])
    if preprocessor_name == "newsgroup_clean":
        configured_lowercase = bool(params.pop("lowercase"))
        configured_strip_accents = params.pop("strip_accents")
        params["preprocessor"] = CleanedTextPreprocessor(
            lowercase=configured_lowercase,
            strip_accents=configured_strip_accents,
        )
        # These stages are deliberately handled by the composed preprocessor.
        # Neutral values make the effective vectorizer state unambiguous.
        params["lowercase"] = False
        params["strip_accents"] = None
    elif preprocessor_name != "identity":
        raise ValueError(f"Unknown preprocessor: {preprocessor_name}")
    return TfidfVectorizer(**params)


def build_model(
    config: dict[str, Any], model_name: str, parameter_value: float, seed: int
):
    model_cfg = config["models"][model_name]
    if model_name == "multinomial_nb":
        return MultinomialNB(alpha=parameter_value)
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=parameter_value,
            max_iter=int(model_cfg["max_iter"]),
            random_state=seed,
            solver="lbfgs",
        )
    if model_name == "linear_svm":
        return LinearSVC(
            C=parameter_value,
            max_iter=int(model_cfg["max_iter"]),
            random_state=seed,
            dual="auto",
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(
    config: dict[str, Any], feature_name: str, model_name: str, parameter_value: float
) -> Pipeline:
    seed = int(config["random_seed"])
    return Pipeline(
        [
            ("tfidf", build_vectorizer(config, feature_name)),
            ("classifier", build_model(config, model_name, parameter_value, seed)),
        ]
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def validate_raw_inputs(train_df: pd.DataFrame, external_test_df: pd.DataFrame) -> None:
    if list(train_df.columns) != ["text", "target"]:
        raise ValueError(f"Unexpected train columns: {train_df.columns.tolist()}")
    if list(external_test_df.columns) != ["text"]:
        raise ValueError(f"Unexpected external test columns: {external_test_df.columns.tolist()}")
    if train_df.isna().any().any() or external_test_df.isna().any().any():
        raise ValueError("Raw data contains missing values.")
    if (train_df["text"].astype(str).str.strip() == "").any():
        raise ValueError("Training data contains empty text.")
    if (external_test_df["text"].astype(str).str.strip() == "").any():
        raise ValueError("External test data contains empty text.")
    labels = sorted(train_df["target"].unique().tolist())
    if labels != list(range(10)):
        raise ValueError(f"Expected labels 0..9, got {labels}")

import unittest
from pathlib import Path

import pandas as pd

from src.lab1_common import (
    PROJECT_ROOT,
    CleanedTextPreprocessor,
    build_vectorizer,
    clean_newsgroup_text,
    load_config,
)


class ProjectTests(unittest.TestCase):
    def test_cleaner_removes_metadata_and_quotes(self):
        sample = "From: a@example.com\nSubject: Useful topic\n\n> quoted text\nReal body"
        cleaned = clean_newsgroup_text(sample)
        self.assertIn("Useful topic", cleaned)
        self.assertIn("Real body", cleaned)
        self.assertNotIn("a@example.com", cleaned)
        self.assertNotIn("quoted text", cleaned)

    def test_config_ratios_sum_to_one(self):
        config = load_config()
        self.assertAlmostEqual(sum(config["split_ratios"].values()), 1.0)
        self.assertEqual(config["random_seed"], 42)

    def test_cleaned_ablation_preserves_tfidf_normalization(self):
        config = load_config()
        vectorizer = build_vectorizer(config, "cleaned_tuned_word_tfidf")
        self.assertIsInstance(vectorizer.preprocessor, CleanedTextPreprocessor)
        self.assertTrue(vectorizer.preprocessor.lowercase)
        self.assertEqual(vectorizer.preprocessor.strip_accents, "unicode")
        self.assertFalse(vectorizer.lowercase)
        self.assertIsNone(vectorizer.strip_accents)

        sample = "Subject: Caf\u00e9 GPU\n\nGPU gpu Caf\u00e9 cafe"
        tokens = vectorizer.build_analyzer()(sample)
        self.assertIn("cafe", tokens)
        self.assertIn("gpu", tokens)
        self.assertNotIn("GPU", tokens)
        self.assertNotIn("caf\u00e9", tokens)

    def test_generated_splits_are_disjoint_when_present(self):
        split_dir = PROJECT_ROOT / "data" / "splits"
        paths = [split_dir / f"{name}.csv" for name in ("train", "validation", "internal_test")]
        if not all(path.exists() for path in paths):
            self.skipTest("experiment has not generated splits yet")
        ids = [set(pd.read_csv(path)["source_row_id"]) for path in paths]
        self.assertFalse(ids[0] & ids[1])
        self.assertFalse(ids[0] & ids[2])
        self.assertFalse(ids[1] & ids[2])

    def test_prediction_format_when_present(self):
        path = PROJECT_ROOT / "submission" / "predictions.csv"
        if not path.exists():
            self.skipTest("experiment has not generated predictions yet")
        frame = pd.read_csv(path, header=None)
        self.assertEqual(frame.shape[1], 1)
        self.assertGreater(len(frame), 0)
        self.assertTrue(frame[0].notna().all())
        self.assertTrue((frame[0] == frame[0].astype(int)).all())
        external_path = PROJECT_ROOT / "data/raw/test_data_unlabeled.csv"
        if external_path.exists():
            external = pd.read_csv(external_path)
            self.assertEqual(len(frame), len(external))
        self.assertTrue(set(frame[0].unique()).issubset(set(range(10))))


if __name__ == "__main__":
    unittest.main()

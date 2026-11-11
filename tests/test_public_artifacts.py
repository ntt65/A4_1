"""Exercise actual exporters to keep corpus text out of public measurements."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.bonus.runner import run_bonus
from src.classifier import train_and_evaluate_classifier
from src.preprocess import TextPreprocessor
from src.tfidf import CustomTfidfVectorizer


class PublicArtifactTests(unittest.TestCase):
    def setUp(self):
        self.sentinels = ("PrivateNameAlice", "PrivateNameRobert")
        self.documents = [
            "PrivateNameAlice space satellite launch orbit",
            "PrivateNameAlice rocket space mission",
            "PrivateNameAlice satellite space station",
            "PrivateNameRobert computer graphics rendering image",
            "PrivateNameRobert graphics pixel display",
            "PrivateNameRobert computer rendering graphics",
        ]
        self.labels = np.array([0, 0, 0, 1, 1, 1])
        self.vectorizer = CustomTfidfVectorizer(TextPreprocessor(stopwords=set()))
        self.features = self.vectorizer.fit_transform(self.documents)
        self.test_documents = [self.documents[0], self.documents[3]]
        self.test_features = self.vectorizer.transform(self.test_documents)

    def assert_public_json(self, path):
        text = path.read_text(encoding="utf-8")
        for sentinel in self.sentinels:
            self.assertNotIn(sentinel.lower(), text.lower())
        data = json.loads(text)
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(data["public_export"]["full_corpus_review"], "pending")

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse({"text", "snippet", "term"} & value.keys())
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        inspect(data)
        return data

    def test_classifier_export_preserves_numeric_error_evidence_without_corpus_text(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "confusion.png"
            classifier, accuracy, macro_f1, matrix = train_and_evaluate_classifier(
                self.features, self.labels, self.test_features, np.array([1, 0]),
                ["space", "graphics"], output_img=output,
                test_docs=self.test_documents, vectorizer=self.vectorizer,
            )
            data = self.assert_public_json(output.with_name("classifier_metrics.json"))
            self.assertEqual(len(data["misclassified_examples"]), 2)
            self.assertEqual(data["accuracy"], accuracy)
            self.assertEqual(data["macro_f1"], macro_f1)
            self.assertEqual(data["confusion_matrix"], matrix.tolist())
            self.assertTrue(output.exists())
            self.assertEqual(classifier.classes_.tolist(), [0, 1])
            for example in data["misclassified_examples"]:
                self.assertGreater(example["decision_margin"], 0)
                features = example["top_predicted_features"] + example["top_actual_features"]
                self.assertTrue(features)
                self.assertTrue(all(isinstance(feature["feature_id"], int) for feature in features))
                self.assertTrue(all(isinstance(feature["contribution"], float) for feature in features))

    def test_bonus_export_preserves_rankings_without_document_snippets(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("src.bonus.runner.ARTIFACTS", Path(folder)):
                with contextlib.redirect_stdout(io.StringIO()):
                    report = run_bonus(
                        self.vectorizer, self.documents, self.labels,
                        self.test_documents, np.array([0, 1]), ["space", "graphics"],
                        X_train=self.features,
                    )
            data = self.assert_public_json(Path(folder) / "bonus_metrics.json")
            self.assertEqual(data, report)
            self.assertEqual(data["inverted_index"]["top5_rank_match_fraction"], 1.0)
            self.assertLess(data["bm25_toy"]["max_absolute_error"], 1e-6)
            for comparison in data["bm25_search_comparison"]:
                for result in comparison["tfidf"] + comparison["bm25"]:
                    self.assertEqual(set(result), {"doc_id", "score"})


if __name__ == "__main__":
    unittest.main()

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.dataset import clean_text
from src.preprocess import TextPreprocessor, load_stopwords
from src.search_engine import SearchEngine, rank_scores
from src.tfidf import CustomTfidfVectorizer


class PreprocessingTests(unittest.TestCase):
    def test_contact_masking_preserves_surrounding_text(self):
        text = "Contact alice.smith+work@example.org or 212-555-0199 about orbit."
        cleaned = clean_text(text)
        self.assertNotIn("alice.smith+work@example.org", cleaned)
        self.assertNotIn("212-555-0199", cleaned)
        self.assertIn("orbit", cleaned)

    def test_default_stopwords_are_independent_of_working_directory(self):
        expected = load_stopwords()
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(load_stopwords(), expected)
                self.assertEqual(TextPreprocessor().stopwords, expected)
            finally:
                os.chdir(original_directory)

    def test_token_pattern_and_custom_stopwords(self):
        processor = TextPreprocessor(stopwords={"the", "and"})
        self.assertEqual(
            processor.tokenize("The SPACE and satellite, a 3D_x!"),
            ["space", "satellite", "3d_x"],
        )


class TfidfTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor(stopwords={"the", "and"})
        self.documents = ["the satellite orbit orbit", "baseball and bat", "orbit bat", ""]

    def test_sklearn_equivalence_for_train_and_unseen_documents(self):
        custom = CustomTfidfVectorizer(self.preprocessor)
        reference = TfidfVectorizer(
            analyzer=self.preprocessor.tokenize,
            dtype=np.float64,
            smooth_idf=True,
            sublinear_tf=False,
            norm="l2",
        )
        actual_train = custom.fit_transform(self.documents)
        expected_train = reference.fit_transform(self.documents).toarray()
        self.assertEqual(custom.vocabulary_, reference.vocabulary_)
        np.testing.assert_allclose(custom.idf_, reference.idf_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual_train, expected_train, rtol=0, atol=1e-12)
        held_out = ["satellite unseenword", "the and", "", "unseenword"]
        np.testing.assert_allclose(
            custom.transform(held_out), reference.transform(held_out).toarray(),
            rtol=0, atol=1e-12,
        )

    def test_transform_preserves_training_vocabulary_and_idf(self):
        vectorizer = CustomTfidfVectorizer(self.preprocessor).fit(self.documents)
        vocabulary = vectorizer.vocabulary_.copy()
        idf = vectorizer.idf_.copy()
        vectorizer.transform(["testonlyword testonlyword orbit"])
        self.assertEqual(vectorizer.vocabulary_, vocabulary)
        np.testing.assert_array_equal(vectorizer.idf_, idf)
        self.assertNotIn("testonlyword", vectorizer.vocabulary_)

    def test_empty_transform_and_oov_are_zero_safe(self):
        vectorizer = CustomTfidfVectorizer(self.preprocessor).fit(self.documents)
        self.assertEqual(vectorizer.transform([]).shape, (0, len(vectorizer.vocabulary_)))
        transformed = vectorizer.transform(["", "the and", "unseenword"])
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_array_equal(transformed, np.zeros_like(transformed))

    def test_transform_requires_fitting(self):
        with self.assertRaises((ValueError, RuntimeError)):
            CustomTfidfVectorizer(self.preprocessor).transform(["orbit"])

    def test_fit_rejects_empty_vocabulary(self):
        for documents in ([], ["", "the and"]):
            with self.subTest(documents=documents):
                with self.assertRaises(ValueError):
                    CustomTfidfVectorizer(self.preprocessor).fit(documents)

    def test_refitting_replaces_prior_vocabulary(self):
        vectorizer = CustomTfidfVectorizer(self.preprocessor).fit(["satellite orbit"])
        vectorizer.fit(["baseball bat"])
        self.assertEqual(set(vectorizer.vocabulary_), {"baseball", "bat"})
        np.testing.assert_array_equal(vectorizer.transform(["satellite orbit"]), [[0, 0]])


class SearchTests(unittest.TestCase):
    def setUp(self):
        documents = ["orbit satellite", "orbit satellite", "baseball bat", ""]
        vectorizer = CustomTfidfVectorizer(TextPreprocessor(stopwords=set())).fit(documents)
        self.engine = SearchEngine(vectorizer, documents)

    def test_query_ties_are_ordered_by_document_id(self):
        results = self.engine.search("orbit", top_k=1)
        self.assertEqual([row[1] for row in results], [0])
        self.assertEqual([row[1] for row in self.engine.search("orbit", top_k=10)], [0, 1])

    def test_empty_and_oov_queries_return_no_results(self):
        self.assertEqual(self.engine.search(""), [])
        self.assertEqual(self.engine.search("unseenword"), [])

    def test_top_k_boundary_ties_and_small_positive_scores(self):
        scores = np.array([0.5, 0.9, 0.5, 0.5, 0.0, 1e-14])
        self.assertEqual(list(rank_scores(scores, 2)), [1, 0])
        self.assertEqual(list(rank_scores(scores, 10)), [1, 0, 2, 3, 5])

    def test_non_positive_top_k_is_rejected(self):
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    self.engine.search("orbit", top_k=top_k)

    def test_non_integer_top_k_is_rejected(self):
        for top_k in (1.5, "2", True):
            with self.subTest(top_k=top_k):
                with self.assertRaises((TypeError, ValueError)):
                    self.engine.search("orbit", top_k=top_k)

    def test_persisted_engine_preserves_search_results(self):
        expected = self.engine.search("orbit satellite", top_k=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "index.pkl"
            self.engine.save(path)
            loaded = SearchEngine.load(path)
            self.assertEqual(loaded.search("orbit satellite", top_k=2), expected)
            self.assertEqual(loaded.search("unseenword"), [])


if __name__ == "__main__":
    unittest.main()

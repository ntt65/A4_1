"""Regression coverage for zero-safe TF, BM25 arithmetic and index rebuilding."""

import unittest

import numpy as np

from src.bonus.bm25 import BM25Okapi, verify_bm25_toy_example
from src.bonus.inverted_index import InvertedIndex
from src.bonus.tf_variations import apply_tf_variant, compute_double_norm_tf
from src.preprocess import TextPreprocessor
from src.search_engine import rank_scores


class BonusRegressionTests(unittest.TestCase):
    def test_double_normalization_preserves_missing_terms_and_empty_documents(self):
        actual = compute_double_norm_tf(
            ["cat cat dog", "", "unknown"],
            TextPreprocessor(stopwords=set()),
            {"cat": 0, "dog": 1, "sat": 2},
        )
        np.testing.assert_allclose(actual, [[1, 0.75, 0], [0, 0, 0], [0, 0, 0]])

    def test_log_normalization_uses_log_one_plus_tf(self):
        counts = np.array([[0.0, 1.0, 3.0]])
        actual = apply_tf_variant(counts, "log")
        np.testing.assert_allclose(actual, [[0, np.log(2), np.log(4)]])

    def test_bm25_matches_independent_hand_calculation(self):
        report = verify_bm25_toy_example()
        self.assertLess(report["max_absolute_error"], 1e-12)

    def test_bm25_empty_corpus_oov_and_repeated_query(self):
        self.assertEqual(BM25Okapi([]).score(["cat"]).size, 0)
        np.testing.assert_array_equal(BM25Okapi([[], []]).score(["cat"]), [0, 0])
        model = BM25Okapi([["cat"], ["dog"]])
        np.testing.assert_allclose(model.score(["cat", "cat"]), 2 * model.score(["cat"]))
        np.testing.assert_array_equal(model.score(["unknown"]), [0, 0])

    def test_bm25_full_length_normalization_keeps_empty_documents_zero(self):
        model = BM25Okapi([[], ["cat"]], b=1.0)
        with np.errstate(divide="raise", invalid="raise"):
            scores = model.score(["cat"])
        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[1], 0.0)
        self.assertTrue(np.isfinite(scores).all())

    def test_inverted_rebuild_replaces_old_postings(self):
        matrix = np.array([[0.6, 0.8], [1.0, 0.0], [0.6, 0.8]])
        query = np.array([1.0, 0.5])
        index = InvertedIndex(3)
        index.build_from_dense(matrix)
        first = index.search(query)
        index.build_from_dense(matrix)
        np.testing.assert_allclose(index.search(query), first)
        np.testing.assert_allclose(first, matrix @ query)
        replacement = np.zeros_like(matrix)
        index.build_from_dense(replacement)
        np.testing.assert_array_equal(index.search(query), np.zeros(3))

    def test_rank_policy_matches_at_tied_topk_boundary(self):
        matrix = np.array([[1.0, 0], [1.0, 0], [1.0, 0], [0, 1.0]])
        query = np.array([1.0, 0])
        index = InvertedIndex(4).build_from_dense(matrix)
        np.testing.assert_array_equal(rank_scores(matrix @ query, 2), [0, 1])
        np.testing.assert_array_equal(rank_scores(index.search(query), 2), [0, 1])
        self.assertEqual(rank_scores(index.search(np.zeros(2)), 5).size, 0)

    def test_index_rejects_wrong_document_count_and_query_shape(self):
        index = InvertedIndex(2)
        with self.assertRaises(ValueError):
            index.build_from_dense(np.zeros((3, 2)))
        index.build_from_dense(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            index.search(np.zeros(3))


if __name__ == "__main__":
    unittest.main()

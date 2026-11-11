"""NumPy BM25 with positive Robertson-Sparck Jones IDF."""

import math
from collections import Counter

import numpy as np


class BM25Okapi:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        if not np.isfinite(k1) or k1 <= 0 or not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lens = np.array([len(doc) for doc in corpus_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_lens.mean()) if self.corpus_size else 0.0
        self.doc_freqs = [dict(Counter(document)) for document in corpus_tokens]
        self.df = Counter(term for document in self.doc_freqs for term in document)
        self.idf = {
            term: float(np.log1p((self.corpus_size - count + 0.5) / (count + 0.5)))
            for term, count in self.df.items()
        }

    def score(self, query_tokens):
        """Repeated query terms contribute repeatedly, matching query term counts."""
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        if self.avgdl == 0:
            return scores
        length_penalty = self.k1 * (1.0 - self.b + self.b * self.doc_lens / self.avgdl)
        for term, query_count in Counter(query_tokens).items():
            if term not in self.idf:
                continue
            frequencies = np.array(
                [document.get(term, 0) for document in self.doc_freqs],
                dtype=np.float64,
            )
            numerator = query_count * self.idf[term] * frequencies * (self.k1 + 1.0)
            denominator = frequencies + length_penalty
            scores += np.divide(
                numerator, denominator, out=np.zeros_like(frequencies),
                where=denominator != 0,
            )
        return scores


def verify_bm25_toy_example():
    corpus = [["cat", "sat", "sat"], ["cat", "dog"], ["dog"]]
    scores = BM25Okapi(corpus, k1=1.5, b=0.75).score(["cat", "sat"])
    # Hand calculation: N=3, avgdl=2, df(cat)=2, df(sat)=1.
    cat_idf = math.log(1.0 + 1.5 / 2.5)
    sat_idf = math.log(1.0 + 2.5 / 1.5)
    expected = np.array([
        cat_idf * 2.5 / (1.0 + 1.5 * 1.375)
        + sat_idf * 5.0 / (2.0 + 1.5 * 1.375),
        cat_idf,
        0.0,
    ])
    max_error = float(np.max(np.abs(scores - expected)))
    if max_error >= 1e-6:
        raise AssertionError(f"BM25 toy error exceeds tolerance: {max_error}")
    return {
        "corpus": corpus,
        "query": ["cat", "sat"],
        "k1": 1.5,
        "b": 0.75,
        "expected_scores": expected.tolist(),
        "actual_scores": scores.tolist(),
        "max_absolute_error": max_error,
        "tolerance": 1e-6,
        "passed": True,
    }


def test_bm25_toy_example():
    """Retain the supplied helper's original equal-length corpus and query."""
    corpus = [["cat", "sat"], ["cat", "dog"], ["dog", "sat"]]
    return BM25Okapi(corpus, k1=1.5, b=0.75).score(["cat"])

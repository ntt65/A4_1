"""Posting lists over nonzero, normalized TF-IDF document weights."""

import numpy as np


class InvertedIndex:
    def __init__(self, n_docs):
        if n_docs < 0:
            raise ValueError("Document count must be nonnegative")
        self.n_docs = n_docs
        self.n_features = None
        self.index = {}

    def build_from_dense(self, tfidf_matrix):
        matrix = np.asarray(tfidf_matrix)
        if matrix.ndim != 2 or matrix.shape[0] != self.n_docs:
            raise ValueError("TF-IDF matrix must have one row per document")
        postings = {}
        for document_id, row in enumerate(matrix):
            for term_id in np.flatnonzero(row):
                postings.setdefault(int(term_id), []).append(
                    (document_id, float(row[term_id]))
                )
        self.index = postings
        self.n_features = matrix.shape[1]
        return self

    def search(self, q_vec):
        query = np.asarray(q_vec, dtype=np.float64)
        if self.n_features is None:
            raise ValueError("Build the inverted index before searching")
        if query.ndim != 1 or len(query) != self.n_features:
            raise ValueError("Query vector must match the indexed vocabulary")
        if not np.isfinite(query).all():
            raise ValueError("Query vector must contain finite values")
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for term_id in np.flatnonzero(query):
            for document_id, weight in self.index.get(int(term_id), ()):
                scores[document_id] += weight * query[term_id]
        return scores

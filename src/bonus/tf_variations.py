"""Term-frequency variants with a shared, fixed training vocabulary."""

import numpy as np


def compute_term_counts(raw_documents, preprocessor, vocabulary):
    counts = np.zeros((len(raw_documents), len(vocabulary)), dtype=np.float64)
    for row, document in enumerate(raw_documents):
        for token in preprocessor.tokenize(document):
            column = vocabulary.get(token)
            if column is not None:
                counts[row, column] += 1.0
    return counts


def apply_tf_variant(counts, variant="raw", K=0.5):
    """Modify a count matrix in place and preserve every absent term as zero."""
    if variant == "raw":
        return counts
    if variant == "log":
        np.log1p(counts, out=counts)
        return counts
    if variant != "double":
        raise ValueError("TF variant must be raw, log, or double")
    if not 0 <= K <= 1:
        raise ValueError("Double-normalization K must be between 0 and 1")
    for row in counts:
        nonzero = row > 0
        maximum = row.max(initial=0.0)
        if maximum > 0:
            row[nonzero] = K + (1.0 - K) * row[nonzero] / maximum
    return counts


def compute_double_norm_tf(raw_documents, preprocessor, vocabulary, K=0.5):
    counts = compute_term_counts(raw_documents, preprocessor, vocabulary)
    return apply_tf_variant(counts, "double", K=K)


def compute_variant_tfidf(raw_documents, vectorizer, variant):
    counts = vectorizer.count_matrix(raw_documents)
    apply_tf_variant(counts, variant)
    counts *= vectorizer.idf_
    norms = np.linalg.norm(counts, axis=1, keepdims=True)
    np.divide(counts, norms, out=counts, where=norms != 0)
    return counts

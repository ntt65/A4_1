import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.dataset import load_data
from src.preprocess import TextPreprocessor
from src.tfidf import CustomTfidfVectorizer


ROOT = Path(__file__).resolve().parent


def main():
    print("=== NumPy TF-IDF vs Scikit-learn ===", flush=True)
    (train_docs, _), (test_docs, _), _ = load_data()
    preprocessor = TextPreprocessor()
    custom = CustomTfidfVectorizer(preprocessor).fit(train_docs)
    reference = TfidfVectorizer(
        analyzer=preprocessor.tokenize, smooth_idf=True,
        norm="l2", sublinear_tf=False, dtype=np.float64,
    ).fit(train_docs)
    assert custom.vocabulary_ == reference.vocabulary_, "Vocabulary or column order mismatch"
    counter = CountVectorizer(
        analyzer=preprocessor.tokenize, vocabulary=reference.vocabulary_,
    )
    reference_counts = counter.transform(train_docs)
    reference_df = np.asarray((reference_counts > 0).sum(axis=0)).ravel()
    np.testing.assert_array_equal(custom.df_, reference_df)
    np.testing.assert_allclose(custom.idf_, reference.idf_, rtol=0, atol=1e-12)
    idf_error = float(np.max(np.abs(custom.idf_ - reference.idf_)))
    results = {}
    groups = {
        "train": train_docs, "test": test_docs,
        "queries": ["space satellite launch orbit", "graphics rendering", "", "the and", "zzzzzzunseenword"],
    }
    for name, docs in groups.items():
        maximum, error_sum, count = 0.0, 0.0, 0
        for start in range(0, len(docs), 64):
            batch = docs[start:start + 64]
            counts = custom.count_matrix(batch)
            np.testing.assert_array_equal(counts, counter.transform(batch).toarray())
            actual = custom.transform(batch)
            expected = reference.transform(batch).toarray()
            assert actual.shape == expected.shape
            errors = np.abs(actual - expected)
            maximum = max(maximum, float(errors.max(initial=0)))
            error_sum += float(errors.sum())
            count += errors.size
        if maximum >= 1e-6:
            raise AssertionError(f"{name}: TF-IDF error {maximum} exceeds tolerance")
        results[name] = {
            "shape": [len(docs), len(custom.vocabulary_)],
            "max_absolute_error": maximum, "mean_absolute_error": error_sum / max(count, 1),
        }
        print(f"{name}: shape={results[name]['shape']} max_error={maximum:.4e} mean_error={results[name]['mean_absolute_error']:.4e}", flush=True)
    assert np.all(custom.transform(["", "the and", "zzzzzzunseenword"]) == 0)
    result = {
        "status": "PASS", "threshold": 1e-6, "idf_max_error": idf_error,
        "settings": {"smooth_idf": True, "sublinear_tf": False, "norm": "l2", "dtype": "float64", "analyzer": "TextPreprocessor.tokenize"},
        "groups": results,
    }
    (ROOT / "artifacts" / "tfidf_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    print(f"Vocabulary/TF/DF/IDF and Train/Test/query checks PASS; IDF error={idf_error:.4e}")
    return result


if __name__ == "__main__":
    main()

"""Run the three bonus experiments on the same training split and queries."""

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from src.bonus.bm25 import BM25Okapi, verify_bm25_toy_example
from src.bonus.inverted_index import InvertedIndex
from src.bonus.tf_variations import apply_tf_variant, compute_variant_tfidf
from src.search_engine import rank_scores


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
QUERIES = [
    "space satellite launch orbit",
    "computer graphics rendering image",
    "baseball pitcher season game",
    "middle east peace politics",
]


def _top_results(scores, documents, top_k=5):
    return [
        {
            "doc_id": int(document_id),
            "score": float(scores[document_id]),
            "snippet": " ".join(documents[document_id].split())[:150],
        }
        for document_id in rank_scores(scores, top_k)
    ]


def _normalize_with_idf(counts, vectorizer, variant):
    apply_tf_variant(counts, variant)
    counts *= vectorizer.idf_
    norms = np.linalg.norm(counts, axis=1, keepdims=True)
    np.divide(counts, norms, out=counts, where=norms != 0)
    return counts


def _compare_tf_variants(vec, train_docs, y_train, test_docs, y_test, raw_matrix):
    results = {}
    test_counts = vec.count_matrix(test_docs)
    for variant in ("raw", "log", "double"):
        print(f"[TF variant] {variant}", flush=True)
        train_matrix = raw_matrix
        if variant != "raw":
            train_matrix = compute_variant_tfidf(train_docs, vec, variant)
        test_matrix = _normalize_with_idf(test_counts.copy(), vec, variant)
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(csr_matrix(train_matrix), y_train)
        prediction = classifier.predict(csr_matrix(test_matrix))
        rankings = {}
        for query in QUERIES:
            query_vector = compute_variant_tfidf([query], vec, variant)[0]
            rankings[query] = _top_results(train_matrix @ query_vector, train_docs)
        results[variant] = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "macro_f1": float(f1_score(y_test, prediction, average="macro")),
            "search_results": rankings,
        }
        print(
            f"  Accuracy={results[variant]['accuracy']:.6f}, "
            f"Macro-F1={results[variant]['macro_f1']:.6f}",
            flush=True,
        )
        del train_matrix, test_matrix, classifier
    for variant, variant_results in results.items():
        overlaps = {}
        for query in QUERIES:
            raw_ids = {item["doc_id"] for item in results["raw"]["search_results"][query]}
            variant_ids = {item["doc_id"] for item in variant_results["search_results"][query]}
            overlaps[query] = len(raw_ids & variant_ids) / len(raw_ids) if raw_ids else 1.0
        variant_results["top5_overlap_with_raw"] = overlaps
    return results


def _benchmark_query(score_function, query_vector, repeats=3):
    rank_scores(score_function(query_vector), 5)
    timings = []
    for _ in range(repeats):
        started = perf_counter()
        rank_scores(score_function(query_vector), 5)
        timings.append((perf_counter() - started) * 1000)
    return median(timings)


def _compare_inverted_index(vec, train_docs, matrix):
    started = perf_counter()
    inverted = InvertedIndex(len(train_docs)).build_from_dense(matrix)
    build_seconds = perf_counter() - started
    experiments = []
    for query in [*QUERIES, "", "zzzzunseenquerytoken", "space space satellite"]:
        query_vector = vec.transform([query])[0]
        dense_scores = matrix @ query_vector
        inverted_scores = inverted.search(query_vector)
        max_error = float(np.max(np.abs(dense_scores - inverted_scores), initial=0.0))
        dense_rank = rank_scores(dense_scores, 5)
        inverted_rank = rank_scores(inverted_scores, 5)
        same_rank = bool(np.array_equal(dense_rank, inverted_rank))
        if max_error >= 1e-6 or not same_rank:
            raise AssertionError(f"Inverted-index equivalence failed for {query!r}")
        experiments.append({
            "query": query,
            "max_absolute_error": max_error,
            "top5_rank_equal": same_rank,
            "dense_top5": dense_rank.tolist(),
            "inverted_top5": inverted_rank.tolist(),
            "dense_query_ms": _benchmark_query(lambda vector: matrix @ vector, query_vector),
            "inverted_query_ms": _benchmark_query(inverted.search, query_vector),
        })
    return {
        "build_seconds": build_seconds,
        "query_timing_scope": "score computation and Top-5 selection, excluding tokenization/vectorization",
        "timing_repeats": 3,
        "blas_threads": 2,
        "max_absolute_error": max(item["max_absolute_error"] for item in experiments),
        "top5_rank_match_fraction": sum(item["top5_rank_equal"] for item in experiments) / len(experiments),
        "queries": experiments,
    }


def run_bonus(vec, train_docs, y_train, test_docs, y_test, target_names, X_train=None):
    """Save measured metrics and return a JSON-serializable experiment report."""
    matrix = X_train if X_train is not None else vec.transform(train_docs)
    with threadpool_limits(limits=2):
        variants = _compare_tf_variants(vec, train_docs, y_train, test_docs, y_test, matrix)
        toy = verify_bm25_toy_example()
        bm25 = BM25Okapi([vec.preprocessor.tokenize(document) for document in train_docs])
        search_comparisons = []
        for query in QUERIES:
            search_comparisons.append({
                "query": query,
                "tfidf": variants["raw"]["search_results"][query],
                "bm25": _top_results(bm25.score(vec.preprocessor.tokenize(query)), train_docs),
            })
        inverted = _compare_inverted_index(vec, train_docs, matrix)
    results = {
        "train_documents": len(train_docs),
        "test_documents": len(test_docs),
        "target_names": list(target_names),
        "random_state": 42,
        "tf_variants": variants,
        "bm25_toy": toy,
        "bm25_search_comparison": search_comparisons,
        "inverted_index": inverted,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACTS / "bonus_metrics.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"BM25 Toy Max Absolute Error: {toy['max_absolute_error']:.4e}")
    print(f"Inverted Index Max Absolute Error: {inverted['max_absolute_error']:.4e}")
    print(f"Top-5 rank match: {inverted['top5_rank_match_fraction']:.0%}")
    print(f"Index build: {inverted['build_seconds']:.6f}s")
    print(f"Bonus measurements saved to {destination}")
    return results

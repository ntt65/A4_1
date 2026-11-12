import argparse
from pathlib import Path
from typing import List, Tuple

from src.dataset import load_data
from src.search_engine import INDEX_PATH, SearchEngine
from src.tfidf import CustomTfidfVectorizer


ROOT = Path(__file__).resolve().parent
_engine = None


def init_system():
    global _engine
    (train_docs, y_train), (test_docs, y_test), target_names = load_data()
    vectorizer = CustomTfidfVectorizer().fit(train_docs)
    _engine = SearchEngine(vectorizer, train_docs)
    return vectorizer, train_docs, y_train, test_docs, y_test, target_names


def search(query: str, top_k: int = 5) -> List[Tuple[float, int, str]]:
    global _engine
    if _engine is None:
        if INDEX_PATH.exists():
            _engine = SearchEngine.load()
        else:
            init_system()
    return _engine.search(query, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="NumPy TF-IDF document search and classification")
    parser.add_argument("--query", type=str, default="space satellite launch orbit")
    parser.add_argument("--topk", type=int, default=5)
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument("--verify", action="store_true")
    operations.add_argument("--build-index", action="store_true")
    operations.add_argument("--train-classifier", action="store_true")
    operations.add_argument("--bonus", action="store_true")
    args = parser.parse_args()
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.verify:
        from verify_tfidf import main as verify
        verify()
        return
    if args.build_index or args.train_classifier or args.bonus:
        vec, train_docs, y_train, test_docs, y_test, target_names = init_system()
        print(f"Train={len(train_docs)} Test={len(test_docs)} Vocabulary={len(vec.vocabulary_)}", flush=True)
        if args.build_index:
            _engine.save()
            print(f"Index saved to {INDEX_PATH}")
        elif args.train_classifier:
            from src.classifier import train_and_evaluate_classifier
            _, acc, f1, cm = train_and_evaluate_classifier(
                _engine.doc_vectors, y_train, vec.transform(test_docs), y_test,
                target_names, test_docs=test_docs, vectorizer=vec,
            )
            print(f"Classification Accuracy: {acc:.4%} | Macro-F1: {f1:.6f}")
            print("Confusion matrix (actual rows, predicted columns):")
            print(cm)
            print("Saved artifacts/confusion_matrix.png and classifier_metrics.json")
        else:
            from src.bonus.runner import run_bonus
            run_bonus(vec, train_docs, y_train, test_docs, y_test, target_names,
                      X_train=_engine.doc_vectors)
        return
    results = search(args.query, args.topk)
    print(f"=== Search Results for '{args.query}' (Top {args.topk}) ===")
    for rank, (score, doc_id, snippet) in enumerate(results, start=1):
        print(f"[{rank}] Score: {score:.4f} | Doc #{doc_id}: {snippet}")
    if not results:
        print("No matching documents")


if __name__ == "__main__":
    main()

import os
import pickle
import tempfile
from numbers import Integral
from pathlib import Path

import numpy as np


INDEX_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "index.pkl"


def rank_scores(scores, top_k=5):
    if isinstance(top_k, bool) or not isinstance(top_k, Integral):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scores = np.asarray(scores)
    candidates = np.flatnonzero(scores > 0)
    if len(candidates) > top_k:
        boundary = np.partition(scores[candidates], -top_k)[-top_k]
        candidates = candidates[scores[candidates] >= boundary]
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order][:top_k]


class SearchEngine:
    def __init__(self, vectorizer, train_docs):
        self.vectorizer = vectorizer
        self.docs = list(train_docs)
        self.doc_vectors = vectorizer.transform(self.docs)

    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])[0]
        scores = self.doc_vectors @ query_vector
        indices = rank_scores(scores, top_k)
        return [(float(scores[i]), int(i), " ".join(self.docs[i].split())[:150])
                for i in indices]

    def save(self, path=INDEX_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = stream.name
                pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path=INDEX_PATH):
        with Path(path).open("rb") as stream:
            engine = pickle.load(stream)
        if not isinstance(engine, cls):
            raise ValueError("File does not contain a SearchEngine")
        return engine

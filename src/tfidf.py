import numpy as np

from src.preprocess import TextPreprocessor


class CustomTfidfVectorizer:
    """NumPy count TF, smooth IDF and zero-safe L2 normalization."""

    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor if preprocessor is not None else TextPreprocessor()
        self.vocabulary_ = {}
        self.idf_ = np.array([], dtype=np.float64)
        self.df_ = np.array([], dtype=np.float64)
        self.n_features_in_ = 0

    def fit(self, raw_documents):
        documents = list(raw_documents)
        tokens = [self.preprocessor.tokenize(doc) for doc in documents]
        terms = sorted({term for doc in tokens for term in doc})
        if not terms:
            raise ValueError("Cannot fit an empty vocabulary")
        self.vocabulary_ = {term: i for i, term in enumerate(terms)}
        self.n_features_in_ = len(terms)
        self.df_ = np.zeros(len(terms), dtype=np.float64)
        for doc in tokens:
            for term in set(doc):
                self.df_[self.vocabulary_[term]] += 1
        self.idf_ = np.log((1.0 + len(documents)) / (1.0 + self.df_)) + 1.0
        return self

    def count_matrix(self, raw_documents):
        if not self.vocabulary_:
            raise ValueError("Fit the vectorizer before transforming documents")
        documents = list(raw_documents)
        counts = np.zeros((len(documents), self.n_features_in_), dtype=np.float64)
        for row, doc in enumerate(documents):
            for token in self.preprocessor.tokenize(doc):
                column = self.vocabulary_.get(token)
                if column is not None:
                    counts[row, column] += 1.0
        return counts

    def transform(self, raw_documents):
        weighted = self.count_matrix(raw_documents)
        weighted *= self.idf_
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        np.divide(weighted, norms, out=weighted, where=norms != 0)
        return weighted

    def fit_transform(self, raw_documents):
        documents = list(raw_documents)
        return self.fit(documents).transform(documents)

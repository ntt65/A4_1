import re
from pathlib import Path


STOPWORDS_PATH = Path(__file__).resolve().parents[1] / "resources" / "stopwords_en.txt"


def load_stopwords(filepath=None):
    path = Path(filepath) if filepath is not None else STOPWORDS_PATH
    return set(path.read_text(encoding="utf-8").lower().split())


class TextPreprocessor:
    def __init__(self, stopwords=None):
        self.stopwords = load_stopwords() if stopwords is None else set(stopwords)
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def tokenize(self, text):
        return [token for token in self.token_pattern.findall(text.lower())
                if token not in self.stopwords]

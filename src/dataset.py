import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["comp.graphics", "rec.sport.baseball", "sci.space", "talk.politics.mideast"]


def clean_text(text):
    """Mask email and common telephone formats. Names require separate review."""
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[EMAIL]", text)
    return re.sub(
        r"(?<!\w)(?:\+?\d{1,3}[- .]?)?(?:\(?\d{2,4}\)?[- .]?)?\d{3}[- .]?\d{4}(?!\w)",
        "[PHONE]", text,
    )


def load_data(random_state=42):
    data = fetch_20newsgroups(
        subset="all", categories=CATEGORIES,
        remove=("headers", "footers", "quotes"), random_state=random_state,
        data_home=str(ROOT / "artifacts" / "data_cache"),
    )
    cleaned = [clean_text(doc) for doc in data.data]
    retained = [i for i, doc in enumerate(cleaned) if doc.strip()]
    docs = [cleaned[i] for i in retained]
    labels = np.asarray(data.target)[retained]
    train_docs, test_docs, y_train, y_test = train_test_split(
        docs, labels, test_size=0.2, random_state=random_state, stratify=labels,
    )
    metadata = {
        "raw_documents": len(cleaned), "documents": len(docs),
        "removed_empty": len(cleaned) - len(docs),
        "train_documents": len(train_docs), "test_documents": len(test_docs),
        "target_names": list(data.target_names), "random_state": random_state,
        "train_counts": np.bincount(y_train).tolist(),
        "test_counts": np.bincount(y_test).tolist(),
        "corpus_sha256": hashlib.sha256("\0".join(docs).encode("utf-8")).hexdigest(),
        "privacy_review": "Email and common telephone masking only. Manual name review pending.",
    }
    output = ROOT / "artifacts" / "dataset_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return (train_docs, y_train), (test_docs, y_test), data.target_names

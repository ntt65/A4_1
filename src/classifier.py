import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]


def train_and_evaluate_classifier(
    X_train, y_train, X_test, y_test, target_names,
    output_img=ROOT / "artifacts" / "confusion_matrix.png", test_docs=None, vectorizer=None,
):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    with threadpool_limits(limits=2):
        clf.fit(csr_matrix(X_train), y_train)
        y_pred = clf.predict(csr_matrix(X_test))
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    labels = np.arange(len(target_names))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    output_img = Path(output_img)
    output_img.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        title=f"Confusion Matrix (Accuracy {acc:.2%})",
        xlabel="Predicted", ylabel="Actual",
        xticks=labels, yticks=labels,
        xticklabels=target_names, yticklabels=target_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, str(cm[row, column]), ha="center", va="center",
                    color="white" if cm[row, column] > cm.max() / 2 else "black")
    fig.savefig(output_img, dpi=160)
    plt.close(fig)
    report = classification_report(
        y_test, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    errors = []
    terms = None
    coefficients = clf.coef_
    intercepts = clf.intercept_
    if vectorizer is not None:
        terms = np.array(sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get))
        if len(clf.classes_) == 2:
            coefficients = np.vstack([-clf.coef_[0] / 2, clf.coef_[0] / 2])
            intercepts = np.array([-clf.intercept_[0] / 2, clf.intercept_[0] / 2])
    for index in np.flatnonzero(y_pred != y_test)[:5]:
        example = {
            "test_doc_id": int(index),
            "actual": target_names[int(y_test[index])],
            "predicted": target_names[int(y_pred[index])],
            "text": test_docs[index] if test_docs is not None else "",
        }
        if terms is not None:
            contribution = X_test[index] * (coefficients[y_pred[index]] - coefficients[y_test[index]])
            example["intercept_difference"] = float(intercepts[y_pred[index]] - intercepts[y_test[index]])
            example["decision_margin"] = float(contribution.sum() + example["intercept_difference"])
            positive = np.flatnonzero(contribution > 0)
            negative = np.flatnonzero(contribution < 0)
            example["top_predicted_terms"] = [
                {"term": str(terms[i]), "contribution": float(contribution[i])}
                for i in positive[np.argsort(-contribution[positive])][:6]
            ]
            example["top_actual_terms"] = [
                {"term": str(terms[i]), "contribution": float(contribution[i])}
                for i in negative[np.argsort(contribution[negative])][:6]
            ]
        errors.append(example)
    metrics = {
        "accuracy": float(acc), "macro_f1": float(macro_f1),
        "correct": int(np.count_nonzero(y_pred == y_test)), "test_documents": len(y_test),
        "confusion_matrix": cm.tolist(), "target_names": list(target_names),
        "classification_report": report, "misclassified_examples": errors,
        "iterations": clf.n_iter_.tolist(),
    }
    output_img.with_name("classifier_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return clf, acc, macro_f1, cm

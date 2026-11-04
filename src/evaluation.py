import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def per_category_metrics(y_true, y_pred, labels, level="chunk"):
    """Return precision, recall, F1, and support for every category."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    return pd.DataFrame({
        "level": level,
        "category": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })


def probability_frame(results, contracts, labels):
    """Align model probability dictionaries with their chunk contracts."""
    if len(results) != len(contracts):
        raise ValueError("Each prediction result must have one corresponding chunk")

    probabilities = pd.DataFrame(index=range(len(results)))
    probabilities["contract"] = list(contracts)
    for label in labels:
        probabilities[label] = [row.get(label, 0.0) for row in results]
    return probabilities


def aggregate_to_contracts(chunk_probabilities, labels, threshold=0.5):
    """Take the maximum chunk probability for each category in each contract."""
    contract_scores = (
        chunk_probabilities.groupby("contract", sort=True)[labels]
        .max()
        .sort_index()
    )
    contract_predictions = (contract_scores >= threshold).astype(int)
    return contract_scores, contract_predictions


def contract_truth_from_chunks(chunks, labels):
    """Build multi-hot contract labels from all categories found in chunks."""
    labels = list(labels)
    contracts = pd.Series(chunks["contract"].unique()).sort_values().tolist()
    truth = pd.DataFrame(0, index=contracts, columns=labels, dtype=int)

    for row in chunks.itertuples():
        for category in row.categories or []:
            normalized = category.casefold() if isinstance(category, str) else category
            if normalized in labels:
                truth.at[row.contract, normalized] = 1
    return truth


def evaluate_predictions(y_true, y_pred, labels, level="chunk"):
    """Return per-category metrics plus overall micro and macro scores."""
    metrics = per_category_metrics(y_true, y_pred, labels, level=level)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _, _, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    overall = pd.DataFrame([{
        "level": level,
        "category": "micro",
        "precision": precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )[0],
        "recall": precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )[1],
        "f1": micro_f1,
        "support": int(y_true.sum()),
    }, {
        "level": level,
        "category": "macro",
        "precision": precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )[0],
        "recall": precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )[1],
        "f1": macro_f1,
        "support": int(y_true.sum()),
    }])

    return metrics, pd.concat([overall, metrics], ignore_index=True)

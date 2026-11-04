
import numpy as np


def calculate_pos_weights(y_train, max_weight=100.0):
    """
    Calculate per-category positive weights
    for multi-label classification.
    """
    y = np.asarray(y_train)

    if y.ndim != 2 or y.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D label matrix")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")

    positive_counts = y.sum(axis=0)
    negative_counts = len(y) - positive_counts

    weights = np.divide(
        negative_counts,
        positive_counts,
        out=np.ones_like(negative_counts, dtype=float),
        where=positive_counts > 0,
    )

    return np.clip(weights, 1.0, max_weight)
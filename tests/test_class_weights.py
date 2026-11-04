
import numpy as np
from src.class_weights import calculate_pos_weights


def test_class_weights():
    labels = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0],
    ])

    weights = calculate_pos_weights(labels)

    assert np.allclose(weights, [3.0, 3.0])


def test_zero_positive_label():
    labels = np.array([
        [1, 0],
        [0, 0],
        [0, 0],
    ])

    weights = calculate_pos_weights(labels)

    assert np.allclose(weights, [2.0, 1.0])
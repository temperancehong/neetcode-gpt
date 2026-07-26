import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        n = len(y_pred)
        loss_v = y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)
        loss = np.sum(loss_v)
        loss = -loss / n
        loss = round(loss,4)

        return loss

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        N, C = y_true.shape
        v = y_true * np.log(y_pred) # (shape: n_samples x n_classes)
        loss = - 1/N * np.sum(v)
        loss = round(loss, 4)

        return loss

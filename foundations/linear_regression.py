import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        # print(X.shape, weights.shape)
        pred = X.dot(weights)
        # print(pred)
        return np.round(pred, 5)
        

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        difference_v = model_prediction - ground_truth
        mse = 1/len(difference_v) * (difference_v.T).dot(difference_v)
        # print(mse)
        return round(mse.item(),5)

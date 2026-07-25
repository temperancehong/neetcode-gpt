import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_max = np.max(z)
        v_nominator = np.exp(z - z_max) # this is a 1D np array 
        denominator = 0
        length = len(z)
        for j in range(length):
            denominator += np.exp(z[j]-z_max)
        
        sftmx = v_nominator/denominator
        return np.round(sftmx, 4)

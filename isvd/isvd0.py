import numpy as np


class ISVD0():
    """Interval Singular Value Decomposition with Naive Assumption
    Taking average from the interval input matrix, then perform ordinary SVD.

    """
    def __init__(self, tolerance = 0.01, full_matrix=False, condition_threshold=25) -> None:
        """Initialize class

        Args:
            tolerance (float, optional): Tolerance limit of the basis truncation. Defaults to 0.01.
            full_matrix (bool, optional): Set the SVD to return full matrix or not. Defaults to False.
        """
        # Assert conditions
        assert tolerance >= 0 , f"Expected tolerance to be larger than equal to 0, got: {tolerance}"
        if not full_matrix:
            assert tolerance > 0 and full_matrix == False, f"If tolerance is greater than zero, full_matrix should be false"
        else:
            assert tolerance == 0 and full_matrix == True, f"If full_matrix is True, then tolerance should be zero"

        self.M_int = None
        self.M_lower = None
        self.M_upper = None
        self.M_avg = None
        self.n_point = 0 
        self.n_dim = 0
        self.U, self.S, self.V = None,None,None
        self.tolerance = tolerance
        self.full_matrix = full_matrix
        self.truncated_idx = None
        self.S_trunc = None
        self.U_trunc = None
        self.V_trunc = None


    def fit(self, M_int: (np.ndarray)) -> None:
        """Fit Interval matrix to the Naive Interval SVD approach

        Args:
            M_int (np.ndarray): A 2D interval matrix, with tuples of lower and upper bound in each elements
        """
        # Assert snapshot dimension, must be equal to 3
        # 2D interval matrix with tuples in each elements would be a 3D array
        assert M_int.ndim == 3, f"Array dimension expected to be 3, got: {M_int.ndim}"

        self.M_int = M_int
        
        # Split lower and upper matrix
        self.M_lower = self.M_int[:,:,0]
        self.M_upper = self.M_int[:,:,1]

        # Take the average of both matrix
        self.M_avg = (self.M_lower + self.M_upper)/2

        # Extract number of point 
        self.n_point, self.n_dim = self.M_lower.shape

        # Compute the Naive SVD from the average matrix:
        self.U, self.S, self.V = np.linalg.svd(self.M_avg, full_matrices=self.full_matrix)

        if not self.full_matrix:
            # Truncate matrix
            temp = 0
            diagsum = np.sum(self.S)
            for idx, sigma in enumerate(self.S):
                temp += sigma
                ratio = temp/diagsum

                if ratio >= (1-self.tolerance):
                    self.truncated_idx = idx
                    break

            self.S_trunc = self.S[:self.truncated_idx]
            self.U_trunc = self.U[:,:self.truncated_idx]
            self.V_trunc = self.V[:self.truncated_idx,:]

    
    def fit_return(self, M_int: (np.ndarray)):
        """Fit Interval matrix to the Naive Interval SVD approach.
        Then return the svd matrices

        Args:
            M_int (np.ndarray): A 2D interval matrix, with tuples of lower and upper bound in each elements

        Returns:
            tuple[np.ndarray,np.ndarray,np.ndarray]: _description_
        """
        self.fit(M_int)

        if not self.full_matrix:
            return self.U_trunc, self.S_trunc, self.V_trunc
        else:
            return self.U, self.S, self.V


if __name__ == "__main__":
    # Test code on simple small matrix 4x3 with intervals.
    mat = np.array([[(2,5),(4,6),(1,3)],
                    [(4,5),(2,7),(3,4)],
                    [(1,4),(8,9),(6,7)],
                    [(7,9),(8,9),(3,6)]])
    
    isvd_0 = ISVD0(tolerance=0.01,full_matrix=False)
    U,S,V = isvd_0.fit_return(mat)

    print(f"U matrix is: {U}")
    print(f"S matrix is: {S}")
    print(f"V matrix is: {V}")
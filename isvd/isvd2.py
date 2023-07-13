import numpy as np
from isvd import tools


class ISVD2():
    """
    Class for ISVD 2

    Main reference:
    Li, M.-L., Mauro, F. D., Candan, K. S., & Sapino, M. L. (2021). Matrix Factorization with Interval-Valued Data. 
    In IEEE Transactions on Knowledge and Data Engineering (Vol. 33, Issue 4, pp. 1644–1658). 
    Institute of Electrical and Electronics Engineers (IEEE). https://doi.org/10.1109/tkde.2019.2942310

    """

    def __init__(self, target_rank: int, full_matrix=False, condition_threshold=25) -> None:
        """Initialize class

        Args:
            target_rank (int): target_rank of the decomposed matrices. 0 for full_matrix decomposition. Target rank is used instead of
            tolerance as in naive_isvd because unsure of the mathematical correctness of using tolerance in interval observation.
            full_matrix (bool, optional): Set the SVD to return full matrix or not. Defaults to False.
            condition_threshold (float, optional): Condition number threshold for Moore-Penrose PseudoInverse
        """
        # Assert conditions
        assert target_rank >= 0 , f"Expected target_rank to be larger than equal to 0, got: {target_rank}"
        if not full_matrix:
            assert target_rank > 0 and full_matrix == False, f"If target_rank is greater than zero, full_matrix should be false"
        else:
            assert target_rank == 0 and full_matrix == True, f"If full_matrix is True, then target_rank should be zero"

        self.M_int = None
        self.A_lower = None
        self.A_upper = None
        self.n_point = 0 
        self.n_dim = 0
        self.target_rank = target_rank
        self.full_matrix = full_matrix
        self.truncated_idx = None
        self.S_int = None
        self.U_int = None
        self.V_int = None
        self.U_avg, self.S_avg, self.V_avg = None,None,None
        self.condition_threshold = condition_threshold
    
    def fit(self, M_int: (np.ndarray)) -> None:
        """Fit Interval matrix to the Naive Interval SVD approach

        Args:
            M_int (np.ndarray): A 2D interval matrix, with tuples of lower and upper bound in each elements
        """
        # Assert snapshot dimension, must be equal to 3
        # 2D interval matrix with tuples in each elements would be a 3D array
        assert M_int.ndim == 3, f"Array dimension expected to be 3, got: {M_int.ndim}"

        self.M_int = M_int
        M_lo = self.M_int[:,:,0]
        M_up = self.M_int[:,:,1]
        
        # Matrix multiplication
        # transpose([1,0,2]) means 3d array transpose that behave like 2d array
        A_int = tools.interval_matmul(self.M_int, self.M_int.transpose([1,0,2]))

        # Split lower and upper matrix A
        self.A_lower = A_int[:,:,0]
        self.A_upper = A_int[:,:,1]

        # Eigen decomposition lower and upper matrix A
        S_lo, U_lo = np.linalg.eig(self.A_lower)
        idx = S_lo.argsort()[::-1]   
        S_lo = S_lo[idx]
        U_lo = U_lo[:,idx]

        S_up, U_up = np.linalg.eig(self.A_upper)
        idx = S_up.argsort()[::-1]   
        S_up = S_up[idx]
        U_up = U_up[:,idx]

        # Truncate to desired rank
        if not self.full_matrix:
            U_lo = U_lo[:,:self.target_rank]
            S_lo = S_lo[:self.target_rank]
            U_up = U_up[:,:self.target_rank]
            S_up = S_up[:self.target_rank]

        ## Obtain V_int by inverting U_avg
        S_lo_mat = np.sqrt(np.diag(S_lo))
        S_up_mat = np.sqrt(np.diag(S_up))

        # invert matrices
        U_lo_inv = self._invmat(self.condition_threshold, U_lo)
        U_up_inv = self._invmat(self.condition_threshold, U_up)
        S_lo_inv = self._invmat(self.condition_threshold, S_lo_mat)
        S_up_inv = self._invmat(self.condition_threshold, S_up_mat)
        
        # Compute V
        V_lo = (S_lo_inv @ U_lo_inv @ M_lo).T
        V_up = (S_up_inv @ U_up_inv @ M_up).T
        self.V_int = np.dstack([V_lo, V_up])

        ## Align eigenvectors
        mapping = tools.generateMapping(U_lo, U_up, S_lo, S_up)

        # Recover modified eigenvectors
        U_up[:, mapping[:,4]==1] = U_up[:, mapping[:,4]==1] * -1
        V_up[:, mapping[:,4]==1] = V_up[:, mapping[:,4]==1] * -1

        # Compute new eigenvalues and eigenvectors based on matching result
        S_lo = np.sqrt(np.diag(np.minimum(mapping[:,2], mapping[:,3])))
        S_up = np.sqrt(np.diag(np.maximum(mapping[:,2], mapping[:,3])))
        self.S_avg = (S_lo + S_up)/2
        self.S_int = np.dstack((S_lo,S_up))

        U_lo = U_lo[:,mapping[:,0].astype(int)]
        U_up = U_up[:,mapping[:,1].astype(int)]

        V_lo = V_lo[:,mapping[:,0].astype(int)]
        V_up = V_up[:,mapping[:,1].astype(int)]

        self.U_int = np.dstack((U_lo, U_up))
        self.U_avg = (U_lo + U_up)/2

        # wrap up everything
        self.V_int = np.dstack([V_lo, V_up])
        self.V_avg = (V_lo + V_up)/2

        self.U_int = np.dstack([U_lo, U_up])
        self.U_avg = (U_lo + U_up)/2


    def fit_return(self, M_int: (np.ndarray), decomp_strategy = "b")->list:
        """Fit Interval matrix to the Naive Interval SVD approach and return based in the decomposition strategy

        Args:
            M_int (np.ndarray): A 2D interval matrix, with tuples of lower and upper bound in each elements
            decomp_strategy (str): Decomposition strategy, "a", "b", or "c", more details on the main reference "Matrix Factorization with Interval-Valued Data".
                - "a": all matrices U, S, V are intervals
                - "b": matrices U and V are average, matrix S is interval
                - "c": all matrices are average
        
        Return:
            list: list comprised of [U, S, V], with each matrix can be the average or interval depends on the decomposition strategy
        """
        assert decomp_strategy.lower() in ["a","b","c"], f"Decomposition strategy choice is not an option"

        self.fit(M_int= M_int)

        if decomp_strategy.lower() == "a":
            return [self.U_int, self.S_int, self.V_int]
        if decomp_strategy.lower() == "b":
            return [self.U_avg, self.S_int, self.V_avg]
        if decomp_strategy.lower() == "c":
            return [self.U_avg, self.S_avg, self.V_avg]
    
    def _invmat(self, threshold, mat):
        cond_num = np.linalg.cond(mat)

        if cond_num > threshold or mat.shape[0] != mat.shape[1]:
            inv = np.linalg.pinv(mat)
        else:
            inv = np.linalg.inv(mat)
        
        return inv

if __name__ == "__main__":
    pass
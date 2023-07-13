"""
Module that contains functions for supporting the ISVD algorithms

Main reference:
Li, M.-L., Mauro, F. D., Candan, K. S., & Sapino, M. L. (2021). Matrix Factorization with Interval-Valued Data. 
In IEEE Transactions on Knowledge and Data Engineering (Vol. 33, Issue 4, pp. 1644–1658). 
Institute of Electrical and Electronics Engineers (IEEE). https://doi.org/10.1109/tkde.2019.2942310

"""
import numpy as np
import scipy


def interval_matmul(a_int: (np.ndarray), b_int: (np.ndarray)) -> np.ndarray:
    """Matrix multiplication for inetrval matrices

    Args:
        a_int (np.ndarray): Interval matrix a, a 2D interval matrix, with tuples of lower and upper bound in each elements.
                            thus, it would be a 3D numpy array. shape: (na x ma x 2)
        b_int (np.ndarray): Interval matrix b, a 2D interval matrix, with tuples of lower and upper bound in each elements.
                            thus, it would be a 3D numpy array. shape: (nb x mb x 2)

    Returns:
        res_int (np.ndarray): result interval matrix, a 2D interval matrix, with tuples of lower and upper bound in each elements.
                            thus, it would be a 3D numpy array. shape: (mb x na x 2)
    """

    # Assert snapshot dimension, must be equal to 3
    # 2D interval matrix with tuples in each elements would be a 3D array
    assert a_int.ndim == 3, f"Array dimension expected to be 3, got: {a_int.ndim}"
    assert b_int.ndim == 3, f"Array dimension expected to be 3, got: {b_int.ndim}"
    assert a_int.shape[1] == b_int.shape[0], f"Array dimension do not match, expected: 2nd dim of a == 1st dim of b"

    # Split lower and upper matrix
    a_lower, a_upper = a_int[:,:,0], a_int[:,:,1]
    b_lower, b_upper = b_int[:,:,0], b_int[:,:,1]

    # Temporary matrices
    t1 = a_lower @ b_lower
    t2 = a_lower @ b_upper
    t3 = a_upper @ b_lower
    t4 = a_upper @ b_upper

    # Initialize result_matrix
    res_lower = np.zeros(shape=t1.shape)
    res_upper = np.zeros(shape=t1.shape)

    # Fill in the matrix
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            res_lower[i,j] = min([t1[i,j],t2[i,j],t3[i,j],t4[i,j]])
            res_upper[i,j] = max([t1[i,j],t2[i,j],t3[i,j],t4[i,j]])
    
    # Stack matrix result
    res_int = np.dstack((res_lower,res_upper))

    return res_int


def matrix_avg_replacement(m_int: (np.ndarray)) -> list([np.ndarray, np.ndarray]):
    """Average replacement for interval matrix witl ill-condition.
    Average replacement is a mechanism to correct matrix that include entries
    that are not proper intervals (minimum values larger than maximum values).

    Note: Some algorithm have potential to introduce such improper interval.
    However, it does not lead to mathematical problem. But before returned to the user,
    the ourput should be corrected

    For vector: your interval vector should be reshaped into 1 x n x 2

    Args:
        m_int (np.ndarray): A 2D interval matrix, with tuples of lower and upper bound in each elements.
                            thus, it would be a 3D numpy array. shape: (n x m x 2)

    Returns:
        m_replaced (np.ndarray): A 2D interval matrix with corrected interval elements(if any). 
                                shape would be the same as the input.
        m_flag (np.ndarray): A 2D matrix of n x m which consists of 0 and 1 as a flag. 
                            1 means the value has been corrected.
    """
    # assert condition
    assert m_int.ndim == 3, f"""Array dimension expected to be 3, got: {a_int.ndim}. 
                            If you have a vector instead matrix, reshape vector into 1 x n x 2"""

    # copy original matrix to avoid aliasing
    m = m_int.copy()
    flag = np.zeros(shape=m.shape[:2])

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i,j,0] > m[i,j,1]:
                val_avg = (m[i,j,0] + m[i,j,1])/2
                m[i,j,0] = val_avg
                m[i,j,1] = val_avg
                flag[i,j] = 1
    
    return [m,flag]


def inverse_interval_core(s_int:(np.ndarray)) -> np.ndarray:
    """Inverse approximation of interval-valued core matrix \Sigma.
    The matrix should be square, and diagonal with shape: r x r x 2
    Refers to section 4.4.2.1 in the main reference:
    Li, M.-L., Mauro, F. D., Candan, K. S., & Sapino, M. L. (2021). Matrix Factorization with Interval-Valued Data.

    Args:
        s_int (np.ndarray): A square and diagonal 2D interval matrix. Shape: r x r x 2

    Returns:
        s_inv (np.ndarray): An inverse approximation of the input interval matrix s_int. Shape: r x r
    """
    
    # Assert matrix condition:
    assert s_int.ndim == 3, f"Array dimension expected to be 3, got: {s_int.ndim}"
    assert s_int.shape[0] == s_int.shape[1], "Input matrix should be square"
    assert (_check_diag_mat(s_int[:,:,0]) and _check_diag_mat(s_int[:,:,1])), "Input matrix should be a diagonal matrix"

    # Initialize inverse matrix result
    s_inv = np.zeros(shape=s_int.shape[:2])

    # Split matrix
    s_lower = s_int[:,:,0]
    s_upper = s_int[:,:,1]

    # Main procedure
    for i in range(s_int.shape[0]):
        if s_lower[i,i] == 0 and s_upper[i,i] == 0:
            s_inv[i,i] = 0
        elif s_lower[i,i] == 0 and s_upper[i,i] != 0:
            s_inv[i,i] = 2 / (s_upper[i,i])
        elif s_lower[i,i] != 0 and s_upper[i,i] == 0:
            s_inv[i,i] = 2 / (s_lower[i,i])
        else:
            s_inv[i,i] = 2 / (s_lower[i,i] + s_upper[i,i])
    
    return s_inv


def l2_matrix_norm(a:(np.ndarray)):
    """L2-Normalization for matrix

    NOTE: Can be replaced by scikitlearn.preprocessing.normalize(a, norm='l2', axis=0, return_norm=True)

    Args:
        a (np.ndarray): A 2D scalar matrix with size n x m
    
    Returns:
        a_hat: A 2D normalized scalar matrix with size n x m
        col_norm: Columns norms of a, size 1 x m
    """
    # Assert matrix condition:
    assert a.ndim == 2, f"Array dimension expected to be 3, got: {a.ndim}"

    # copy to avoid aliasing
    a_hat = a.copy()
    
    norm = []
    for j in range(a.shape[1]):
        norm.append(np.linalg.norm(a_hat[:,j]))
        a_hat[:,j] = a_hat[:,j]/norm[j]
    
    return a_hat, np.array(norm)


def check_ill_matrix(a: (np.ndarray)):
    """Checking ill interval matrix

    Check interval matrices that are not proper intervals (minimum values larger than maximum values).

    Args:
        a (np.ndarray): A 2D interval matrix, with tuples of lower and upper bound in each elements.
                        thus, it would be a 3D numpy array. shape: (n x m x 2)

    Returns:
        ill (bool): Boolean that indicates if an interval matrix is ill (True) or good (False)
        flag (np.ndarray): A 2D matrix of n x m which consists of 0 and 1 as a flag. 
                            1 means the value is ill.
    """
    # assert condition
    assert a.ndim == 3, f"""Array dimension expected to be 3, got: {a_int.ndim}. 
                            If you have a vector instead matrix, reshape vector into 1 x n x 2"""

    # copy original matrix to avoid aliasing
    flag = np.zeros(shape=a.shape[:2])

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i,j,0] > a[i,j,1]:
                flag[i,j] = 1
    
    if flag.sum() > 0:
        ill = True
    else:
        ill = False

    return [ill,flag]

def generateMapping(umin: np.ndarray, umax: np.ndarray, smin: np.ndarray, smax: np.ndarray) -> np.ndarray:
    """Generate vector mapping for alignment

    Args:
        vmin (np.ndarray): Lower bound matrix eigenvector
        vmax (np.ndarray): Upper bound matrix eigenvector
        smin (np.ndarray): Lower bound matrix eigenvalue
        smax (np.ndarray): Upper bound matrix eigenvalue

    Returns:
        np.ndarray: mapping result
    """

    # Compute cosine distance matrix
    distMat = scipy.spatial.distance.cdist(umin.T, umax.T, "cosine")

    # Identify negative vectors
    neg = np.where(distMat > 1)
    distMat[neg] = 2 - distMat[neg]

    # Eigenvector matching
    mapping = _eigvecMatching(distMat)

    # Construct mapping matrix
    map_mat = np.concatenate([mapping.reshape(-1,1), np.array(range(umin.shape[1])).reshape(-1,1)], axis = 1)
    cosine_dist = np.array([distMat[a,b] for a,b in zip(map_mat[:,0], map_mat[:,1])]).reshape(-1,1)
    degrees = (np.arccos(1 - cosine_dist) / (2*np.pi)) * 360
    map_mat = np.concatenate([map_mat, cosine_dist, degrees], axis=1)
    map_mat = map_mat[map_mat[:, 3].argsort()]


    # Degree cutting
    degreeth = 90
    map_mat = map_mat[map_mat[:,3]<= degreeth, :2]

    # Update mapping
    bool_neg = np.zeros(distMat.shape)
    bool_neg[neg] = 1
    map5 = np.array([bool_neg[int(a),int(b)] for a,b in zip(map_mat[:,0], map_mat[:,1])]).reshape(-1,1)
    map_mat = np.concatenate([map_mat, smin[map_mat[:,0].astype(int)].reshape(-1,1), 
                              smax[map_mat[:,1].astype(int)].reshape(-1,1), map5], axis=1)
    map_mat = map_mat[map_mat[:, 1].argsort()]
    
    return map_mat


def _eigvecMatching(distMat: np.ndarray):
    """Eigenvector Matching

    Args:
        distMat (np.ndarray): _description_
    """
    n = distMat.shape[0]

    # Extract minimum value of each column and its index
    minval = np.min(distMat, axis=0)
    minidx = np.argmin(distMat, axis=0)

    # Find indices that min. eigenvectors doesn't match any max eigenvectors
    nosel = np.setdiff1d(np.array(range(n)), minidx)
    
    # Find duplicated index of min. eigenvectors
    nn, bin = _histc(minidx, np.unique(minidx))

    # Find indices of bin for duplicated min. eigenvector
    multiple = np.argwhere(nn > 1).flatten()

    # In case of duplicated index
    if len(multiple) > 0:
        # wrap as a function to make debugging easier
        minidx = _multduplicated(distMat, minval, minidx, nosel, nn, bin, multiple)

    return minidx

def _multduplicated(distMat, minval, minidx, nosel, nn, bin, multiple):
    adj_bin = bin-1  # Adjusted binning for python index, in the originial code in MATLAB, this variable is used for indexing 
    dupidx1 = np.argwhere(np.isin(adj_bin, multiple))
    dupidx2 = minidx[dupidx1.flatten()].reshape(-1,1)
    dupidx3 = minval[dupidx1.flatten()].reshape(-1,1)
    dupidx = np.concatenate([dupidx1, dupidx2, dupidx3], axis=1)

    duptargetids = np.unique(dupidx2)

    for i in range(len(duptargetids)):
        dupids =  dupidx[np.argwhere(dupidx2.flatten()==duptargetids[i]), [0,2]]
        
        #sort by cosine distance
        dupids = dupids[dupids[:, 1].argsort()]

        for j in range(1, dupids.shape[0]):
            vector = distMat[:, int(dupids[j,0])].reshape(-1,1)
            vector2 = np.array(range(vector.shape[0])).reshape(-1,1)
            vectors = np.concatenate([vector, vector2], axis=1)

            #sort by cosine distance
            vectors = vectors[vectors[:, 0].argsort()]

            # Find unmatched min. eigenvector to rematch
            p = 0
            k = 1

            while p == 0:
                if np.any(vectors[k,1]==nosel):
                    minidx[int(dupids[j,0])] = vectors[k,1]
                    nosel = np.delete(nosel, np.where(nosel == vectors[k,1]))
                    p = vectors[k,1]
                else:
                    k += 1
    return minidx

def _histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return [r, map_to_bins]

def _check_diag_mat(a: (np.ndarray)) -> bool:
    """Check if a matrix is a diagonal matrix

    Args:
        a (np.ndarray): a 2D matrix

    Returns:
        bool: True or False
    """
    diag_elem = np.diag(a).copy()
    np.fill_diagonal(a,0)
    out = (a==0).all()
    np.fill_diagonal(a,diag_elem)
    return out


if __name__ == "__main__":
    # Test code on simple small matrix 4x3 with intervals.
    a_int = np.array([[(2,5),(4,6),(1,3)],
                    [(4,5),(2,7),(3,4)],
                    [(1,4),(8,9),(6,7)],
                    [(7,9),(8,9),(3,6)]])
    
    b_int = np.array([[(1,5),(4,7)],
                      [(2,4),(3,4)],
                      [(6,8),(4,5)]])
    
    b_ill = np.array([[(1,5),(4,7)],
                      [(6,4),(3,4)],
                      [(6,8),(4,5)]])
    
    s_test = np.array([[(1,5),(0,0),(0,0)],
                      [(0,0),(3,4),(0,0)],
                      [(0,0),(0,0),(2,6)]])
    
    # x = interval_matmul(a_int,b_int)  # test interval matmul
    # c,flag = matrix_avg_replacement(b_ill)  # test matrix avg replacement
    # s_inv = inverse_interval_core(s_test)
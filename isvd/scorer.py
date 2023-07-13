import numpy as np
import scipy

def reconstruction_acc(rec_int: np.ndarray, ori_int: np.ndarray):
    
    # Split lower and upper matrix
    rec_lower, rec_upper = rec_int[:,:,0], rec_int[:,:,1]
    ori_lower, ori_upper = ori_int[:,:,0], ori_int[:,:,1]

    # Calculate norm
    fnorm_lo = _calc_diff(rec_lower, ori_lower)
    fnorm_up = _calc_diff(rec_upper, ori_upper)

    # Reconstruction accuracy
    accuracy = _calc_fscore2(fnorm_lo, fnorm_up)

    return accuracy

def _calc_diff(rec_mat: np.ndarray, ori_mat: np.ndarray):
    nominator = np.linalg.norm((rec_mat - ori_mat), ord="fro")
    denom = np.linalg.norm(ori_mat, "fro")
    diff_res = nominator / denom

    return diff_res

def _calc_fscore2(fnorm_min: np.ndarray, fnorm_max: np.ndarray):
    
    acc_min = np.maximum(1-fnorm_min, 0)
    acc_max = np.maximum(1-fnorm_max, 0)

    if acc_min == 0 and acc_max == 0:
        result = 0
    else:
        result = (2 * acc_min * acc_max) / (acc_min + acc_max)
    
    return result
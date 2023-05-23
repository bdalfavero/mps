#!/usr/bin/env python3

import numpy as np
import scipy.linalg as la

def matrix_svd(m):
    """
    matrix_svd(m)
    SVD of matrix, but the sigma matrix is a matrix instead of a vector.

    Arguments:
    m: np.array, matrix to take the SVD of.

    Returns:
    u: np.array, matrix of left singular vectors.
    s: np.array, matrix of singular values, quasidiagonal.
    v_dag: np.arrray, matrix of right singular vectors.
    """

    u, s, v_dag = la.svd(m)
    s_mat = np.zeros((u.shape[1], v_dag.shape[0]))
    for i in range(s.size):
        s_mat[i, i] = s[i]
    
    return (u, s_mat, v_dag)
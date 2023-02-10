"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError(
    #     "`normalize_points` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )
    (n, _) = points.shape
    ci = np.mean(a=points, axis=0)
    points_sub_means = points - ci
    si = 1/np.std(a=points_sub_means, axis=0)
    S = np.eye(3)
    S[0][0] = si[0]
    S[1][1] = si[1]

    C = np.eye(3)
    C[0][-1] = -ci[0]
    C[1][-1] = -ci[1]
    
    T = np.matmul(S, C)

    points_append_1 = np.append(arr=points, values = np.ones((n, 1)), axis = 1)
    points_normalized = np.transpose(np.matmul(T, np.transpose(points_append_1)))
    points_normalized = points_normalized[:,:-1]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError(
    #     "`unnormalize_F` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )
    F_orig = np.matmul(np.matmul(np.transpose(T_b),F_norm),T_a)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # normalize points
    (n_points_a, T_a) = normalize_points(points_a)
    (n_points_b, T_b) = normalize_points(points_b)
    
    A = []
    (n, _) = points_a.shape

    for i in range(n):
      uai = n_points_a[i, 0]
      ubi = n_points_b[i, 0]
      vai = n_points_a[i, 1]
      vbi = n_points_b[i, 1]
      A.append([uai*ubi,vai*ubi,ubi,uai*vbi,vai*vbi,vbi,uai,vai,1])
      
    A = np.array(A)
    f = np.linalg.svd(a = A)[2][-1]

    n_F = f.reshape([3,3])

    Uf, Sf, Vhf = np.linalg.svd(n_F, full_matrices=False)
    Sf[-1] = 0
    n_F = np.dot(Uf*Sf, Vhf)

    F = unnormalize_F(n_F, T_a, T_b)


    # raise NotImplementedError(
    #     "`estimate_fundamental_matrix` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F

import numpy as np


def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray
) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_3d: A numpy array of shape (N, 3)

    Returns:
        M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError(
    #     "`calculate_projection_matrix` function in "
    #     + "`projection_matrix.py` needs to be implemented"
    # )
    A = []
    (n, _) = points_2d.shape
    append = np.append
    for i in range(n):
      m1 = append(points_3d[i], 1)
      m1 = append(m1, np.zeros(4))
      m1 = append(m1, -points_2d[i][0]*points_3d[i])
      m1 = append(m1, -points_2d[i][0])
      A.append(m1)
      m2 = append(np.zeros(4), points_3d[i])
      m2 = append(m2, 1)
      m2 = append(m2, -points_2d[i][1]*points_3d[i])
      m2 = append(m2, -points_2d[i][1])
      A.append(m2)
    A = np.array(A)
    M = np.linalg.svd(a = A)[2][-1]
    M = M.reshape(3,4)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z] in non-homogenous coordinates to
    (x,y) in non-homogenous image coordinates.
    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError(
    #     "`projection` function in " + "`projection_matrix.py` needs to be implemented"
    # )
    (n, _) = points_3d.shape
    points_3d_extended = np.append(points_3d, np.ones((n,1)),axis = 1)
    projected_points_2d = np.matmul(P, np.transpose(points_3d_extended))
    projected_points_2d = projected_points_2d/projected_points_2d[-1]
    projected_points_2d = projected_points_2d[:-1]
    projected_points_2d = np.transpose(projected_points_2d)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError(
    #     "`calculate_camera_center` function in "
    #     + "`projection_matrix.py` needs to be implemented"
    # )
    Q = M[:,:3]
    cc = -np.matmul(np.linalg.inv(Q),M[:,-1])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc

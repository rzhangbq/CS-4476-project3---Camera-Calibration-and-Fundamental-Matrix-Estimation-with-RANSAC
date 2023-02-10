import math

import numpy as np
from vision.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError(
    #     "`calculate_num_ransac_iterations` function "
    #     + "in `ransac.py` needs to be implemented"
    # )
    log = math.log
    down = log(1-ind_prob_correct**sample_size + 1e-7) - 1e-7
    up = log(1-prob_success)
    num_samples = up/down
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray, num_samples = 9, prob_success: float = 0.95, threshold: float=0.1
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError(
    #     "`ransac_fundamental_matrix` function in "
    #     + "`ransac.py` needs to be implemented"
    # )
    (n, _) = matches_a.shape
    matches_a_append_1 = np.append(arr=matches_a, values = np.ones((n, 1)), axis = 1)
    matches_b_append_1 = np.append(arr=matches_b, values = np.ones((n, 1)), axis = 1)

    choice = np.random.choice
    matmul = np.matmul
    diag = np.diag
    abs = np.abs
    count_nonzero = np.count_nonzero
    transpose = np.transpose


    best_F = None
    inliers_a = None
    inliers_b = None
    ind_prob_success_best = 0
    for i in range(1000):
      selected_indices = choice(n, num_samples)
      selected_a_append_1 = matches_a_append_1[selected_indices]
      selected_b_append_1 = matches_b_append_1[selected_indices]
      selected_a = matches_a[selected_indices]
      selected_b = matches_b[selected_indices]
      F = estimate_fundamental_matrix(selected_a, selected_b)
      error = abs(diag(matmul(selected_b_append_1, matmul(F,transpose(selected_a_append_1)))))
      success = error<threshold
      ind_prob_success = count_nonzero(success) / num_samples
      if ind_prob_success > ind_prob_success_best:
        best_F = F
        inliers_a = selected_a[success]
        inliers_b = selected_b[success]
        # ind_prob_success_best = ind_prob_success
      if i >= max(calculate_num_ransac_iterations(prob_success, num_samples, ind_prob_success), 10):
        break

      


    # print(prob_success, ind_prob_success_best)
    # print(i, n, num_samples)
    # print(threshold)
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b

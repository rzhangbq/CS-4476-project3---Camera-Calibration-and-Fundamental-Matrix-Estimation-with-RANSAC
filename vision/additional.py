import cv2
import numpy as np
import matplotlib.pyplot as plt
from vision.utils import (
    verify,
    evaluate_points,
    visualize_points,
    visualize_points_image,
    plot3dview,
    load_image,
    draw_epipolar_lines,
    get_matches,
    get_matches_with_ORB,
    show_correspondence2,
)
from vision.part3_ransac import (
    calculate_num_ransac_iterations,
    ransac_fundamental_matrix,
)

def ransac_with_ORB_matching(pic_a = './additional_data/my_image_a.jpeg', 
                          pic_b = './additional_data/my_image_b.jpeg'):
    scale_a = 0.5
    scale_b = 0.5
    n_feat = 4e3
    pic_a = load_image(pic_a)
    pic_b = load_image(pic_b)
    pic_a = cv2.resize(pic_a, None, fx=scale_a, fy=scale_a)
    pic_b = cv2.resize(pic_b, None, fx=scale_b, fy=scale_b)

    points_2d_pic_a, points_2d_pic_b = get_matches_with_ORB(pic_a, pic_b, n_feat)

    F, matched_points_a, matched_points_b = ransac_fundamental_matrix(
        points_2d_pic_a, points_2d_pic_b
    )

    # Draw the epipolar lines on the images and corresponding matches
    match_image = show_correspondence2(
        pic_a,
        pic_b,
        matched_points_a[:, 0],
        matched_points_a[:, 1],
        matched_points_b[:, 0],
        matched_points_b[:, 1],
    )

    return F, pic_a, pic_b, matched_points_a, matched_points_b
def ransac_with_SIFT_matching(pic_a = './additional_data/my_image_a.jpeg', 
                          pic_b = './additional_data/my_image_b.jpeg'):
    scale_a = 0.5
    scale_b = 0.5
    n_feat = 4e3
    pic_a = load_image(pic_a)
    pic_b = load_image(pic_b)
    pic_a = cv2.resize(pic_a, None, fx=scale_a, fy=scale_a)
    pic_b = cv2.resize(pic_b, None, fx=scale_b, fy=scale_b)

    points_2d_pic_a, points_2d_pic_b = get_matches(pic_a, pic_b, n_feat)

    F, matched_points_a, matched_points_b = ransac_fundamental_matrix(
        points_2d_pic_a, points_2d_pic_b
    )

    # Draw the epipolar lines on the images and corresponding matches
    match_image = show_correspondence2(
        pic_a,
        pic_b,
        matched_points_a[:, 0],
        matched_points_a[:, 1],
        matched_points_b[:, 0],
        matched_points_b[:, 1],
    )

    return F, pic_a, pic_b, matched_points_a, matched_points_b
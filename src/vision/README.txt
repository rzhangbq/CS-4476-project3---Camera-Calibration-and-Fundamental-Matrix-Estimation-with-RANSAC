The additional work is in `vision/addtional.py`

API:

def ransac_with_ORB_matching(pic_a, pic_b):
  args: pic_a, pic_b: path of image a and image b
  return: F, pic_a, pic_b, matched_points_a, matched_points_b
def ransac_with_SIFT_matching(pic_a, pic_b):
  args: pic_a, pic_b: path of image a and image b
  return: F, pic_a, pic_b, matched_points_a, matched_points_b


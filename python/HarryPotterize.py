import numpy as np
import cv2

from matchPics import matchPics
from planarH import computeH_ransac, compositeH

DISPLAY = True

"""
Q3.9
Automatically detect the book cover in cv_desk and warp hp_cover onto it.
"""
if __name__ == "__main__":

    # 1) Read images
    cv_cover = cv2.imread('../data/cv_cover.jpg')  
    cv_desk = cv2.imread('../data/cv_desk.png')    
    hp_cover = cv2.imread('../data/hp_cover.jpg')   

    matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

    # 3) Build correspondences (x1 from cv_cover, x2 from cv_desk)
    #    NOTE: matchPics returns (row, col) = (y, x). We want (x, y).
    x1 = []
    x2 = []
    for m in matches:
        idx_cover = m[0]  # index into locs1
        idx_desk  = m[1]  # index into locs2

        row_cover, col_cover = locs1[idx_cover]  # (y, x)
        row_desk,  col_desk  = locs2[idx_desk]   # (y, x)

        x1.append([col_cover, row_cover])  # (x, y) from cover
        x2.append([col_desk,  row_desk])   # (x, y) from desk

    x1 = np.array(x1)
    x2 = np.array(x2)

    H, inliers = computeH_ransac(x1, x2, nSamples=20, threshold=10)

    # 5) Resize hp_cover to the same shape as cv_cover
    resizeShape = (cv_cover.shape[1], cv_cover.shape[0])  # (width, height)
    hp_cover_resized = cv2.resize(hp_cover, dsize=resizeShape)

    # 6) Warp Harry Potter cover onto the desk image
    compositeImg = compositeH(H, hp_cover_resized, cv_desk)
    cv2.imshow("Composite Image", compositeImg)
    cv2.waitKey(0)

    if DISPLAY:
        pts_cover = x1.reshape(-1,1,2).astype(np.float32)
        pts_desk  = x2.reshape(-1,1,2).astype(np.float32)
        H_cv, mask_cv = cv2.findHomography(pts_desk, pts_cover, cv2.RANSAC, 5.0)
        desk_h, desk_w = cv_desk.shape[:2]
        warpedImg_cv = cv2.warpPerspective(hp_cover_resized, np.linalg.inv(H_cv), (desk_w, desk_h))
        cv2.imshow("OpenCV Homography Check", warpedImg_cv)
        cv2.waitKey(0)
